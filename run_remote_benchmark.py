#!/usr/bin/env python3
"""
Automated LLM Benchmark Runner

This script reads configuration from config.yaml and runs benchmarks on remote servers via SSH.
It supports multiple servers and models, running all combinations.
It will skip benchmarks if the results file already exists locally.
"""

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"❌ Error: Config file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML config: {e}")
        sys.exit(1)


def validate_config(config: dict) -> None:
    """Validate that required configuration fields are present."""
    # Check for required sections
    if 'benchmark' not in config:
        print("❌ Error: Missing 'benchmark' section in config.")
        sys.exit(1)

    if 'servers' not in config or not config['servers']:
        print("❌ Error: Missing 'servers' section or empty servers list in config.")
        sys.exit(1)

    # Check benchmark fields
    required_benchmark_fields = ['results_file', 'local_results_dir']
    for field in required_benchmark_fields:
        if field not in config['benchmark']:
            print(f"❌ Error: Missing '{field}' in 'benchmark' section.")
            sys.exit(1)

    # Validate server entries
    required_server_fields = ['name', 'address', 'ssh_key', 'models']
    for idx, server in enumerate(config['servers']):
        for field in required_server_fields:
            if field not in server:
                print(f"❌ Error: Missing '{field}' in server entry {idx} ({server.get('name', 'unnamed')}).")
                sys.exit(1)

        # Validate models list
        if not server['models'] or not isinstance(server['models'], list):
            print(f"❌ Error: Server '{server['name']}' must have a non-empty 'models' list.")
            sys.exit(1)


def expand_path(path: str) -> str:
    """Expand user home directory and environment variables in path."""
    return os.path.expanduser(os.path.expandvars(path))


def check_results_exist(server: dict, model_name: str, config: dict) -> bool:
    """Check if benchmark results already exist locally."""
    local_results_dir = Path(expand_path(config['benchmark']['local_results_dir']))
    results_file = config['benchmark']['results_file']
    model_name_safe = model_name.replace('/', '_')
    server_name = server['name']

    # Flat structure with prefix: {server_name}_{model_name}_{results_file}
    prefixed_results_file = f"{server_name}_{model_name_safe}_{results_file}"
    local_results_path = local_results_dir / prefixed_results_file

    return local_results_path.exists()


def run_ssh_command(server: dict, command: str, capture_output: bool = False) -> Optional[str]:
    """Execute a command on the remote server via SSH."""
    ssh_key = expand_path(server['ssh_key'])
    address = server['address']
    port = server.get('port', 22)

    ssh_cmd = [
        'ssh',
        '-i', ssh_key,
        '-p', str(port),
        '-o', 'StrictHostKeyChecking=no',
        address,
        command
    ]

    try:
        if capture_output:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
            return result.stdout
        else:
            subprocess.run(ssh_cmd, check=True)
            return None
    except subprocess.CalledProcessError as e:
        print(f"❌ SSH command failed: {e}")
        if capture_output and e.stderr:
            print(f"Error output: {e.stderr}")
        return None


def setup_remote_repo(server: dict) -> bool:
    """Clone or update the server-benchmark repository on remote server. Returns True if successful."""
    print("📦 Setting up repository on remote server...")

    ssh_key = expand_path(server['ssh_key'])
    address = server['address']
    port = server.get('port', 22)

    # Check if repo exists, clone if not, update if it does
    setup_cmd = (
        'if [ ! -d "server-benchmark" ]; then '
        'git clone https://github.com/cloudrift-ai/server-benchmark.git; '
        'else '
        'cd server-benchmark && git pull; '
        'fi'
    )

    try:
        ssh_cmd = [
            'ssh',
            '-i', ssh_key,
            '-p', str(port),
            '-o', 'StrictHostKeyChecking=no',
            address,
            setup_cmd
        ]
        subprocess.run(ssh_cmd, check=True)
        print("✅ Repository ready on remote server")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to setup repository: {e}")
        return False

    # Run setup script to install dependencies
    print("🔧 Installing dependencies on remote server...")
    setup_script_cmd = 'cd server-benchmark && ./scripts/setup.sh'

    try:
        ssh_cmd = [
            'ssh',
            '-i', ssh_key,
            '-p', str(port),
            '-o', 'StrictHostKeyChecking=no',
            address,
            setup_script_cmd
        ]
        subprocess.run(ssh_cmd, check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def run_benchmark(server: dict, model_name: str, config: dict) -> bool:
    """Run the benchmark on the remote server. Returns True if successful, False otherwise."""
    print(f"🚀 Starting benchmark for model: {model_name}")
    print(f"📡 Connecting to: {server['address']} ({server['name']})")

    # Setup repository on remote server
    if not setup_remote_repo(server):
        return False

    # Get benchmark parameters from config (if any)
    benchmark_params = config.get('benchmark_params', {})
    max_concurrency = benchmark_params.get('max_concurrency', 200)
    num_prompts = benchmark_params.get('num_prompts', 1000)
    random_input_len = benchmark_params.get('random_input_len', 1000)
    random_output_len = benchmark_params.get('random_output_len', 1000)

    # Clean up old marker files
    ssh_key = expand_path(server['ssh_key'])
    address = server['address']
    port = server.get('port', 22)

    cleanup_cmd = [
        'ssh',
        '-i', ssh_key,
        '-p', str(port),
        '-o', 'StrictHostKeyChecking=no',
        address,
        'cd server-benchmark && rm -f *.marker'
    ]
    subprocess.run(cleanup_cmd, capture_output=True)

    # Start the benchmark in background
    print("⏳ Starting benchmark in background...")
    env_vars = (
        f'MODEL_NAME="{model_name}" '
        f'MAX_CONCURRENCY={max_concurrency} '
        f'NUM_PROMPTS={num_prompts} '
        f'RANDOM_INPUT_LEN={random_input_len} '
        f'RANDOM_OUTPUT_LEN={random_output_len}'
    )

    ssh_cmd = [
        'ssh',
        '-i', ssh_key,
        '-p', str(port),
        '-o', 'StrictHostKeyChecking=no',
        address,
        f'cd server-benchmark && nohup bash -c "{env_vars} ./scripts/run_benchmarks.sh" > benchmark.log 2>&1 &'
    ]

    try:
        subprocess.run(ssh_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start benchmark: {e}")
        return False

    # Monitor for marker files and download results as they become available
    print("📊 Monitoring benchmark progress and downloading results as they complete...")

    downloaded = {
        'system_info': False,
        'vllm': False,
        'yabs': False
    }

    system_info_file = config['benchmark'].get('system_info_file', 'system_info.txt')
    results_file = config['benchmark']['results_file']

    # Poll for marker files (check every 30 seconds, timeout after 2 hours)
    max_wait = 7200  # 2 hours
    check_interval = 30
    elapsed = 0

    while elapsed < max_wait:
        # Check system info
        if not downloaded['system_info'] and check_marker_exists(server, 'system_info_ready.marker'):
            print("\n✅ System information ready!")
            download_single_file(server, model_name, config, system_info_file, system_info_file)
            downloaded['system_info'] = True

        # Check vLLM benchmark
        if not downloaded['vllm'] and check_marker_exists(server, 'vllm_benchmark_ready.marker'):
            print("\n✅ vLLM benchmark completed!")
            download_single_file(server, model_name, config, results_file, results_file)
            downloaded['vllm'] = True

        # Check YABS
        if not downloaded['yabs'] and check_marker_exists(server, 'yabs_ready.marker'):
            print("\n✅ YABS benchmark completed!")
            download_single_file(server, model_name, config, 'yabs_results.txt', 'yabs_results.txt')
            downloaded['yabs'] = True

        # If all downloaded, we're done
        if all(downloaded.values()):
            print("\n🎉 All benchmark results downloaded successfully!")
            return True

        # Wait before next check
        time.sleep(check_interval)
        elapsed += check_interval

        # Print progress
        completed = sum(downloaded.values())
        print(f"⏳ Progress: {completed}/3 results downloaded... (elapsed: {elapsed}s)")

    print(f"\n⚠️  Timeout after {max_wait}s. Downloaded {sum(downloaded.values())}/3 results.")
    return sum(downloaded.values()) >= 2  # Consider success if we got at least system info and vLLM


def check_marker_exists(server: dict, marker_file: str) -> bool:
    """Check if a marker file exists on the remote server."""
    ssh_key = expand_path(server['ssh_key'])
    address = server['address']
    port = server.get('port', 22)

    ssh_cmd = [
        'ssh',
        '-i', ssh_key,
        '-p', str(port),
        '-o', 'StrictHostKeyChecking=no',
        address,
        f'[ -f server-benchmark/{marker_file} ] && echo "EXISTS" || echo "NOT_FOUND"'
    ]

    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip() == "EXISTS"
    except subprocess.CalledProcessError:
        return False


def download_single_file(server: dict, model_name: str, config: dict, remote_file: str, local_prefix: str) -> bool:
    """Download a single result file from remote server."""
    ssh_key = expand_path(server['ssh_key'])
    address = server['address']
    port = server.get('port', 22)
    model_name_safe = model_name.replace('/', '_')
    server_name = server['name']

    # Create local results directory
    local_results_dir = Path(expand_path(config['benchmark']['local_results_dir']))
    local_results_dir.mkdir(parents=True, exist_ok=True)

    # Build local and remote paths
    local_file = f"{server_name}_{model_name_safe}_{local_prefix}"
    local_path = local_results_dir / local_file
    remote_path = f"server-benchmark/{remote_file}"

    print(f"📥 Downloading {local_prefix} to: {local_path}")

    scp_cmd = [
        'scp',
        '-i', ssh_key,
        '-P', str(port),
        '-o', 'StrictHostKeyChecking=no',
        f'{address}:{remote_path}',
        str(local_path)
    ]

    try:
        subprocess.run(scp_cmd, check=True)
        print(f"✅ {local_prefix} saved")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Failed to download {local_prefix}: {e}")
        return False




def run_benchmark_combination(server: dict, model: str, config: dict, force: bool = False) -> tuple:
    """Run a single server-model combination benchmark. Returns (server_name, model, success)."""
    server_name = server['name']
    skip_if_exists = config['benchmark'].get('skip_if_exists', True)

    # Check if results already exist
    if skip_if_exists and not force and check_results_exist(server, model, config):
        local_results_dir = Path(expand_path(config['benchmark']['local_results_dir']))
        results_file = config['benchmark']['results_file']
        model_name_safe = model.replace('/', '_')
        prefixed_results_file = f"{server_name}_{model_name_safe}_{results_file}"
        local_results_path = local_results_dir / prefixed_results_file

        print(f"⏭️  Benchmark results already exist: {local_results_path}")
        print("   Skipping...\n")
        return (server_name, model, True)

    # Run benchmark (downloads results incrementally)
    success = run_benchmark(server, model, config)

    return (server_name, model, success)


def run_server_benchmarks(server: dict, models: List[str], config: dict, force: bool = False) -> List[tuple]:
    """Run all benchmarks for a single server sequentially. Returns list of (server_name, model, success)."""
    results = []
    server_name = server['name']

    print(f"\n{'='*80}")
    print(f"Starting benchmarks for server: {server_name}")
    print(f"{'='*80}\n")

    for model in models:
        print(f"\n{'='*80}")
        print(f"Server: {server_name} | Model: {model}")
        print(f"{'='*80}\n")

        result = run_benchmark_combination(server, model, config, force)
        results.append(result)

    print(f"\n{'='*80}")
    print(f"Completed benchmarks for server: {server_name}")
    print(f"{'='*80}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run LLM benchmarks on remote servers via SSH for multiple server-model combinations'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force benchmark even if results already exist'
    )
    parser.add_argument(
        '--server',
        help='Run benchmarks only for a specific server (by name)'
    )
    parser.add_argument(
        '--model',
        help='Run benchmarks only for a specific model'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run benchmarks on different servers in parallel (models on same server run sequentially)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Maximum number of parallel server benchmarks (default: number of servers)'
    )

    args = parser.parse_args()

    # Load and validate configuration
    config = load_config(args.config)
    validate_config(config)

    # Filter servers based on command-line argument
    servers = config['servers']

    if args.server:
        servers = [s for s in servers if s['name'] == args.server]
        if not servers:
            print(f"❌ Error: Server '{args.server}' not found in config.")
            sys.exit(1)

    # Build list of (server, models) tuples for parallel execution
    server_tasks = []
    total_combinations = 0

    for server in servers:
        models = server['models']

        # Filter models if specified
        if args.model:
            models = [m for m in models if m == args.model]
            if not models:
                print(f"⚠️  Warning: Model '{args.model}' not found in server '{server['name']}'. Skipping this server.")
                continue

        server_tasks.append((server, models))
        total_combinations += len(models)

    if not server_tasks:
        print("❌ Error: No server-model combinations to benchmark.")
        sys.exit(1)

    print(f"📊 Running benchmarks for {total_combinations} server-model combination(s) across {len(server_tasks)} server(s)")
    if args.parallel:
        print(f"🚀 Parallel mode: Servers will run in parallel (max workers: {args.max_workers or len(server_tasks)})\n")
    else:
        print("⏳ Sequential mode: Servers will run one at a time\n")

    # Track results
    results = []

    # Run benchmarks - either in parallel or sequentially
    if args.parallel:
        # Parallel execution: each server runs in its own thread
        max_workers = args.max_workers or len(server_tasks)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all server benchmark tasks
            future_to_server = {
                executor.submit(run_server_benchmarks, server, models, config, args.force): server['name']
                for server, models in server_tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_server):
                server_name = future_to_server[future]
                try:
                    server_results = future.result()
                    results.extend(server_results)
                except Exception as exc:
                    print(f"❌ Server {server_name} generated an exception: {exc}")
                    # Add failed results for all models on this server
                    for server, models in server_tasks:
                        if server['name'] == server_name:
                            for model in models:
                                results.append((server_name, model, False))
    else:
        # Sequential execution
        for server, models in server_tasks:
            server_results = run_server_benchmarks(server, models, config, args.force)
            results.extend(server_results)

    # Print summary
    print(f"\n{'='*80}")
    print("📋 SUMMARY")
    print(f"{'='*80}\n")

    successful = [r for r in results if r[2]]
    failed = [r for r in results if not r[2]]

    print(f"✅ Successful: {len(successful)}/{len(results)}")
    if successful:
        for server_name, model, _ in successful:
            print(f"   - {server_name} × {model}")

    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(results)}")
        for server_name, model, _ in failed:
            print(f"   - {server_name} × {model}")

    print("\n🎉 All done!")


if __name__ == '__main__':
    main()
