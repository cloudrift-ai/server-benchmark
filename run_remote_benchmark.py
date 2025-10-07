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
from pathlib import Path
from typing import Optional, List, Dict

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


def run_benchmark(server: dict, model_name: str) -> bool:
    """Run the benchmark on the remote server. Returns True if successful, False otherwise."""
    print(f"🚀 Starting benchmark for model: {model_name}")
    print(f"📡 Connecting to: {server['address']} ({server['name']})")

    # Update the model name in the remote script
    print("📝 Updating model configuration on remote server...")
    update_model_cmd = f'cd server-benchmark && sed -i "s|^MODEL_NAME=.*|MODEL_NAME=\\"{model_name}\\"|" scripts/run_vllm_benchmark.sh'
    result = run_ssh_command(server, update_model_cmd)
    if result is None and not isinstance(result, str):
        # Check if command succeeded (when capture_output=False, result is None on success)
        pass

    # Run the benchmark
    print("⏳ Running benchmark (this may take a while)...")
    try:
        ssh_key = expand_path(server['ssh_key'])
        address = server['address']
        port = server.get('port', 22)

        ssh_cmd = [
            'ssh',
            '-i', ssh_key,
            '-p', str(port),
            '-o', 'StrictHostKeyChecking=no',
            address,
            'cd server-benchmark && ./scripts/run_benchmarks.sh'
        ]
        subprocess.run(ssh_cmd, check=True)
        print("✅ Benchmark completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Benchmark failed: {e}")
        return False


def download_results(server: dict, model_name: str, config: dict) -> bool:
    """Download benchmark results from the remote server. Returns True if successful."""
    ssh_key = expand_path(server['ssh_key'])
    address = server['address']
    port = server.get('port', 22)
    results_file = config['benchmark']['results_file']
    system_info_file = config['benchmark'].get('system_info_file', 'system_info.txt')
    model_name_safe = model_name.replace('/', '_')
    server_name = server['name']

    # Create local results directory (flat structure)
    local_results_dir = Path(expand_path(config['benchmark']['local_results_dir']))
    local_results_dir.mkdir(parents=True, exist_ok=True)

    # Download benchmark results with prefix
    prefixed_results_file = f"{server_name}_{model_name_safe}_{results_file}"
    local_results_path = local_results_dir / prefixed_results_file
    remote_results_path = f"server-benchmark/{results_file}"

    print(f"📥 Downloading benchmark results to: {local_results_path}")

    scp_cmd = [
        'scp',
        '-i', ssh_key,
        '-P', str(port),
        '-o', 'StrictHostKeyChecking=no',
        f'{address}:{remote_results_path}',
        str(local_results_path)
    ]

    try:
        subprocess.run(scp_cmd, check=True)
        print(f"✅ Benchmark results saved to: {local_results_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download benchmark results: {e}")
        return False

    # Download system info with prefix
    prefixed_system_info_file = f"{server_name}_{model_name_safe}_{system_info_file}"
    local_system_info_path = local_results_dir / prefixed_system_info_file
    remote_system_info_path = f"server-benchmark/{system_info_file}"

    print(f"📥 Downloading system info to: {local_system_info_path}")

    scp_cmd = [
        'scp',
        '-i', ssh_key,
        '-P', str(port),
        '-o', 'StrictHostKeyChecking=no',
        f'{address}:{remote_system_info_path}',
        str(local_system_info_path)
    ]

    try:
        subprocess.run(scp_cmd, check=True)
        print(f"✅ System info saved to: {local_system_info_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Failed to download system info: {e}")
        # Don't fail the entire operation if system info download fails
        return True


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

    # Run benchmark and download results
    success = run_benchmark(server, model)
    if success:
        success = download_results(server, model, config)

    return (server_name, model, success)


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

    # Build list of (server, model) combinations
    combinations = []
    for server in servers:
        models = server['models']

        # Filter models if specified
        if args.model:
            models = [m for m in models if m == args.model]
            if not models:
                print(f"⚠️  Warning: Model '{args.model}' not found in server '{server['name']}'. Skipping this server.")
                continue

        for model in models:
            combinations.append((server, model))

    if not combinations:
        print("❌ Error: No server-model combinations to benchmark.")
        sys.exit(1)

    print(f"📊 Running benchmarks for {len(combinations)} server-model combination(s)\n")

    # Track results
    results = []

    # Run benchmarks for all combinations
    for server, model in combinations:
        print(f"\n{'='*80}")
        print(f"Server: {server['name']} | Model: {model}")
        print(f"{'='*80}\n")

        result = run_benchmark_combination(server, model, config, args.force)
        results.append(result)

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
