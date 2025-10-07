#!/usr/bin/env python3
"""
Generate docker-compose.yml for multi-instance vLLM deployment with nginx load balancer.
"""

import argparse
import sys
from typing import List


def generate_vllm_service(instance_id: int, gpu_list: str, port: int,
                          container_name: str, tensor_parallel_size: int,
                          model_path: str, model_name: str, hf_directory: str,
                          hf_token: str) -> str:
    """Generate a single vLLM service definition."""
    return f"""
  vllm_{instance_id}:
    image: vllm/vllm-openai:latest
    container_name: {container_name}_{instance_id}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['{gpu_list}']
              capabilities: [gpu]
    volumes:
      - {hf_directory}:{hf_directory}
    environment:
      - HUGGING_FACE_HUB_TOKEN={hf_token}
      - VLLM_WORKER_MULTIPROC_METHOD=spawn
      - OMP_NUM_THREADS=16
      - CUDA_VISIBLE_DEVICES={gpu_list}
    ports:
      - "{port}:8000"
    shm_size: '16gb'
    ipc: host
    command: >
      --disable-log-requests
      --trust-remote-code
      --max-model-len=8192
      --gpu-memory-utilization=0.90
      --host 0.0.0.0
      --port 8000
      --tensor-parallel-size {tensor_parallel_size}
      --model {model_path}
      --served-model-name {model_name}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 30
      start_period: 300s
"""


def generate_nginx_service(num_instances: int, nginx_conf_path: str) -> str:
    """Generate nginx load balancer service definition."""
    depends_on = "\n".join([f"      - vllm_{i}" for i in range(num_instances)])

    return f"""
  nginx:
    image: nginx:alpine
    container_name: nginx_lb
    ports:
      - "8080:8080"
    volumes:
      - {nginx_conf_path}:/etc/nginx/nginx.conf:ro
    depends_on:
{depends_on}
"""


def generate_nginx_conf(num_instances: int) -> str:
    """Generate nginx configuration file content."""
    upstream_servers = "\n".join([
        f"        server vllm_{i}:8000;" for i in range(num_instances)
    ])

    return f"""events {{
    worker_connections 4096;
}}

http {{
    upstream vllm_backend {{
{upstream_servers}
    }}

    server {{
        listen 8080;

        location / {{
            proxy_pass http://vllm_backend;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # Increase timeouts for long-running LLM requests
            proxy_connect_timeout 600s;
            proxy_send_timeout 600s;
            proxy_read_timeout 600s;

            # Disable buffering for streaming responses
            proxy_buffering off;
        }}
    }}
}}
"""


def calculate_gpu_list(instance_id: int, gpus_per_instance: int) -> str:
    """Calculate comma-separated GPU list for an instance."""
    start = instance_id * gpus_per_instance
    return ','.join(str(g) for g in range(start, start + gpus_per_instance))


def generate_compose_file(
    num_instances: int,
    tensor_parallel_size: int,
    container_name: str,
    model_path: str,
    model_name: str,
    hf_directory: str,
    hf_token: str,
    nginx_conf_path: str = None
) -> str:
    """Generate complete docker-compose.yml content."""

    compose_content = "version: '3.8'\n\nservices:"

    # Add vLLM service instances
    for i in range(num_instances):
        gpu_list = calculate_gpu_list(i, tensor_parallel_size)
        port = 8000 + i

        compose_content += generate_vllm_service(
            instance_id=i,
            gpu_list=gpu_list,
            port=port,
            container_name=container_name,
            tensor_parallel_size=tensor_parallel_size,
            model_path=model_path,
            model_name=model_name,
            hf_directory=hf_directory,
            hf_token=hf_token
        )

    # Add nginx load balancer if multiple instances
    if num_instances > 1 and nginx_conf_path:
        compose_content += generate_nginx_service(num_instances, nginx_conf_path)

    return compose_content


def main():
    parser = argparse.ArgumentParser(
        description='Generate docker-compose.yml for vLLM multi-instance deployment'
    )
    parser.add_argument('--num-instances', type=int, required=True,
                        help='Number of vLLM instances')
    parser.add_argument('--tensor-parallel-size', type=int, required=True,
                        help='Tensor parallel size (GPUs per instance)')
    parser.add_argument('--container-name', required=True,
                        help='Base container name')
    parser.add_argument('--model-path', required=True,
                        help='Path to model')
    parser.add_argument('--model-name', required=True,
                        help='Model name')
    parser.add_argument('--hf-directory', required=True,
                        help='HuggingFace cache directory')
    parser.add_argument('--hf-token', default='',
                        help='HuggingFace token')
    parser.add_argument('--nginx-conf', default=None,
                        help='Path to nginx config file (for multi-instance)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output docker-compose.yml file path')
    parser.add_argument('--nginx-conf-output', default=None,
                        help='Output nginx.conf file path (for multi-instance)')

    args = parser.parse_args()

    # Generate nginx config if needed
    if args.num_instances > 1:
        if not args.nginx_conf_output:
            print("Error: --nginx-conf-output required for multi-instance setup",
                  file=sys.stderr)
            sys.exit(1)

        nginx_conf_content = generate_nginx_conf(args.num_instances)
        with open(args.nginx_conf_output, 'w') as f:
            f.write(nginx_conf_content)

        # Convert to absolute path for Docker Compose
        import os
        nginx_conf_path = os.path.abspath(args.nginx_conf_output)
    else:
        nginx_conf_path = None

    # Generate docker-compose.yml
    compose_content = generate_compose_file(
        num_instances=args.num_instances,
        tensor_parallel_size=args.tensor_parallel_size,
        container_name=args.container_name,
        model_path=args.model_path,
        model_name=args.model_name,
        hf_directory=args.hf_directory,
        hf_token=args.hf_token,
        nginx_conf_path=nginx_conf_path
    )

    with open(args.output, 'w') as f:
        f.write(compose_content)

    print(f"Generated docker-compose.yml: {args.output}")
    if nginx_conf_path:
        print(f"Generated nginx.conf: {nginx_conf_path}")


if __name__ == '__main__':
    main()
