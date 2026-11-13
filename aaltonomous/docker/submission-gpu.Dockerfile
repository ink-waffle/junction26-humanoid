# GPU submission example: torch (CUDA 12.8 wheels, Blackwell / RTX 5090 support) + a policy that refuses to start
# without a GPU and reports it in its metadata name, e.g. "gpu-hold (NVIDIA GeForce RTX 5090, torch 2.x+cu128)".
#
#   docker build -f aaltonomous/docker/submission-gpu.Dockerfile -t myteam/policy-gpu:v1 .
#   docker run --rm --gpus all -p 8000:8000 myteam/policy-gpu:v1      # log line: "CUDA OK: NVIDIA GeForce RTX 5090"
FROM nvidia/cuda:12.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install --no-cache-dir numpy msgpack websockets

WORKDIR /app
COPY aaltonomous/__init__.py aaltonomous/__init__.py
COPY aaltonomous/deploy aaltonomous/deploy
COPY aaltonomous/docker/example_gpu_policy.py /app/policy.py
# COPY weights/ /app/weights/

EXPOSE 8000
CMD ["python3", "/app/policy.py"]
