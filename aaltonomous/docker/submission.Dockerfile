# Submission base image (spec §8): x86_64, CUDA >= 12.8 (RTX 5090 = Blackwell), starts the policy server on port 8000
# in < 120 s, weights inside the image, fits in 32 GB VRAM.
#
# Build from the repo root:  docker build -f aaltonomous/docker/submission.Dockerfile -t myteam/policy:v1 .
# Test locally:              docker run --rm --gpus all -p 8000:8000 myteam/policy:v1
#                            python -m aaltonomous.deploy.replay --synthetic 100 --policy ws://localhost:8000
# GPU / torch example:       aaltonomous/docker/submission-gpu.Dockerfile
#
# `base` has the CUDA driver plumbing only; pip torch/jax wheels bring their own CUDA + cuDNN. If you need the
# system CUDA/cuDNN libraries (TensorRT, onnxruntime-gpu, custom kernels), use
# nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04 (+~3 GB) or a -devel tag to compile.
FROM nvidia/cuda:12.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir numpy msgpack websockets
# Your own dependencies, e.g. torch with CUDA 12.8 (see submission-gpu.Dockerfile):
#   RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app
COPY aaltonomous/__init__.py aaltonomous/__init__.py
COPY aaltonomous/deploy aaltonomous/deploy
# Your policy and weights (weights go INSIDE the image, no downloads at startup):
COPY aaltonomous/docker/example_policy.py /app/policy.py
# COPY weights/ /app/weights/

EXPOSE 8000
CMD ["python3", "/app/policy.py"]
