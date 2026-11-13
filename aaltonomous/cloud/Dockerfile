# Aalto Isaac sim golden image: Isaac Sim 6.2 + Isaac Lab 3.0 + unitree_sim_isaaclab + aaltonomous (M10).
# Same recipe as setup_instance.sh (it runs it, one step per layer, so no layer is > ~11 GB).
#
#   bash aaltonomous/cloud/make_bundle.sh /tmp/aalto_bundle            # context without organiser-only files
#   docker build -t aalto-sim:v1 -f /tmp/aalto_bundle/aaltonomous/cloud/Dockerfile /tmp/aalto_bundle
#   docker run --rm -it --gpus all --network host --ipc host \
#       -v aalto-kitcache:/root/miniforge3/envs/isaac/lib/python3.12/site-packages/isaacsim/kit/cache \
#       -v $HOME/work:/work aalto-sim:v1                              # then: aalto sim && aalto baseline
# Host: NVIDIA driver >= 580.65.06 + nvidia-container-toolkit. The kit cache volume keeps the RTX shader cache
# (first start renders black frames for minutes while it fills).
FROM ubuntu:24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all \
    OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git unzip cmake build-essential iproute2 procps less nano \
        libturbojpeg libglu1-mesa libxt6 libvulkan1 libegl1 libgl1 libxrandr2 libxi6 libsm6 libice6 \
        libxkbcommon0 libglib2.0-0t64 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM base AS full
WORKDIR /opt/aalto
ENV AALTO_NO_GPU=1
# Steps that do not depend on our code first (cached across code updates).
COPY aaltonomous/cloud/setup_instance.sh aaltonomous/cloud/isaac-constraints.txt aaltonomous/cloud/
RUN bash aaltonomous/cloud/setup_instance.sh prereq && bash aaltonomous/cloud/setup_instance.sh conda
RUN bash aaltonomous/cloud/setup_instance.sh torch
RUN bash aaltonomous/cloud/setup_instance.sh isaacsim kit
RUN bash aaltonomous/cloud/setup_instance.sh isaacsim rest
RUN bash aaltonomous/cloud/setup_instance.sh isaaclab
RUN bash aaltonomous/cloud/setup_instance.sh cyclonedds && bash aaltonomous/cloud/setup_instance.sh repos
RUN bash aaltonomous/cloud/setup_instance.sh assets
COPY aaltonomous aaltonomous
COPY QUICKSTART.md .
RUN bash aaltonomous/cloud/setup_instance.sh aalto && bash aaltonomous/cloud/setup_instance.sh verify && \
    ln -s /opt/aalto/aaltonomous/cloud/aalto.sh /usr/local/bin/aalto && \
    echo 'source /root/miniforge3/etc/profile.d/conda.sh && conda activate isaac && export PYTHONPATH=/opt/aalto' >> /root/.bashrc
ENV AALTO_NO_GPU=
ENTRYPOINT ["/opt/aalto/aaltonomous/cloud/entrypoint.sh"]
CMD ["bash"]
