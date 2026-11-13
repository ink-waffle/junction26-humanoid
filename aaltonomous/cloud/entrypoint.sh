#!/bin/bash
# Container entrypoint: add the NVIDIA Vulkan/EGL ICD files if the container toolkit did not mount them
# (Isaac Sim renders through Vulkan; without the ICD it finds no GPU), then run the command.
if ! ls /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json >/dev/null 2>&1; then
    mkdir -p /etc/vulkan/icd.d
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3"}}' \
        > /etc/vulkan/icd.d/nvidia_icd.json 2>/dev/null || true
fi
if ! ls /usr/share/glvnd/egl_vendor.d/10_nvidia.json >/dev/null 2>&1; then
    mkdir -p /usr/share/glvnd/egl_vendor.d
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json 2>/dev/null || true
fi
exec "$@"
