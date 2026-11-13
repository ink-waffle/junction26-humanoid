#!/usr/bin/env bash
set -Eeuo pipefail

# Verda startup script: dedicated Ubuntu 24.04 x86_64 GPU instances only.
# Keep the challenge repository's documented minimum for this Aalto scene.
MINIMUM_DRIVER=580.65.06

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[[ ${EUID} -eq 0 ]] || fail 'Run this script as root.'
[[ $(uname -m) == x86_64 ]] || fail 'Generic ARM instances are unsupported; use x86_64.'

# shellcheck disable=SC1091
source /etc/os-release
[[ ${ID} == ubuntu && ${VERSION_ID} == 24.04 ]] || fail 'Ubuntu 24.04 is required.'
command -v nvidia-smi >/dev/null 2>&1 || fail 'No NVIDIA driver found. Select a CUDA-enabled Verda OS image.'

DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
if [[ $(printf '%s\n' "${MINIMUM_DRIVER}" "${DRIVER_VERSION}" | sort -V | head -n 1) != "${MINIMUM_DRIVER}" ]]; then
  fail "Driver ${DRIVER_VERSION} is older than the Isaac Sim 6.2.0 tested driver ${MINIMUM_DRIVER}."
fi

if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Eq 'A100|H100'; then
  fail 'A100 and H100 have no RT cores and are unsupported by Isaac Sim. Select an RTX GPU.'
fi

printf 'GPU configuration:\n'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

PCI_RAW=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -n 1)
PCI_BUS_FUNCTION=${PCI_RAW#*:}
BUS_HEX=${PCI_BUS_FUNCTION%%:*}
DEVICE_FUNCTION=${PCI_BUS_FUNCTION#*:}
DEVICE_HEX=${DEVICE_FUNCTION%%.*}
FUNCTION_HEX=${DEVICE_FUNCTION##*.}
printf -v GPU_BUS_ID 'PCI:%d:%d:%d' "$((16#${BUS_HEX}))" "$((16#${DEVICE_HEX}))" "$((16#${FUNCTION_HEX}))"

if [[ -f /sys/module/nvidia_drm/parameters/modeset && $(< /sys/module/nvidia_drm/parameters/modeset) == N ]]; then
  for grub_file in /etc/default/grub /etc/default/grub.d/*.cfg; do
    [[ -f ${grub_file} ]] || continue
    sed -i 's/nvidia-drm.modeset=0/nvidia-drm.modeset=1/g' "${grub_file}"
  done
  update-grub
  fail 'Enabled NVIDIA display modesetting in GRUB. Reboot, then run this bootstrap again.'
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl docker-buildx docker-compose-v2 docker.io gnupg iproute2 make \
  openbox python3 rsync ufw x11-utils x11vnc xauth \
  xserver-xorg-core xserver-xorg-input-libinput

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit

nvidia-ctk runtime configure --runtime=docker
systemctl enable docker
systemctl restart docker
docker buildx version
docker compose version

VNC_USER=$(getent passwd 1000 | cut -d: -f1)
[[ -n ${VNC_USER} ]] || fail 'No desktop user with UID 1000 was found.'

install -d -m 0755 /etc/isaac-sim-vnc
cat > /etc/isaac-sim-vnc/xorg.conf <<'EOF'
Section "ServerLayout"
    Identifier     "IsaacSimLayout"
    Screen      0  "IsaacSimScreen" 0 0
EndSection

Section "ServerFlags"
    Option         "AutoAddDevices" "false"
    Option         "AutoEnableDevices" "false"
    Option         "DontVTSwitch" "true"
    Option         "BlankTime" "0"
    Option         "StandbyTime" "0"
    Option         "SuspendTime" "0"
    Option         "OffTime" "0"
EndSection

Section "Module"
    Load           "glx"
EndSection

Section "Monitor"
    Identifier     "IsaacSimMonitor"
    HorizSync       28.0 - 80.0
    VertRefresh     48.0 - 75.0
EndSection

Section "Device"
    Identifier     "NvidiaGPU"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
    Option         "AllowEmptyInitialConfiguration" "true"
    # Virtualized GPUs need a simulated display rather than UseDisplayDevice=none.
    Option         "ConnectedMonitor" "DFP"
    Option         "UseEDID" "False"
EndSection

Section "Screen"
    Identifier     "IsaacSimScreen"
    Device         "NvidiaGPU"
    Monitor        "IsaacSimMonitor"
    DefaultDepth    24
    SubSection     "Display"
        Depth       24
        Virtual     1920 1080
        Modes       "1920x1080"
    EndSubSection
EndSection
EOF

sed -i "/Driver         \"nvidia\"/a\\    BusID         \"${GPU_BUS_ID}\"" /etc/isaac-sim-vnc/xorg.conf

cat > /etc/systemd/system/isaac-xorg.service <<'EOF'
[Unit]
Description=Isaac Sim GPU-backed X display
After=systemd-udev-settle.service

[Service]
Type=simple
ExecStart=/usr/lib/xorg/Xorg :0 -config /etc/isaac-sim-vnc/xorg.conf -nolisten tcp -noreset -ac
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/isaac-openbox.service <<EOF
[Unit]
Description=Isaac Sim window manager
Requires=isaac-xorg.service
After=isaac-xorg.service

[Service]
Type=simple
User=${VNC_USER}
Environment=DISPLAY=:0
ExecStartPre=/bin/sh -c 'i=0; while [ \$\$i -lt 60 ]; do /usr/bin/xdpyinfo -display :0 >/dev/null 2>&1 && exit 0; i=\$\$((i+1)); sleep 1; done; exit 1'
ExecStart=/usr/bin/openbox-session
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/isaac-vnc.service <<EOF
[Unit]
Description=Loopback-only VNC for Isaac Sim
Requires=isaac-xorg.service
After=isaac-xorg.service

[Service]
Type=simple
User=${VNC_USER}
Environment=DISPLAY=:0
ExecStartPre=/bin/sh -c 'i=0; while [ \$\$i -lt 60 ]; do /usr/bin/xdpyinfo -display :0 >/dev/null 2>&1 && exit 0; i=\$\$((i+1)); sleep 1; done; exit 1'
ExecStart=/usr/bin/x11vnc -display :0 -localhost -nopw -forever -shared -rfbport 5901 -xkb -repeat
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable isaac-xorg.service isaac-openbox.service isaac-vnc.service
systemctl restart isaac-xorg.service

for _ in {1..60}; do
  if DISPLAY=:0 xdpyinfo >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! DISPLAY=:0 xdpyinfo >/dev/null 2>&1; then
  journalctl -u isaac-xorg.service -n 100 --no-pager >&2 || true
  fail 'The GPU-backed X display did not start. Inspect /var/log/Xorg.0.log.'
fi

systemctl restart isaac-openbox.service isaac-vnc.service
sleep 2
systemctl is-active --quiet isaac-openbox.service || fail 'Openbox did not start.'
systemctl is-active --quiet isaac-vnc.service || fail 'x11vnc did not start.'
ss -ltn | grep -Eq '127\.0\.0\.1:5901[[:space:]]' \
  || fail 'VNC is not listening on loopback port 5901.'

docker run --rm --runtime=nvidia --gpus all \
  nvcr.io/nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi

printf 'Isaac Sim host prerequisites are ready. VNC is loopback-only on 127.0.0.1:5901.\n'
