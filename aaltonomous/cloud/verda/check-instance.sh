#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

command -v nvidia-smi >/dev/null 2>&1 || fail 'nvidia-smi is unavailable.'
nvidia-smi

for service in isaac-xorg isaac-openbox isaac-vnc; do
  systemctl is-active --quiet "${service}.service" \
    || fail "${service}.service is not active; run: sudo journalctl -u ${service}.service -n 100 --no-pager"
  printf '%s: active\n' "${service}"
done

DISPLAY=:0 xdpyinfo >/dev/null 2>&1 \
  || fail 'X display :0 is unavailable; inspect /var/log/Xorg.0.log.'

ss -ltn | grep -Eq '127\.0\.0\.1:5901[[:space:]]' \
  || fail 'VNC is not listening on 127.0.0.1:5901.'
printf 'VNC: 127.0.0.1:5901 (loopback only)\n'

cd "${PROJECT_DIR}"
docker compose -f compose.verda.yaml config --quiet
docker compose -f compose.verda.yaml ps

printf '\nFrom your computer, open the tunnel with:\n'
printf '  ssh -N -T -o ExitOnForwardFailure=yes -L 5901:127.0.0.1:5901 root@INSTANCE_IP\n'
printf 'Then connect a VNC client to vnc://localhost:5901 (no VNC password).\n'
