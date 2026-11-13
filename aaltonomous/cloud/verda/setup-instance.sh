#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
DATA_ROOT=/opt/aalto-verda/data
CLIENT_CIDR_FILE=/etc/isaac-sim-vnc/client-cidr

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[[ ${EUID} -eq 0 ]] || fail 'Run with sudo from the project directory.'
[[ -f ${PROJECT_DIR}/.env ]] || fail "Copy ${PROJECT_DIR}/.env.example to ${PROJECT_DIR}/.env and set CLIENT_CIDR."

set -a
# shellcheck disable=SC1091
source "${PROJECT_DIR}/.env"
set +a

[[ -n ${CLIENT_CIDR:-} ]] || fail 'Set CLIENT_CIDR to the client public IP/32 before enabling UFW.'
[[ ${AALTO_HOST_DATA:-${DATA_ROOT}} == "${DATA_ROOT}" ]] \
  || fail "This setup script requires AALTO_HOST_DATA=${DATA_ROOT}."

python3 - "${CLIENT_CIDR}" <<'PY'
import ipaddress
import sys

network = ipaddress.ip_network(sys.argv[1], strict=False)
if network.prefixlen == 0:
    raise SystemExit("CLIENT_CIDR must not allow the entire Internet")
PY

for service in isaac-xorg isaac-openbox isaac-vnc; do
  systemctl is-active --quiet "${service}.service" \
    || fail "${service}.service is not active; finish bootstrap and inspect its journal first."
done

install -d -m 0755 "${DATA_ROOT}"
install -d -m 0755 \
  "${DATA_ROOT}/kit-cache" \
  "${DATA_ROOT}/work" \
  "${DATA_ROOT}/runs"

ufw default deny incoming
ufw default allow outgoing
if [[ -f ${CLIENT_CIDR_FILE} ]]; then
  PREVIOUS_CLIENT_CIDR=$(< "${CLIENT_CIDR_FILE}")
  if [[ ${PREVIOUS_CLIENT_CIDR} != "${CLIENT_CIDR}" ]]; then
    ufw delete allow from "${PREVIOUS_CLIENT_CIDR}" to any port 22 proto tcp || true
  fi
fi
ufw allow from "${CLIENT_CIDR}" to any port 22 proto tcp comment 'Isaac Sim admin SSH'
printf '%s\n' "${CLIENT_CIDR}" > "${CLIENT_CIDR_FILE}"
# Do not open 5901 or the simulator service ports: only SSH needs inbound access.
ufw --force enable

printf 'Aalto data directories are ready. SSH is restricted to %s; VNC has no public firewall rule.\n' "${CLIENT_CIDR}"
printf 'Next: cd %s && sudo docker compose -f compose.verda.yaml up --build -d\n' "${PROJECT_DIR}"
