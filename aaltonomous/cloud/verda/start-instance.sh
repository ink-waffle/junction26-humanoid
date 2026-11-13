#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
CLIENT_CIDR_FILE=/etc/isaac-sim-vnc/client-cidr

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[[ ${EUID} -eq 0 ]] || fail 'Run with sudo: it starts host-networked services without public authentication.'
[[ -f ${PROJECT_DIR}/.env ]] || fail 'Create .env and run setup-instance.sh before starting the simulator.'

set -a
# shellcheck disable=SC1091
source "${PROJECT_DIR}/.env"
set +a

systemctl is-active --quiet ufw.service \
  || fail 'UFW is inactive. Run setup-instance.sh before starting host-networked services.'
ufw status verbose | grep -Eq 'Default: deny \(incoming\)' \
  || fail 'UFW must deny unsolicited inbound traffic before starting host-networked services.'
[[ -f ${CLIENT_CIDR_FILE} ]] || fail 'The managed SSH CIDR is missing. Run setup-instance.sh.'
[[ $(< "${CLIENT_CIDR_FILE}") == "${CLIENT_CIDR:-}" ]] \
  || fail 'CLIENT_CIDR differs from the active managed rule. Re-run setup-instance.sh first.'

for service in isaac-xorg isaac-openbox isaac-vnc; do
  systemctl is-active --quiet "${service}.service" \
    || fail "${service}.service is not active; inspect its journal before starting Isaac Sim."
done

cd "${PROJECT_DIR}"
exec docker compose -f compose.verda.yaml up --build -d
