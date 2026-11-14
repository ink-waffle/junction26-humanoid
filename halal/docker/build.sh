#!/bin/bash
# build.sh <version> [platform]: build the submission image halal/policy:<version> from this checkout (default linux/amd64).
set -eo pipefail
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
docker buildx build --platform "${2:-linux/amd64}" --load -f "$REPO/halal/docker/Dockerfile" \
    --build-context third_party="$(readlink -f "$REPO/third_party")" -t "halal/policy:${1:?usage: build.sh <version> [platform]}" "$REPO"
