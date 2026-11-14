#!/bin/bash
# publish.sh <version>: copy halal/policy:<version> from the dev instance and push it to GHCR from this machine; prints the digest to hand in.
set -eo pipefail
VERSION="${1:?usage: publish.sh <version>}"
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
MAIN="$(git -C "$REPO" worktree list --porcelain | sed -n '1s/^worktree //p')"
set -a; source "$MAIN/.env"; set +a
SSH=(ssh -i "${VERDA_SSH_KEY/#\~/$HOME}" -o BatchMode=yes)
OWNER="$(gh api user --jq .login)"
TARGET="ghcr.io/$OWNER/halal-policy:$VERSION"
ARCHIVE="halal_policy_$VERSION.tar.zst"
mkdir -p "$MAIN/runs/images"

"${SSH[@]}" "$VERDA_USER@$VERDA_HOST" "docker save halal/policy:$VERSION | zstd -T4 -3 -q -f -o /data/scratch/$ARCHIVE"
rsync -a -e "${SSH[*]}" "$VERDA_USER@$VERDA_HOST:/data/scratch/$ARCHIVE" "$MAIN/runs/images/"
docker load -i "$MAIN/runs/images/$ARCHIVE"
docker tag "halal/policy:$VERSION" "$TARGET"
gh auth token | docker login ghcr.io -u "$OWNER" --password-stdin >/dev/null
trap 'docker logout ghcr.io >/dev/null' EXIT   # the token never stays in the Docker config
docker push "$TARGET"
echo "hand in: ghcr.io/$OWNER/halal-policy@$(docker image inspect "$TARGET" --format '{{.Id}}')"
