#!/bin/bash
# push.sh: rsync this checkout to the dev instance (main -> /data/halal, worktree ws/<name> -> /data/wt/<name>); prints the remote path.
set -eo pipefail
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
MAIN="$(git -C "$REPO" worktree list --porcelain | sed -n '1s/^worktree //p')"
set -a; source "$MAIN/.env"; set +a
BRANCH="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
DEST="/data/halal"; [[ "$BRANCH" != main ]] && DEST="/data/wt/${BRANCH##*/}"
SSH=(ssh -i "${VERDA_SSH_KEY/#\~/$HOME}" -o BatchMode=yes)
rsync -az --delete -e "${SSH[*]}" \
    --exclude .git --exclude .venv --exclude third_party --exclude runs --exclude data --exclude weights \
    --exclude '__pycache__' --exclude '*.pyc' --exclude .env --exclude .DS_Store --exclude 'team/SETUP.md' --exclude 'team/notes' \
    "$REPO/" "$VERDA_USER@$VERDA_HOST:$DEST/"
"${SSH[@]}" "$VERDA_USER@$VERDA_HOST" "ln -sfn /data/aaltonomous-humanoid/third_party $DEST/third_party && mkdir -p $DEST/runs"
echo "$DEST"
