#!/bin/bash
# Install the Aalto sim stack (Isaac Sim 6.2 + Isaac Lab 3.0 + unitree_sim_isaaclab + aaltonomous) with no sudo.
# Same recipe and pins as the working env `isaac` on twinflow (docs/ISAAC_TWINFLOW.md); the Dockerfile runs it too.
#
#   bash aaltonomous/cloud/setup_instance.sh            # all steps; re-run resumes (done steps are skipped)
#   bash aaltonomous/cloud/setup_instance.sh assets     # one step (see STEPS below); FORCE=1 redoes it
#
# Needs: x86_64 Ubuntu 22.04/24.04, NVIDIA driver >= 580.65.06 (RTX-class GPU), ~40 GB free disk, and
# apt packages (checked in step `prereq`, prints the one sudo line if something is missing).
# Installs into $CONDA_ROOT (default ~/miniforge3, env `isaac`) and <repo>/third_party/ (clones at pinned revs).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
TP="$REPO/third_party"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
ENV_NAME=isaac

# ---- pins (= twinflow 2026-11-10/11) ----
ISAACLAB_REV=e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4        # v3.0.0
SIM_REV=c0d2e68eb87b1d44a10f957c140c947bdddaa0ff             # unitree_sim_isaaclab main
TELEIMAGER_REV=8a8971aa3ba09bd60cc7aee7fa3c71353cc98e16      # its submodule
SDK2PY_REV=814556d15970dd2ecf1c9984e845ca02ab07e206
CYCLONE_REV=5041f3560c088c99e5088b2b8520b69169621196         # releases/0.10.x (C lib 0.10.5)
XRTELEOP_REV=817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b        # only for the G1 URDF (safety filter, IK)
ASSETS_REV=55e40ff4705e1c6b26831862a6227839ac0e10d1         # HF dataset unitree_sim_isaaclab_usds
export PIP_CONSTRAINT="$HERE/isaac-constraints.txt"           # pip freeze of the working env
export PIP_NO_CACHE_DIR=1 OMNI_KIT_ACCEPT_EULA=YES
export CYCLONEDDS_HOME="$TP/cyclonedds/install"

STEPS=(prereq conda torch isaacsim isaaclab cyclonedds repos assets aalto verify)
mkdir -p "$TP"

say() { echo -e "\n=== [setup $(date +%H:%M:%S)] $*"; }

activate() {
    # shellcheck disable=SC1091
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
}

clone_at() {  # clone_at <url> <dir> <sha>: shallow fetch of exactly one commit
    local url=$1 dir=$2 sha=$3
    if [[ -d "$dir/.git" ]] && [[ "$(git -C "$dir" rev-parse HEAD)" == "$sha" ]]; then return; fi
    rm -rf "$dir"; git init -q "$dir"
    git -C "$dir" remote add origin "$url"
    git -C "$dir" fetch -q --depth 1 origin "$sha"
    git -C "$dir" -c advice.detachedHead=false checkout -q FETCH_HEAD
}

step_prereq() {
    local miss=() cmd lib ldc
    ldc=$(ldconfig -p 2>/dev/null || /sbin/ldconfig -p)
    [[ "$(uname -m)" == x86_64 ]] || { echo "x86_64 only"; exit 1; }
    for cmd in git curl unzip cmake make gcc g++; do command -v $cmd >/dev/null || miss+=("$cmd"); done
    for lib in libturbojpeg.so.0 libGLU.so.1 libXt.so.6 libvulkan.so.1 libEGL.so.1 libGL.so.1 libXrandr.so.2 libXi.so.6 libSM.so.6 libICE.so.6 libxkbcommon.so.0 libglib-2.0.so.0; do
        grep -qF "$lib (" <<<"$ldc" || miss+=("$lib")
    done
    if ((${#miss[@]})); then
        echo "Missing: ${miss[*]}"
        echo "Run:  sudo apt-get update && sudo apt-get install -y --no-install-recommends git curl unzip cmake build-essential \\"
        echo "        libturbojpeg libglu1-mesa libxt6 libvulkan1 libegl1 libgl1 libxrandr2 libxi6 libsm6 libice6 libxkbcommon0 libglib2.0-0"
        echo "(Ubuntu 24.04: libglib2.0-0t64)"
        exit 1
    fi
    if [[ -z "${AALTO_NO_GPU:-}" ]]; then
        command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: install the NVIDIA driver (>= 580.65.06)"; exit 1; }
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        local drv; drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        [[ "$(printf '%s\n' 580.65.06 "$drv" | sort -V | head -1)" == 580.65.06 ]] \
            || { echo "driver $drv < 580.65.06 (Isaac Sim 6.2 needs >= 580.65.06)"; exit 1; }
    fi
    local free; free=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc 0-9)
    ((free >= 40)) || echo "WARNING: only ${free} GB free here; the install needs ~40 GB"
}

step_conda() {
    if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
        curl -fsSL -o /tmp/miniforge.sh \
            https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
        bash /tmp/miniforge.sh -b -p "$CONDA_ROOT"; rm -f /tmp/miniforge.sh
    fi
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda env list | grep -q "^$ENV_NAME " || conda create -y -n "$ENV_NAME" -c conda-forge python=3.12 libstdcxx-ng
    conda env config vars set -n "$ENV_NAME" OMNI_KIT_ACCEPT_EULA=YES CYCLONEDDS_HOME="$CYCLONEDDS_HOME" >/dev/null
    activate
    python -m pip install -q --upgrade pip
}

step_torch() {
    activate
    pip install torch==2.11.0 torchvision==0.26.0 \
        --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple
}

step_isaacsim() {
    activate
    # Isaac Lab 3 uses newer click/urllib3. Apply Isaac Sim's exact pins only
    # during its own resolver, then let Isaac Lab install its required versions.
    local isaacsim_constraints="$TP/constraints-isaacsim${1:+_$1}.txt"
    grep -q '^urllib3==' "$PIP_CONSTRAINT"
    sed -e 's/^urllib3==.*/urllib3==2.7.0/' -e 's/^click==.*/click==8.3.3/' \
      "$PIP_CONSTRAINT" > "$isaacsim_constraints"
    # The 10.5 GB extension cache first, as its own step (= its own image layer in the Dockerfile).
    if [[ "${1:-}" != rest ]]; then
        PIP_CONSTRAINT="$isaacsim_constraints" pip install "isaacsim-extscache-kit==6.2.0.0" --extra-index-url https://pypi.nvidia.com
    fi
    if [[ "${1:-}" != kit ]]; then
        PIP_CONSTRAINT="$isaacsim_constraints" pip install "isaacsim[all,extscache]==6.2.0.0" --extra-index-url https://pypi.nvidia.com
    fi
}

step_isaaclab() {
    activate
    clone_at https://github.com/isaac-sim/IsaacLab.git "$TP/IsaacLab" "$ISAACLAB_REV"
    (cd "$TP/IsaacLab" && ./isaaclab.sh --install)       # default = all (as on twinflow)
}

step_cyclonedds() {
    clone_at https://github.com/eclipse-cyclonedds/cyclonedds "$TP/cyclonedds" "$CYCLONE_REV"
    (cd "$TP/cyclonedds" && mkdir -p build install && cd build &&
     cmake .. -DCMAKE_INSTALL_PREFIX=../install -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF >/dev/null &&
     cmake --build . --target install -j"$(nproc)" >/dev/null)
}

step_repos() {
    activate
    clone_at https://github.com/unitreerobotics/unitree_sdk2_python.git "$TP/unitree_sdk2_python" "$SDK2PY_REV"
    pip install -e "$TP/unitree_sdk2_python"
    clone_at https://github.com/unitreerobotics/unitree_sim_isaaclab.git "$TP/unitree_sim_isaaclab" "$SIM_REV"
    clone_at https://github.com/unitreerobotics/teleimager "$TP/unitree_sim_isaaclab/teleimager" "$TELEIMAGER_REV"
    (cd "$TP/unitree_sim_isaaclab" && pip install -r requirements.txt && pip install -e "./teleimager[server]")
    pip install "PyTurboJPEG<2"                            # Ubuntu's libjpeg-turbo is 2.x
    clone_at https://github.com/unitreerobotics/xr_teleoperate.git "$TP/xr_teleoperate" "$XRTELEOP_REV"
}

step_assets() {
    local d="$TP/unitree_sim_isaaclab"
    [[ -d "$d/assets/robots" ]] && [[ -z "${FORCE:-}" ]] && return
    local tmp; tmp=$(mktemp -d -p "$TP")
    curl -fL --retry 3 -o "$tmp/assets.zip" \
        "https://huggingface.co/datasets/unitreerobotics/unitree_sim_isaaclab_usds/resolve/$ASSETS_REV/assets.zip"
    (cd "$tmp" && unzip -oq assets.zip)                   # -o: the zip has duplicate entries
    rm -rf "$d/assets"; mv "$tmp/assets" "$d/assets"; rm -rf "$tmp"
}

step_aalto() {
    activate
    pip install msgpack websockets                         # deploy protocol (versions from the constraints)
}

step_verify() {
    activate
    cd "$REPO"
    python -c "import isaacsim, isaaclab, unitree_sdk2py, teleimager.client, pinocchio, msgpack, websockets, trimesh, cv2; print('imports OK')"
    PYTHONPATH="$REPO" python -m unittest aaltonomous/interface/test_safety.py aaltonomous/interface/test_env.py \
        aaltonomous/scene/test_scene.py aaltonomous/deploy/test_deploy.py aaltonomous/sim/aalto_task/test_layout.py
}

run_step() {
    local s=$1; shift
    local mark="$TP/.setup_done_$s${1:+_$1}"
    if [[ -f "$mark" && -z "${FORCE:-}" && $s != verify && $s != prereq ]]; then echo "skip $s (done)"; return; fi
    say "$s"; local t0=$SECONDS
    "step_$s" "$@"
    echo "$s: $((SECONDS - t0)) s" | tee -a "$TP/.setup_times"
    touch "$mark"
}

if (($#)); then run_step "$@"; else for s in "${STEPS[@]}"; do run_step "$s"; done; fi
say "done. Next: bash aaltonomous/cloud/aalto.sh sim"
