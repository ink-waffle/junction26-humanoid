#!/bin/bash
# Headless unitree_sim_isaaclab, G1 29-DoF + Dex3, on twinflow. Extra args are passed to sim_main.py.
#   bash aaltonomous/sim/run_sim.sh                      # RedBlock pick-place task
#   TASK=Isaac-PickPlace-Cylinder-G129-Dex3-Joint bash aaltonomous/sim/run_sim.sh
#   TASK=Aalto-Sort-G129-Dex3-Joint bash aaltonomous/sim/run_sim.sh     # OUR scene (aalto_task/), rung 1-2
#   TASK=Aalto-Insert-G129-Dex3-Joint bash aaltonomous/sim/run_sim.sh   # + rung-3 lid
#   Aalto-*: mono head 640x480 on 55555, head RGB-D on 55560, scene service on 55580 (scene_client.py),
#            sim clock on 55581 (AaltoEnv paces on it; AALTO_PROFILE=1 prints a per-step time breakdown)
#   AALTO_DR=train|eval_heldout|<yaml>   domain randomization (aalto_task/dr.py; off when unset)
#   AALTO_INSTANCE=b   isolated 2nd sim: DDS domain $AALTO_DDS_DOMAIN (7), cams 56555-7, rgbd 56560, scene 56580;
#                      clients: AALTO_DDS_DOMAIN=7 AALTO_CAM_PORT_BASE=56555 AALTO_SCENE_PORT=56580 AALTO_RGBD_PORT=56560
# DDS domain 1 (same topics as the real robot); cameras on teleimager ZMQ 55555-55557 (WebRTC off).
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate isaac
export OMNI_KIT_ACCEPT_EULA=YES
TASK="${TASK:-Isaac-PickPlace-RedBlock-G129-Dex3-Joint}"
cd "$REPO/third_party/unitree_sim_isaaclab"
if [[ "$TASK" == Aalto-* ]]; then
    export TELEIMAGER_CONFIG="$REPO/aaltonomous/sim/teleimager_aalto.yaml"
    if [[ -n "$AALTO_INSTANCE" ]]; then   # isolated 2nd sim: own DDS domain, shm names, ports (aalto_sim_main.py)
        export AALTO_DDS_DOMAIN="${AALTO_DDS_DOMAIN:-7}" AALTO_CAM_PORT_BASE="${AALTO_CAM_PORT_BASE:-56555}"
        export AALTO_SCENE_PORT="${AALTO_SCENE_PORT:-56580}" AALTO_RGBD_PORT="${AALTO_RGBD_PORT:-56560}"
        cfg="/tmp/teleimager_aalto_${AALTO_INSTANCE}.yaml"
        sed -e "s/zmq_port: 55555/zmq_port: $AALTO_CAM_PORT_BASE/" -e "s/zmq_port: 55556/zmq_port: $((AALTO_CAM_PORT_BASE+1))/" \
            -e "s/zmq_port: 55557/zmq_port: $((AALTO_CAM_PORT_BASE+2))/" \
            -e "s/webrtc_port: 6000\([1-3]\)/webrtc_port: 6100\1/" "$TELEIMAGER_CONFIG" > "$cfg"
        export TELEIMAGER_CONFIG="$cfg"
    fi
    export PYTHONPATH="$REPO:$PYTHONPATH"
    MAIN="$REPO/aaltonomous/sim/aalto_sim_main.py"
    EXTRA=(--camera_include front_camera,left_wrist_camera,right_wrist_camera)
else
    export TELEIMAGER_CONFIG="$REPO/aaltonomous/sim/teleimager_isaacsim_zmq.yaml"
    MAIN=sim_main.py
    EXTRA=()
fi
exec python "$MAIN" --device cpu --task "$TASK" --enable_dex3_dds --robot_type g129 "${EXTRA[@]}" "$@"
