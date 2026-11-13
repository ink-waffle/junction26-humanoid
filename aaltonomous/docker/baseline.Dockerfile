# Reference submission: the classical rung-1 baseline (aaltonomous/baselines/classical) as a policy server.
# CPU only (HSV + depth perception, Pinocchio IK); runs the same with or without --gpus.
#
#   docker build -f aaltonomous/docker/baseline.Dockerfile -t aalto/baseline-classical:v1 .     # from the repo root
#   docker run --rm -p 8000:8000 aalto/baseline-classical:v1
#   python -m aaltonomous.deploy.replay --synthetic 100 --container aalto/baseline-classical:v1 --gpus none
#
# Needs third_party/xr_teleoperate (URDF for IK + the planner's safety pre-check): scripts/setup_host.sh or
# cloud/setup_instance.sh clone it. Calibration (camera intrinsics/extrinsics) = interface/calibration.yaml at build
# time: rebuild after recalibrating. Build context: the root .dockerignore (legacy builder) or
# baseline.Dockerfile.dockerignore (BuildKit) keep it to a few MB.
FROM nvidia/cuda:12.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip libgomp1 && rm -rf /var/lib/apt/lists/*
# pin = the Pinocchio wheel. The conda env `aalto` has pinocchio 3.1.0 + numpy 1.26, but 3.1.0 is not on PyPI and the
# pin 3.3.1 wheel needs numpy >= 2.2 (cmeel-boost), so the image runs numpy 2.2 (tested: same plans/results).
RUN pip3 install --no-cache-dir numpy==2.2.6 scipy==1.15.2 opencv-python-headless==4.10.0.84 pyyaml \
        msgpack "websockets>=12" pin==3.3.1

WORKDIR /app
COPY third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf third_party/xr_teleoperate/assets/g1/
COPY aaltonomous/__init__.py aaltonomous/
COPY aaltonomous/deploy/__init__.py aaltonomous/deploy/protocol.py aaltonomous/deploy/policy_server.py aaltonomous/deploy/
COPY aaltonomous/interface/__init__.py aaltonomous/interface/safety.py aaltonomous/interface/calibration.py \
     aaltonomous/interface/joint_spec.yaml aaltonomous/interface/calibration.yaml aaltonomous/interface/
COPY aaltonomous/scene/__init__.py aaltonomous/scene/scene.py aaltonomous/scene/scene.yaml aaltonomous/scene/
COPY aaltonomous/baselines/__init__.py aaltonomous/baselines/
COPY aaltonomous/baselines/classical/*.py aaltonomous/baselines/classical/config.yaml aaltonomous/baselines/classical/
# Load-time check at build: config, URDF and calibration parse (fails the build, not the run).
RUN python3 -c "from aaltonomous.baselines.classical.policy_server import ClassicalServerPolicy; ClassicalServerPolicy()"

EXPOSE 8000
CMD ["python3", "-m", "aaltonomous.baselines.classical.policy_server"]
