# Submission image: the Halal policy server on port 8000. linux/amd64, CUDA 12.8 base (Blackwell), CPU-only code.
#   bash halal/docker/build.sh v0     builds halal/policy:v0 from the repo root (needs BuildKit)
# Python 3.12 and requirements.txt match the environment the policy is developed and evaluated in.
FROM nvidia/cuda:12.8.0-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip libgomp1 && rm -rf /var/lib/apt/lists/*
COPY halal/docker/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
# third_party is a symlink in our checkouts, so the URDF comes from a named build context (see build.sh).
COPY --from=third_party xr_teleoperate/assets/g1/g1_body29_hand14.urdf third_party/xr_teleoperate/assets/g1/
COPY aaltonomous aaltonomous
COPY halal halal
# Fails the build, not the run, when config, URDF, calibration or an import is broken.
RUN python3 -c "from halal.policy import make_policy; from halal.server import SortServer; SortServer(make_policy())"

EXPOSE 8000
CMD ["python3", "-m", "halal.server"]
