# Aaltonomous Humanoid: 15 minutes to the baseline in sim

You get: Isaac Sim 6.2 + Isaac Lab 3.0, Unitree's G1 sim (`unitree_sim_isaaclab`), our challenge scene
`Aalto-Sort-G129-Dex3-Joint`, the `aaltonomous` package (interface, safety filter, evaluator, oracle baseline).

## 1. Instance

| Need | Why |
|---|---|
| **RTX-class GPU** (RTX Pro 6000 / L40S / A6000 / RTX 4090/5090), >= 24 GB | Cameras are ray-traced. **A100 / H100 have no RT cores: the sim will not render.** |
| NVIDIA driver **>= 580.65.06**, x86_64 Ubuntu 22.04/24.04 | Isaac Sim 6.2 requirement. Check: `nvidia-smi` |
| >= 8 CPU cores, >= 32 GB RAM, >= 80 GB disk | Physics runs on the CPU (~6 cores); image ~30 GB |
| **On-demand, not spot** | Spot instances can disappear within seconds. |
| A **persistent volume** mounted at e.g. `/data` | Checkpoints, datasets, logs. If the instance dies, attach the volume to a new one. Also push your best weights to a private Hugging Face repo. |

Work over SSH inside `tmux` (the sim keeps running when SSH drops).

## 2. Start (Docker, recommended)

```bash
docker load -i /path/to/aalto-sim-v1.tar.zst     # or: docker pull <registry>/aalto-sim:v1  (organisers tell you which)
docker run -d --name aalto --gpus all --network host --ipc host \
    -v aalto-kitcache:/root/miniforge3/envs/isaac/lib/python3.12/site-packages/isaacsim/kit/cache \
    -v /data:/data aalto-sim:v1 sleep infinity
docker exec -it aalto bash                         # all commands below run in here
aalto sim          # starts the sim headless in the background, returns when ready (log: /tmp/aalto_sim.log)
aalto check        # scene checks: should print 8 PASS / 0 FAIL
aalto snap         # one image per camera -> /tmp/aalto_snap (mean ~0 = shader cache still warming, wait)
aalto baseline     # oracle baseline, rung 1, 3 trials -> table + runs/eval/...
```

The code lives in the container at `/opt/aalto` (eval results in `/opt/aalto/runs/`). To build your submission
image on the host you need it there too: `mkdir -p ~/aalto_bundle && docker cp aalto:/opt/aalto/aaltonomous ~/aalto_bundle/`, then build from
`~/aalto_bundle` (`aaltonomous/docs/participants/SUBMISSION.md` §4).

No `nvidia-container-toolkit` on the host (`docker: could not select device driver ""`)? Install it
(`sudo apt-get install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker &&
sudo systemctl restart docker`) or use the bare-metal path below.

The `aalto-kitcache` volume keeps the RTX shader cache. **The first start compiles shaders: ~5-10 min** before
`aalto sim` returns (measured 9 min on a busy RTX 5090 host, 6 CPUs); later starts ~20 s-2 min. One baseline
trial takes ~75 s. Start `aalto sim` first, then read section 3 while it warms up.

## 2b. Bare metal (fallback, ~30-40 min)

```bash
tar xf aalto_bundle.tar.gz && cd aalto_bundle            # contains aaltonomous/ + this file
bash aaltonomous/cloud/setup_instance.sh                  # no sudo; prints the one apt line if libs are missing
bash aaltonomous/cloud/aalto.sh sim                       # then check / snap / baseline as above
```
It installs Miniforge to `~/miniforge3` (env `isaac`) and clones the pinned upstream repos into `third_party/`.
Re-running resumes; `bash aaltonomous/cloud/setup_instance.sh verify` re-checks imports + unit tests.

## 2c. GUI on Verda (VNC over SSH)

Use this path for the interactive Kit viewport in `Aalto-Sort-G129-Dex3-Joint`, or set `TASK=Aalto-Insert-G129-Dex3-Joint` for the lid variant. VNC is bound to loopback and is reachable only through an SSH tunnel. The simulator's host-networked ports stay behind a default-deny firewall.

Install and authenticate the Verda CLI yourself. On your laptop, from the repository root:

```bash
brew install verda-cloud/tap/verda-cli   # macOS; skip if already installed
verda auth login                         # interactive, user-only
make discover
make discover TYPE=<instance-type>
```

Choose an available RTX GPU with >= 24 GB VRAM and a CUDA-enabled Ubuntu 24.04 image. Check the availability, SSH key, and cost estimate printed by discovery. Only after approving those values, create the startup script and instance (200 GB is plenty for the image and initial caches):

```bash
verda --agent startup-script add --name aalto-sim-vnc \
  --file aaltonomous/cloud/verda/bootstrap-instance.sh -o json

verda --agent vm create --kind gpu \
  --instance-type <instance-type> --location <location> \
  --os <image_type> --hostname aalto-sim --ssh-key <ssh-key-id> \
  --startup-script <startup-script-id> --os-volume-size 200 \
  --wait --wait-timeout 5m -o json
```

This uses on-demand billing, not spot. Get the VM's public IP with `verda --agent vm describe <vm-id> -o json`. Check VNC bootstrap with `verda ssh aalto-sim`. If it reports that NVIDIA display modesetting has just been enabled in GRUB, run `systemctl reboot`, reconnect, and rerun `bash aaltonomous/cloud/verda/bootstrap-instance.sh`.

Copy the checkout to the instance from the repository root (no GitHub credentials are needed on the instance):

```bash
rsync -az --exclude='.git/' --exclude='.venv/' --exclude='third_party/' \
  ./ root@<instance-ip>:/opt/aaltonomous-humanoid/
```

Then, on the instance:

```bash
cd /opt/aaltonomous-humanoid
cp .env.example .env
nano .env                       # set CLIENT_CIDR=<your-public-ip>/32
make host-setup                 # cache directories + default-deny UFW
make up                         # build and launch the GUI scene
make logs                       # watch startup / AppReady
make aalto ARGS=check          # scene checks after startup
```

From your laptop, keep this tunnel open while using the GUI:

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 5901:127.0.0.1:5901 root@<instance-ip>
```

Connect a VNC client to `vnc://localhost:5901` (macOS: `open vnc://localhost:5901`). There is no VNC password; the SSH tunnel is the access boundary. First shader compilation can take 5-10 minutes. Stop the simulator with `make down`; stop or delete the VM separately to end its charges.

Useful commands: `make validate` (static checks), `make check` (GPU/Xorg/VNC/container checks), `make ps`, and `make build` (rebuild without starting). If Xorg fails, inspect `journalctl -u isaac-xorg.service` and `/var/log/Xorg.0.log`. NVIDIA display modesetting and a simulated monitor are required on Verda's virtualized GPU; the bootstrap configures both.

## 3. What to use

| You want | Where |
|---|---|
| Challenge brief, rules, interface, submission | `aaltonomous/docs/participants/README.md` |
| The robot interface (obs/actions, 30 Hz, `AaltoEnv(backend="sim")`) | `aaltonomous/interface/env.py`, joint order `interface/joint_spec.yaml` |
| The safety filter your actions go through | `aaltonomous/interface/safety.py` |
| Policy-server template (your submission) | `aaltonomous/deploy/README.md`, `deploy/policy_server.py`, `docker/submission.Dockerfile` |
| Evaluate your server in sim | `aalto eval --policy ws://localhost:8000 --team me --trials 10` |
| Rung 2 (language mapping) | `aalto eval --policy oracle --task sort_language --trials 3` |
| Classical baseline / oracle code | `aaltonomous/baselines/classical/`, `aaltonomous/sim/oracle.py` |
| Domain randomization for training | `AALTO_DR=train aalto sim` (knobs in `aaltonomous/sim/dr/train.yaml`) |
| Scene geometry, success rule | `aaltonomous/scene/scene.yaml`, `scene/scene.py` |

`aalto` without arguments lists all commands; `aalto py script.py` runs Python with the env and paths set.
Ports on the host (with `--network host`): head camera 55555, wrists 55556/55557 (teleimager ZMQ), head RGB-D
55560, scene service 55580; robot state/commands on DDS domain 1 (same topics as the real G1).

Viewing cameras headless: `aalto snap` and `scp` the JPEGs, or run your own viewer on your laptop through an SSH
tunnel (`ssh -L 55555:localhost:55555 ...`). There is no GUI window; you do not need one.

## 4. Troubleshooting

| Symptom | Fix |
|---|---|
| Black camera images | Shader cache is cold (first start). Wait 2-5 min or restart the sim; keep the `aalto-kitcache` volume. |
| `aalto sim` says the sim exited | `tail -50 /tmp/aalto_sim.log`. "port in use": another sim is running (`aalto status`, `aalto stop`). |
| No GPU / Vulkan errors in the log | `nvidia-smi` inside the container must list the GPU; driver >= 580.65.06; not an A100/H100. |
| Evaluate is slow / gate fails on rate (< 27 Hz) | CPU-starved: one sim per instance, don't train on the same cores. |
| Two sims on one machine | `AALTO_INSTANCE=b aalto sim` and the same env var for every client command (own DDS domain + ports). |
| Need to stop | `aalto stop` (never `pkill -f sim_main`, it kills the launching shell too). |

Ignore "Please left-click on the Sim window", `extension.toml doesn't exist`, GLFW and `omni.rtx` warnings.
By running the image you accept the NVIDIA Omniverse / Isaac Sim license (EULA).
