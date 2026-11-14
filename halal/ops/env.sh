# source halal/ops/env.sh [a|b|c|d]: isaac env, this checkout on PYTHONPATH, clients pointed at that sim and kept off the sims' cores.
_halal_root="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
source "${CONDA_ROOT:-$HOME/miniforge3}/etc/profile.d/conda.sh"; conda activate isaac
cd "$_halal_root"; export PYTHONPATH="$_halal_root"
if [ -n "$1" ]; then _halal_env="$(bash "$_halal_root/halal/ops/sims.sh" env "$1")" || return 1; eval "$_halal_env"; fi   # an unknown tag must not fall back to sim a
taskset -cp "$(bash "$_halal_root/halal/ops/sims.sh" client-cores)" $BASHPID >/dev/null
