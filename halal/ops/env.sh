# source halal/ops/env.sh [a|b|c]: isaac env, this checkout on PYTHONPATH, evaluator clients pointed at the given sim.
_halal_root="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
source "${CONDA_ROOT:-$HOME/miniforge3}/etc/profile.d/conda.sh"; conda activate isaac
cd "$_halal_root"; export PYTHONPATH="$_halal_root"
[ -n "$1" ] && eval "$(bash "$_halal_root/halal/ops/sims.sh" env "$1")"
