#!/usr/bin/env bash
# Push the rescue task package, the harness scripts and a checkpoint to a training host.
#
# Runs from the laptop (it is an rsync push, not a remote script). Companion to
# setup_isaac.sh, which builds the stack this drops code into.
#
# The task package goes into IsaacLab's own task tree rather than being pip-installed,
# because isaaclab_tasks/__init__.py calls import_packages() to walk its subpackages and
# register every gym env it finds. Landing the directory there is the whole registration
# step -- no __init__.py anywhere needs editing, which is why `Isaac-Rescue-Anymal-C-v0`
# resolves at all.
#
# Usage:
#   ./deploy_task.sh                                  # code only
#   ./deploy_task.sh --checkpoint <local model_*.pt>  # code + seed a resume checkpoint

set -euo pipefail

HOST="${HOST:-chilly-salmon-lungfish}"
ISAACLAB="${ISAACLAB:-/home/shadeform/isaac/IsaacLab}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TASK_DST="${ISAACLAB}/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c_rescue"

CHECKPOINT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

echo "==> task package -> ${TASK_DST}"
# --delete so a file deleted locally does not linger remotely and keep getting imported.
# Stale __pycache__ is the specific hazard: the tree is imported by directory walk, so an
# orphaned .pyc can keep registering an env whose source is gone.
ssh "${HOST}" "mkdir -p '${TASK_DST}'"
rsync -avh --delete --exclude='__pycache__' \
    "${REPO}/src/anymal_c_rescue/" "${HOST}:${TASK_DST}/"

echo
echo "==> harness -> ${ISAACLAB}/harness"
# Lands beside IsaacLab's own scripts so `./isaaclab.sh -p harness/eval_rescue.py` works with
# IsaacLab as cwd -- which also puts harness/ on sys.path[0], which is how eval_rescue.py
# resolves `import spl`.
rsync -avh --exclude='__pycache__' \
    "${REPO}/harness/" "${HOST}:${ISAACLAB}/harness/"
ssh "${HOST}" "chmod +x '${ISAACLAB}'/harness/*.sh"

if [ -n "${CHECKPOINT}" ]; then
    # Seed the run directory a resume will read from. The run name is preserved so the
    # provenance of a resumed policy stays traceable back to the run that produced it.
    RUN="$(basename "$(dirname "${CHECKPOINT}")")"
    DST="${ISAACLAB}/logs/rsl_rl/anymal_c_rescue/${RUN}"
    echo
    echo "==> checkpoint -> ${DST}/$(basename "${CHECKPOINT}")"
    ssh "${HOST}" "mkdir -p '${DST}'"
    rsync -avh --progress "${CHECKPOINT}" "${HOST}:${DST}/"

    # params/ must travel with it: rsl_rl's resume path reads agent.yaml from the run dir,
    # and a resume into a run directory that has none fails late, after Isaac Sim has spent
    # a minute starting up.
    if [ -d "$(dirname "${CHECKPOINT}")/params" ]; then
        rsync -avh "$(dirname "${CHECKPOINT}")/params/" "${HOST}:${DST}/params/"
    fi
fi

echo
echo "Deployed to ${HOST}."
