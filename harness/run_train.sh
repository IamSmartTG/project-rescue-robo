#!/usr/bin/env bash
# Launch a training run detached in tmux so it survives SSH drops and pod-side
# disconnects. Appends a HARNESS_EXIT=<code> sentinel to the log on exit, win or
# lose -- that sentinel is what the local Monitor watches for, so a crash is as
# visible as a clean finish.
#
# Usage: run_train.sh <session-name> <logfile> <command...>
#   run_train.sh phase2 /workspace/train.log ./isaaclab.sh -p scripts/.../train.py --task ...

set -euo pipefail

# Working directory for the launched command. Defaults to the RunPod Isaac Lab checkout;
# override for any other host (the Brev L40S keeps its install under ~/isaac/IsaacLab).
# This used to be hardcoded, which silently turned every run on a different box into an
# instant HARNESS_EXIT=1 -- the `cd` failed and && skipped the command entirely.
WORKDIR="${WORKDIR:-/workspace/IsaacLab}"

SESSION="${1:?session name}"
LOG="${2:?logfile}"
shift 2
[ "$#" -gt 0 ] || { echo "no command given" >&2; exit 2; }
[ -d "$WORKDIR" ] || { echo "WORKDIR does not exist: $WORKDIR" >&2; exit 2; }

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ALREADY_RUNNING $SESSION (tmux kill-session -t $SESSION to replace)"
  exit 0
fi

mkdir -p "$(dirname "$LOG")"
# Keep the old log rather than clobbering it -- prior runs are the only record of
# what a crashed run was doing.
[ -f "$LOG" ] && mv "$LOG" "$LOG.$(date +%Y%m%d-%H%M%S)"

printf -v CMD '%q ' "$@"
tmux new-session -d -s "$SESSION" \
  "cd '$WORKDIR' && $CMD >>'$LOG' 2>&1; echo \"HARNESS_EXIT=\$?\" >>'$LOG'"

echo "STARTED $SESSION -> $LOG"
