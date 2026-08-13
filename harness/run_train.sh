#!/usr/bin/env bash
# Launch a training run detached in tmux so it survives SSH drops and pod-side
# disconnects. Appends a HARNESS_EXIT=<code> sentinel to the log on exit, win or
# lose -- that sentinel is what the local Monitor watches for, so a crash is as
# visible as a clean finish.
#
# Usage: run_train.sh <session-name> <logfile> <command...>
#   run_train.sh phase2 /workspace/train.log ./isaaclab.sh -p scripts/.../train.py --task ...

set -euo pipefail

SESSION="${1:?session name}"
LOG="${2:?logfile}"
shift 2
[ "$#" -gt 0 ] || { echo "no command given" >&2; exit 2; }

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
  "cd /workspace/IsaacLab && $CMD >>'$LOG' 2>&1; echo \"HARNESS_EXIT=\$?\" >>'$LOG'"

echo "STARTED $SESSION -> $LOG"
