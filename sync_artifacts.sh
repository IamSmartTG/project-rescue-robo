#!/usr/bin/env bash
# Pull training artifacts off the RunPod pod to this laptop.
# Usage: ./sync_artifacts.sh
#
# If this fails with "Connection refused", the pod restarted and RunPod reassigned
# the SSH port -- update Port in ~/.ssh/config from the console's Connect tab.

set -euo pipefail

DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/artifacts"
REMOTE_LOGS="/workspace/IsaacLab/logs/rsl_rl"

echo "==> checkpoints + tensorboard logs"
rsync -avh --progress \
  --include="*/" \
  --include="*.pt" \
  --include="events.out.tfevents.*" \
  --include="*.yaml" \
  --include="*.json" \
  --exclude="*" \
  runpod:"${REMOTE_LOGS}/" "${DEST}/checkpoints/"

echo "==> videos"
rsync -avh --progress \
  --include="*/" --include="*.mp4" --exclude="*" \
  runpod:"${REMOTE_LOGS}/" "${DEST}/videos/" || echo "(no videos yet)"

echo "==> run logs"
rsync -avh runpod:/workspace/train.log "${DEST}/benchmarks/train.log" || true

echo
echo "Done. Artifacts in: ${DEST}"
du -sh "${DEST}"/* 2>/dev/null || true
