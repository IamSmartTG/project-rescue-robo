#!/usr/bin/env bash
# Pull training artifacts off the training host to this laptop.
# Usage: ./sync_artifacts.sh            # Brev (current host)
#        HOST=runpod REMOTE_LOGS=/workspace/IsaacLab/logs/rsl_rl ./sync_artifacts.sh
#
# The RunPod pod is terminated -- RunPod cannot create a Vulkan instance, which kills the
# thermal camera, so training moved to the Brev L40S. `brev login` if ssh says no such host.

set -euo pipefail

DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/artifacts"
HOST="${HOST:-mute-teal-zebra}"
REMOTE_LOGS="${REMOTE_LOGS:-/home/shadeform/isaac/IsaacLab/logs/rsl_rl}"
# Run logs live in the remote home dir, one per stage. Space-separated, remote-side paths.
REMOTE_RUNLOGS="${REMOTE_RUNLOGS:-/home/shadeform/stage1.log /home/shadeform/stage2.log}"

echo "==> checkpoints + tensorboard logs"
rsync -avh --progress \
  --include="*/" \
  --include="*.pt" \
  --include="events.out.tfevents.*" \
  --include="*.yaml" \
  --include="*.json" \
  --exclude="*" \
  "${HOST}":"${REMOTE_LOGS}/" "${DEST}/checkpoints/"

echo "==> videos"
rsync -avh --progress \
  --include="*/" --include="*.mp4" --exclude="*" \
  "${HOST}":"${REMOTE_LOGS}/" "${DEST}/videos/" || echo "(no videos yet)"

echo "==> run logs"
for log in ${REMOTE_RUNLOGS}; do
  rsync -avh "${HOST}":"${log}" "${DEST}/benchmarks/$(basename "${log}")" || true
done

echo
echo "Done. Artifacts in: ${DEST}"
du -sh "${DEST}"/* 2>/dev/null || true
