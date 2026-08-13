#!/usr/bin/env bash
# One command, whole picture. Replaces the usual hunt of "is the pod up / did the
# port move / is training alive / how far in / did it crash".
#
# Fixes the stale-SSH-port problem on the way in rather than making you notice it.

set -uo pipefail

LOG="${LOG:-/workspace/train.log}"

echo "== pod =="
runpod-port || { echo "  (API said no running pod -- it is stopped or terminated)"; exit 1; }

if ! timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=10 runpod 'exit 0' 2>/dev/null; then
  cat <<'EOF'
  SSH refused despite the pod running.
  RunPod wipes /root/.ssh/authorized_keys on restart. Fix from the pod's web terminal:
    mkdir -p ~/.ssh && chmod 700 ~/.ssh
    echo "<contents of ~/.ssh/runpod_ed25519.pub>" >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
  Permanent fix: add that pubkey at RunPod Settings -> SSH Public Keys.
EOF
  exit 1
fi

timeout 45 ssh -o BatchMode=yes runpod "
  echo '== gpu =='
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

  echo '== training =='
  if tmux has-session -t phase2 2>/dev/null; then
    echo '  tmux session phase2: ALIVE'
  else
    echo '  tmux session phase2: not running'
  fi

  # HARNESS_EXIT is stamped by run_train.sh on any exit, so its absence while the
  # tmux session is gone means the run was killed rather than finishing.
  if grep -q 'HARNESS_EXIT=' '$LOG' 2>/dev/null; then
    echo -n '  exited: '; grep -o 'HARNESS_EXIT=[0-9]*' '$LOG' | tail -1
  fi

  # A pipeline ending in tail/head always exits 0, so '|| echo none' never fires.
  # Capture instead and default on empty.
  field() { printf '  %-11s%s\n' \"\$1\" \"\${2:-none}\"; }

  field 'iteration:' \"\$(grep -oE 'Learning iteration [0-9]+/[0-9]+' '$LOG' 2>/dev/null | tail -1)\"
  field 'reward:'    \"\$(grep -E 'Mean reward:' '$LOG' 2>/dev/null | tail -1 | tr -s ' ')\"
  field 'eta:'       \"\$(grep -E 'ETA:' '$LOG' 2>/dev/null | tail -1 | tr -s ' ')\"

  echo '== recent errors =='
  errs=\$(grep -nE 'Traceback|CUDA error|out of memory|Killed|Segmentation fault' '$LOG' 2>/dev/null | tail -3)
  echo \"\${errs:-  none}\"

  echo '== checkpoints =='
  ckpts=\$(ls -1t /workspace/IsaacLab/logs/rsl_rl/anymal_c_rough/*/model_*.pt 2>/dev/null | head -3)
  echo \"\${ckpts:-  none}\"
"
