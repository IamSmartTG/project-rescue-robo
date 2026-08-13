#!/usr/bin/env bash
# Stage 2 of two-stage training: resume the stage-1 locomotion policy and learn search
# on top of it.
#
# The observation space is identical between stages (proprioception + height scan + LiDAR +
# thermal), which is the whole reason stage 1 keeps the sensors attached -- the checkpoint
# loads straight in. Only the victim reward weights change (0 -> live), plus the camera-FOV
# detection gate that makes the robot turn and scan rather than walk at a victim it was
# always able to see.
#
# max_iterations is additive on resume: rsl_rl computes total_it = start_it + n, so 1200
# here means stage1_end -> stage1_end + 1200.

set -euo pipefail

export OMNI_KIT_ACCEPT_EULA=YES
source ~/isaac/venv/bin/activate
cd ~/isaac/IsaacLab

# Both stages share experiment_name "anymal_c_rescue", so the newest run dir is stage 1's.
RUN=$(ls -1t logs/rsl_rl/anymal_c_rescue/ | head -1)
CKPT=$(ls -1t "logs/rsl_rl/anymal_c_rescue/${RUN}"/model_*.pt | head -1 | xargs -n1 basename)
echo "resuming from run=${RUN} checkpoint=${CKPT}"

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Rescue-Anymal-C-v0 \
  --num_envs 2048 \
  --max_iterations 1200 \
  --resume \
  --load_run "${RUN}" \
  --checkpoint "${CKPT}" \
  --enable_cameras \
  --headless
