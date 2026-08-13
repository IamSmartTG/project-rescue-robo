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
#
# 4096 envs, not the 2048 stage 1 used. That is the batch size AnymalCRoughPPORunnerCfg's
# learning rate, desired_kl and num_mini_batches were tuned against, so 2048 was actually
# the mismatched setting. Measured headroom made it safe: 8.4 GB of 46 GB and 33% GPU
# utilisation at 2048. Changing env count between stages is fine -- it does not touch the
# policy network, so the stage-1 checkpoint still loads.
#
# Going further (8192) would be slower in wall-clock, not faster, and would leave the
# tuned regime for gradient estimates that are already accurate enough.

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
  --num_envs 4096 \
  --max_iterations 1200 \
  --resume \
  --load_run "${RUN}" \
  --checkpoint "${CKPT}" \
  --enable_cameras \
  --headless
