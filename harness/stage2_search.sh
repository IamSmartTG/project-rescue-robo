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
#
# That mtime heuristic is only right when stage 1 ran on this box. After rebuilding a host and
# rsync-ing checkpoints back in, mtimes reflect the *upload* order and "newest" means nothing
# -- so RUN and CKPT are overridable. Pass them explicitly whenever resuming a run that was
# not trained here.
RUN="${RUN:-$(ls -1t logs/rsl_rl/anymal_c_rescue/ | head -1)}"
CKPT="${CKPT:-$(ls -1t "logs/rsl_rl/anymal_c_rescue/${RUN}"/model_*.pt | head -1 | xargs -n1 basename)}"
# Additive on resume: rsl_rl computes total_it = start_it + n.
ITERS="${ITERS:-1200}"
# TASK is overridable so the same script can fine-tune the walker on the *navigator* task,
# where the velocity command comes from the thermal-driven seeker instead of a RNG. That
# needs no code change: AnymalCRescueNavEnvCfg only ADDS a "navigator" observation group,
# and the walker's obs_groups still names only policy+thermal, so its input width -- and
# therefore model_4197.pt -- is unchanged.
#
# Why it is worth doing: the walker learned under uniform random commands resampled every
# 10 s. The seeker feeds it smooth, sustained, purposeful motion on terrain that now contains
# 300 kg obstacles. That is inside the trained command *range* but a different distribution
# and a different scene, and the ~0.52 fall rate is the binding constraint on search success.
TASK="${TASK:-Isaac-Rescue-Anymal-C-v0}"
echo "resuming from run=${RUN} checkpoint=${CKPT} for ${ITERS} iterations"

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task "${TASK}" \
  --num_envs 4096 \
  --max_iterations "${ITERS}" \
  --resume \
  --load_run "${RUN}" \
  --checkpoint "${CKPT}" \
  --enable_cameras \
  --headless
