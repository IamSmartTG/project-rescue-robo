#!/usr/bin/env bash
# First GPU run of the sensor env. Small and short on purpose: 32 envs for a few
# iterations surfaces a construction crash, a bad tensor layout, or the thermal
# assert in seconds instead of after a 4096-env scene build.
#
# --enable_cameras is mandatory -- without it the RTX renderer never starts and the
# camera silently produces nothing.
#
# What this is checking:
#   1. env constructs at all (gym registration, obs_groups keys, cfg validation)
#   2. thermal camera returns colorized (N,H,W,4) uint8 -- rescue_mdp asserts this
#   3. LiDAR returns 288 rays after the vertical-FOV change
#   4. victims settle instead of being ejected by the solver
#   5. throughput, to decide whether 1500 iterations fits before Aug 16

set -euo pipefail

export OMNI_KIT_ACCEPT_EULA=YES
source /workspace/venv-isaaclab/bin/activate
cd /workspace/IsaacLab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Rescue-Anymal-C-v0 \
  --num_envs 32 \
  --max_iterations 5 \
  --enable_cameras \
  --headless
