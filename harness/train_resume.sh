#!/usr/bin/env bash
# Finish the Phase 2 baseline from where the pod outage cut it off.
#
# The run died at iteration 1365/1500 with model_1350.pt on disk. rsl_rl's
# learn() computes total_it = start_it + num_learning_iterations, and start_it
# comes from the loaded checkpoint -- so 150 here means 1350 -> 1500, not 150
# fresh iterations. Flags otherwise match train_baseline.sh so the benchmark
# stays comparable.

set -euo pipefail

export OMNI_KIT_ACCEPT_EULA=YES
source /workspace/venv-isaaclab/bin/activate
cd /workspace/IsaacLab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-Anymal-C-v0 \
  --num_envs 4096 \
  --max_iterations 150 \
  --resume \
  --load_run 2026-08-12_14-32-14 \
  --checkpoint model_1350.pt \
  --viz none
