# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 3 sensor-equipped rescue task for ANYmal-C."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Rescue-Anymal-C-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rescue_env_cfg:AnymalCRescueEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRescuePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Rescue-Anymal-C-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rescue_env_cfg:AnymalCRescueEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRescuePPORunnerCfg",
    },
)

# Camera-free variant. Registered separately so the RTX-dependent env stays untouched;
# see AnymalCRescueNoCamEnvCfg for why it exists.
gym.register(
    id="Isaac-Rescue-Anymal-C-NoCam-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rescue_env_cfg:AnymalCRescueNoCamEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRescueNoCamPPORunnerCfg",
    },
)

# Stage 1 of two-stage training: same scene, same observations, victim rewards zeroed.
# Shares the runner cfg so the checkpoint loads straight into the stage-2 run.
gym.register(
    id="Isaac-Rescue-Anymal-C-Locomotion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rescue_env_cfg:AnymalCRescueLocomotionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRescuePPORunnerCfg",
    },
)

# Hierarchical variant: the velocity command is driven by the thermal camera (scripted
# seeker + learned residual + LiDAR shield) instead of by UniformVelocityCommand. Carries a
# second entry point, since training the navigator needs its own runner cfg while the frozen
# walker is rebuilt from the standard one.
gym.register(
    id="Isaac-Rescue-Anymal-C-Nav-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rescue_env_cfg:AnymalCRescueNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRescuePPORunnerCfg",
        "navigator_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NavigatorPPORunnerCfg",
    },
)

# Inspection variant of the navigator task. Without this the demo renderer falls back to
# Isaac-Rescue-Anymal-C-Play-v0, which still carries the random UniformVelocityCommand -- so
# the video would show random wandering while claiming to show a trained navigator.
gym.register(
    id="Isaac-Rescue-Anymal-C-Nav-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rescue_env_cfg:AnymalCRescueNavEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRescuePPORunnerCfg",
        "navigator_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NavigatorPPORunnerCfg",
    },
)
