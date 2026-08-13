# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class AnymalCRescuePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for the ANYmal-C rescue task.

    Mirrors ``AnymalCRoughPPORunnerCfg`` but routes both the ``policy`` group
    (proprioception + height scan + LiDAR) and the ``thermal`` group (flattened 64x64
    pseudo-thermal image) into the actor and the critic.

    No symmetry variant: ``mdp.symmetry.anymal.compute_symmetric_states`` is written
    against the baseline observation layout and would need updating for the extra LiDAR
    and thermal terms.
    """

    num_steps_per_env = 24
    # 1500, matching the baseline -- not 3000. The sensor env renders 4096 x 64 x 64
    # segmentation per policy step, so it runs several times slower than the 41k
    # env-steps/s baseline; 3000 iterations projected to 9-26 h against an Aug 16 deadline.
    # Raise it only after the measured throughput of a real run says it fits.
    max_iterations = 1500
    save_interval = 50
    experiment_name = "anymal_c_rescue"
    obs_groups = {"actor": ["policy", "thermal"], "critic": ["policy", "thermal"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class AnymalCRescueNoCamPPORunnerCfg(AnymalCRescuePPORunnerCfg):
    """Runner for the camera-free variant: no ``thermal`` observation group exists."""

    experiment_name = "anymal_c_rescue_nocam"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
