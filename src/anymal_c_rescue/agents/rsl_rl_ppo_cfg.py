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
    # 25, not 50. Two massedcompute L40S instances were reclaimed out from under this project
    # within twelve hours, one of them mid-run, so the question is not whether a run gets
    # interrupted but how much it costs when it does. At ~7.5 s/iter a 50-iteration gap is
    # ~6 minutes of lost work; 25 halves that. A checkpoint is 13 MB and writing one is
    # negligible against the iteration cost, so the only thing traded away is disk.
    save_interval = 25
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


@configclass
class NavigatorPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO for the residual navigator -- a much smaller problem than the walker.

    Three actions instead of twelve, perception-only input, and the hard part (locomotion on
    rough terrain) already solved and frozen. The network is sized accordingly: [256, 128]
    against the walker's [512, 256, 128]. A bigger net here would train more slowly for no
    benefit, since the mapping "warm pixels on the left -> turn left" is not complex.

    ``init_std`` is 0.3, not the walker's 1.0, and that is the residual doing its job. The
    policy output is a *correction* to the scripted seeker, so it must start near zero --
    at init this behaves like the scripted controller and training moves away from a known
    decent baseline rather than from noise. Start at 1.0 and the first rollouts are the
    scripted command plus up to a full m/s of random garbage, which throws away the guarantee
    that makes a residual worth using.

    ``gamma`` is 0.995, above the walker's 0.99. The navigator acts at ~10 Hz, so 0.99 gives
    an effective horizon of ~100 decisions = 10 s -- workable, but 0.995 covers a full 20 s
    episode, which is what it takes for "turned toward the casualty" to get credit for
    "reached the casualty".
    """

    num_steps_per_env = 24
    max_iterations = 600
    save_interval = 25
    experiment_name = "navigator"
    obs_groups = {"actor": ["navigator"], "critic": ["navigator"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.3),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128],
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
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
