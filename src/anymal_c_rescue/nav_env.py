# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The navigator's decision problem, shaped the way rsl-rl expects a VecEnv.

Lives in the package rather than in ``harness/train_navigator.py`` because the renderer needs
it too: without it ``render_chase.py`` can only run the frozen walker, so a demo on the Nav
task would show the *scripted seeker* while claiming to show a trained navigator. Importing
it from the training script is not an option -- that module launches Isaac Sim at import.
"""

from __future__ import annotations

import torch


class NavigatorEnv:
    """Presents the navigator's decision problem in the VecEnv shape rsl_rl expects.

    One ``step`` here is ``inner_steps`` physics steps: install the navigator's delta on the
    command term, then run the frozen walker that many times, accumulating reward.

    Deliberately not a ``gym.Env`` subclass. ``OnPolicyRunner`` only touches ``num_envs``,
    ``num_actions``, ``device``, ``get_observations`` and ``step``, and wrapping a wrapper in
    a third adapter is how observation plumbing quietly breaks.
    """

    def __init__(self, env, walker_policy, command_term, inner_steps: int):
        self._env = env
        self._walker = walker_policy
        self._cmd = command_term
        self._inner = inner_steps

        self.device = env.unwrapped.device
        self.num_envs = env.unwrapped.num_envs
        self.num_actions = 3  # (vx, vy, wz) delta

        self._obs = env.get_observations()
        self.num_obs = self._nav_obs(self._obs).shape[1]
        # Episode length in *navigator* steps, which is what rsl_rl's bootstrapping uses.
        self.max_episode_length = int(env.unwrapped.max_episode_length) // inner_steps

    def _nav_obs(self, obs):
        """The navigator sees perception only -- it decides where to go, not how to walk.

        Joint angles and velocities are the walker's business and would be pure distraction
        in a 3-dim command policy. Falls back to the full observation if the task does not
        define a dedicated group, so this still runs on the plain rescue task.
        """
        # Probe for key access rather than isinstance(obs, dict). rsl-rl 5.x hands back a
        # TensorDict, which is NOT a dict subclass -- the isinstance check silently fell
        # through and returned the container itself, whose .shape is the batch shape
        # (num_envs,), so .shape[1] raised IndexError several lines later.
        try:
            keys = list(obs.keys())
        except (AttributeError, TypeError):
            keys = None
        if keys:
            if "navigator" in keys:
                return obs["navigator"]
            return torch.cat([obs[k] for k in sorted(keys)], dim=-1)
        return obs

    def get_observations(self):
        """Return the FULL observation container, not just the navigator slice.

        rsl-rl selects groups itself from ``obs_groups`` (here ``{"actor": ["navigator"]}``),
        so handing it a bare tensor makes it try `"navigator" in tensor` and raise
        `Tensor.__contains__ only supports Tensor or scalar`. `_nav_obs` stays for measuring
        the width at startup; group selection belongs to the runner.
        """
        return self._obs

    def reset(self):
        self._obs = self._env.get_observations()
        return self.get_observations(), {}

    def __getattr__(self, name):
        """Delegate anything not overridden here to the wrapped env.

        The class docstring above claimed the runner only touches num_envs, num_actions,
        device, get_observations and step. That was wrong -- it also wants ``cfg``, and
        discovering the rest one AttributeError at a time costs a two-minute simulator
        launch per attribute. Only the members this adapter deliberately redefines are
        intercepted; everything else is the real wrapper's.

        The leading-underscore guard prevents infinite recursion: ``self._env`` inside this
        method would re-enter ``__getattr__`` if ``_env`` were not yet assigned.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._env, name)

    def step(self, delta: torch.Tensor):
        """Hold one navigator decision for ``inner_steps`` physics steps."""
        self._cmd.set_delta(delta)

        total_reward = torch.zeros(self.num_envs, device=self.device)
        any_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        extras = {}

        for _ in range(self._inner):
            with torch.inference_mode():
                action = self._walker(self._obs)
                self._obs, reward, dones, extras = self._env.step(action)
            # Accumulate rather than take the last step's reward: the navigator is
            # responsible for everything that happened while its command was in force.
            total_reward += reward
            any_done |= dones.bool()
            # Do NOT break early on done. Isaac Lab auto-resets inside step(), so the
            # remaining inner steps belong to the next episode and must still be driven --
            # bailing out here would leave those envs frozen for the rest of the rollout.

        return self.get_observations(), total_reward, any_done, extras
