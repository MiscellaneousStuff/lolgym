# MIT License
# 
# Copyright (c) 2020 MiscellaneousStuff
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Escape the enemy champion only by moving along the X-Axis."""

import numpy as np

import gym
from gym.spaces import Box, Discrete

from pylol.lib import actions, point

from lolgym.envs.lol_game import LoLGameEnv

_NO_OP = [actions.FUNCTIONS.no_op.id]

class Escape1DEnv(LoLGameEnv):
    def __init__(self, **kwargs) -> None:
        super().__init__()
    
    def transform_obs(self, obs):
        obs = np.array(obs[0].observation["enemy_unit"].distance_to_me, dtype=np.float32)[None]
        return obs

    def reset(self):
        obs = self.transform_obs(super().reset())
        return obs

    def _safe_step(self, act):
        act_x = 8 if act else 0
        act_y = 4
        act = [[1, point.Point(act_x, act_y)],
                _NO_OP]

        obs_n, reward_n, done_n, _ = super()._safe_step(act)

        obs = self.transform_obs(obs_n) # obs_n[0].observation["enemy_unit"].distance_to_me
        reward = obs_n[0].observation["enemy_unit"].distance_to_me.item()
        reward = reward if reward else 0.0 # Ensures reward is something sensible
        done = all(done_n)
        return obs, reward, done, {}

    @property
    def action_space(self):
        if self._env is None:
            self._init_env()
        action_space = Discrete(2)
        return action_space

    @property
    def observation_space(self):
        if self._env is None:
            self._init_env()
        observation_space = Box(low=0, high=24000, shape=(1,), dtype=np.float32)
        return observation_space