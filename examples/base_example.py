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
"""Base example which only ..."""

import numpy as np
import gym
from absl import flags

import lolgym.envs


FLAGS = flags.FLAGS
FLAGS([__file__])

class BaseExample(object):
    def __init__(self, env_name) -> None:
        super().__init__()
        self.env_name = env_name
    
    def run(self, num_episodes=1):
        env = gym.make(self.env_name)

        episode_rewards = np.zeros((num_episodes,), dtype=np.int32)
        episodes_done = 0
        for i in range(num_episodes):
            obs = env.reset()

            done = False
            while not done:
                action = self.get_action(env, obs)
                obs, reward, done, _ = env.step(action)
            
            if obs is None:
                break
                
            episodes_done += 1
            episode_rewards[i] = env.episode_reward
        
        env.close()

        return episode_rewards[:episodes_done]

    def get_action(self, env, obs):
        raise NotImplementedError("Inherited classes must override get_action() method")