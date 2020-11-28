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
"""Example of a basic full game environment with random actions."""

import sys

import numpy as np
import gym
from absl import flags

import lolgym.envs
from pylol.lib import actions, features

FLAGS = flags.FLAGS

def main():
    env = gym.make("LoLGame-v0")
    env.settings["map_name"] = "Howling Abyss"
    env.settings["human_observer"] = True # Set to true to run league client
    env.settings["host"] = "192.168.0.16" # Set this to localhost or "hostname -i" on Linux
    env.settings["players"] = "Lucian.BLUE,Lucian.PURPLE"

    done_n = [False for _ in range(env.n_agents)]
    ep_reward = 0

    obs_n = env.reset()

    #while not all(done_n):
    for _ in range(500): # Use number of steps instead of deaths to end episode
        actions = [random_action(env, timestep) for timestep in obs_n]
        obs_n, reward_n, done_n, _ = env.step(actions)
        ep_reward += sum(reward_n)

        if any(done_n):
            break

    env.close()

def random_action(env, obs):
    function_id = np.random.choice(obs.observation["available_actions"])
    args = [[np.random.randint(0, size) for size in arg.sizes]
            for arg in env.action_spec[0].functions[function_id].args]
    return [function_id] + args

if __name__ == "__main__":
    main()