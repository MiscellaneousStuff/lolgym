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
"""Example of a basic full game environment with no actions."""

import sys

import gym
from absl import flags
from pylol.lib import actions

import lolgym.envs

FLAGS = flags.FLAGS

_NO_OP = [actions.FUNCTIONS.no_op.id]

def main():
    env = gym.make("LoLGame-v0")
    env.settings["map_name"] = "New Summoners Rift"
    # env.settings["human_observer"] = True # Set to true to run league client
    env.settings["host"] = "127.0.1.1" # Set this to localhost ip

    obs_n = env.reset()

    for _ in range(100):
        action = [_NO_OP] * env.n_agents
        obs_n, reward_n, done_n, _ = env.step(action)
        if any(done_n):
            break

    env.close()
    
if __name__ == "__main__":
    main()