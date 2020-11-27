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
"""Example of a basic full game environment with scripted actions."""

import numpy as np
import gym
from absl import flags

import lolgym.envs
from pylol.lib import actions, features, point

FLAGS = flags.FLAGS

_NO_OP = [actions.FUNCTIONS.no_op.id]
_MOVE = [actions.FUNCTIONS.move.id]
_SPELL = [actions.FUNCTIONS.spell.id]

def main():
    env = gym.make("LoLGame-v0")
    env.settings["map_name"] = "New Summoners Rift"
    # env.settings["human_observer"] = True # Set to true to run league client
    env.settings["host"] = "192.168.0.16" # Set this using "hostname -i" ip on Linux
    env.settings["players"] = "Nidalee.BLUE,Lucian.PURPLE"

    done_n = [False for _ in range(env.n_agents)]
    ep_reward = 0

    obs_n = env.reset()

    env.teleport(1, point.Point(7500.0, 7500.0))

    while not all(done_n):
        actions = [scripted_action(env, timestep) for timestep in obs_n]
        obs_n, reward_n, done_n, _ = env.step(actions)
        ep_reward += sum(reward_n)

    env.close()

def scripted_action(env, obs):
    enemy_position = point.Point(obs.observation["enemy_unit"].position_x,
                                 obs.observation["enemy_unit"].position_y)
    function_id = _SPELL if 2 in obs.observation["available_actions"] else _NO_OP

    if function_id == _SPELL:
        return function_id + [[0], enemy_position]
    else:
        return function_id + []

if __name__ == "__main__":
    main()
