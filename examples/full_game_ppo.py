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
"""Example of a basic full game environment implementing PPO."""

import uuid

import random
import time
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Activation, Lambda

tf.compat.v1.disable_eager_execution()

import gym
import lolgym.envs
import ppo
from pylol.lib import actions, features, point
from pylol.lib import point

from absl import flags
FLAGS = flags.FLAGS

_NO_OP = [actions.FUNCTIONS.no_op.id]
_MOVE = [actions.FUNCTIONS.move.id]
_SPELL = [actions.FUNCTIONS.spell.id]

"""
NEXT THING WHICH NEEDS TO BE IMPLEMENTED:
- THE ACTION SPACE AND OBSERVATION SPACE NEEDS TO BE MINIMIZED TO ONLY WHATS NEEDED
  FOR THE SIMPLEST EXAMPLES (E.G. SUCCESSFULLY KILLING ANOTHER AGENT, FARMING MINIONS,
  KITING, ETC.)
"""

def main():
    final_out = "" # Used to store outputs
    hidden_layers = 2 # <= try changing this next...
    gamma = 0.99
    epochs = 100
    batch_steps = 25
    episode_steps = 25
    experiment_name = "run_away"

    env = gym.make("LoLGame-v0")
    env.settings["map_name"] = "Old Summoners Rift"
    # env.settings["human_observer"] = True # Set to true to run league client
    env.settings["host"] = "192.168.0.16" # Set this using "hostname -i" ip on Linux
    env.settings["players"] = "Ezreal.BLUE,Ezreal.PURPLE"

    agent = ppo.PPOAgent(
        hidden_layers=hidden_layers,
        gamma=gamma,
        action_space=2)

    lll = []

    for epoch in range(epochs):
        st = time.perf_counter()
        X, Y, V, P = [], [], [], []
        ll = []
        while len(X) < batch_steps:
            obs = env.reset()
            env.teleport(1, point.Point(7100.0, 7500.0))
            raw_obs = obs
            obs = np.array(raw_obs[0].observation["enemy_unit"].distance_to_me)[None]
            rews = []
            steps = 0
            while True:
                steps += 1
                pred, act = [x[0] for x in agent.pf(obs[None])]
                P.append(pred)

                # Save this state action pair
                X.append(np.copy(obs))
                act_mask = np.zeros((agent.action_space.n))
                act_mask[act] = 1.0
                Y.append(act_mask)
                # print("PRED, ACT:", pred, act)

                # Convert act into pylol action
                # print("ACT:", act)
                act_x = 8 if act else 0
                print("ACT X:", act_x)
                act_y = 4
                target_pos = point.Point(raw_obs[0].observation["me_unit"].position_x,
                                         raw_obs[0].observation["me_unit"].position_y)
                act = [
                    [1, point.Point(act_x, act_y)],
                    _NO_OP # _SPELL + [[0], target_pos]
                ]

                # Take the action and save the reward
                obs, rew, done, _ = env.step(act)
                raw_obs = obs
                obs = np.array(raw_obs[0].observation["enemy_unit"].distance_to_me)[None]
                
                # Default reward is our HP but now we get agent to move away from enemy
                # rew = rew[0] # Default reward is our own HP
                rew = +raw_obs[0].observation["enemy_unit"].distance_to_me

                done = done[0]
                rews.append(rew)

                if done or steps == episode_steps:
                    ll.append(np.sum(rews))
                    for i in range(len(rews)-2, -1, -1):
                        rews[i] += rews[i+1] * gamma
                    V.extend(rews)
                    break
        X, Y, V, P = [np.array(x) for x in [X, Y, V, P]]

        # Subtract value baseline to get advantage
        A = V - agent.vf(X)[:, 0]

        loss = agent.popt.fit([X, A, P], Y, batch_size=5, epochs=20, shuffle=True, verbose=0)
        loss = loss.history["loss"][-1]
        vloss = agent.v.fit(X, V, batch_size=5, epochs=20, shuffle=True, verbose=0)
        vloss = vloss.history["loss"][-1]

        lll.append((epoch, np.mean(ll), loss, vloss, len(X), len(ll), time.perf_counter() - st))
        print("%3d  ep_rew:%9.2f  loss:%7.2f   vloss:%9.2f  counts: %5d/%3d tm: %.2f s" % lll[-1])

        sign = "+" if lll[-1][1] >= 0 else ""
        final_out += sign + str(lll[-1][1])
    
    ppo.plot_data(lll)

    with open(experiment_name + "_" + str(hidden_layers) + "_layers_" + str(uuid.uuid4()) + ".txt", "w") as f:
        f.write(final_out)

    obs = env.reset()
    raw_obs = obs
    obs = np.array(raw_obs[0].observation["enemy_unit"].distance_to_me)[None]
    rews = []
    steps = 0
    while True:
        steps += 1
        pred, act = [x[0] for x in agent.pf(obs[None])]
        act = np.argmax(pred)
        
        # Convert act into pylol action
        # print("ACT:", act)
        act_x = 8 if act else 0
        print("ACT X:", act_x)
        act_y = 4
        target_pos = point.Point(raw_obs[0].observation["me_unit"].position_x,
                                    raw_obs[0].observation["me_unit"].position_y)
        act = [
            [1, point.Point(act_x, act_y)],
            _NO_OP # _SPELL + [[0], target_pos]
        ]

        # Take the action and save the reward
        obs, rew, done, _ = env.step(act)
        raw_obs = obs
        obs = np.array(raw_obs[0].observation["enemy_unit"].distance_to_me)[None]
        
        # Default reward is our HP but now we get agent to move away from enemy
        # rew = rew[0] # Default reward is our own HP
        rew = +raw_obs[0].observation["enemy_unit"].distance_to_me

        done = done[0]
        rews.append(rew)

        if done or steps == 25:
            break

    print("Ran %d steps, got %f reward" % (len(rews), np.sum(rews)))
    env.close()

if __name__ == "__main__":
    main()