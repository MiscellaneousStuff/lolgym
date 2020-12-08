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
from pylol.lib import actions, features, point
from pylol.lib import point

from absl import flags
FLAGS = flags.FLAGS

_NO_OP = [actions.FUNCTIONS.no_op.id]
_MOVE = [actions.FUNCTIONS.move.id]
_SPELL = [actions.FUNCTIONS.spell.id]

import gym
from gym.spaces import Box, Tuple, Discrete, Dict, MultiDiscrete

import matplotlib.pyplot as plt

def plot_data(lll):
    plt.figure(figsize=(16, 8))
    plt.subplot(1,2,1)
    plt.plot([x[1] for x in lll], label="Mean Episode Reward")
    plt.plot([x[2] for x in lll], label="Epoch Loss")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot([x[3] for x in lll], color='green', label="value Loss")
    plt.legend()

class PPOAgent(object):
    """Basic PPO implementation for LoLGym environment."""
    def __init__(self, hidden_layers=1, gamma=0.99, action_space=2):
        observation_space = Box(low=0, high=800, shape=(1,), dtype=np.float32)
        action_space = Discrete(action_space)
        
        self.observation_space = observation_space
        self.action_space = action_space

        self.init_policy_function(hidden_layers=hidden_layers)
        self.init_value_function(hidden_layers=hidden_layers)

        self.gamma = gamma
        self.lll = []

    def init_value_function(self, hidden_layers):
        observation_space = self.observation_space

        # value function
        x = in1 = Input(observation_space.shape)
        x = Dense(hidden_layers, activation='elu')(x)
        x = Dense(hidden_layers, activation='elu')(x)
        x = Dense(hidden_layers, activation='elu')(x)
        x = Dense(1)(x)
        v = Model(in1, x)
        v.compile(Adam(1e-3), 'mse')
        v.summary()

        vf = K.function(v.layers[0].input, v.layers[-1].output)

        self.vf = vf
        self.v = v

    def init_policy_function(self, hidden_layers):
        observation_space = self.observation_space
        action_space = self.action_space

        # policy function
        x = in_state = Input(observation_space.shape)
        x = Dense(hidden_layers, activation='elu')(x) # x = Dense(16, activation='elu')(x)
        x = Dense(hidden_layers, activation='elu')(x) # x = Dense(16, activation='elu')(x)
        x = Dense(hidden_layers, activation='elu')(x) # x = Dense(16, activation='elu')(x)
        # x = Dense(action_space_n)(x)
        x = Dense(action_space.n)(x)
        action_dist = Lambda(lambda x: tf.nn.log_softmax(x, axis=-1))(x)
        #print("Action Dist:", action_dist)
        p = Model(in_state, action_dist)
        #print("POLICY MODEL LAYERS:", p.layers[0].input, [p.layers[-1].output,
        #                tf.random.categorical(p.layers[-1].output, 1)[0]])
        pf = K.function(p.layers[0].input,
                        [p.layers[-1].output,
                        tf.random.categorical(p.layers[-1].output, 1)[0]])
        #print("God knows:", tf.random.categorical(p.layers[-1].output, 1)[0])
        in_advantage = Input((1,))
        in_old_prediction = Input((action_space.n,))

        def loss(y_true, y_pred):
            advantage = tf.reshape(in_advantage, (-1,))
        
            # y_pred is the log probs of the actions
            # y_true is the action mask
            prob = tf.reduce_sum(y_true * y_pred, axis=-1)
            old_prob = tf.reduce_sum(y_true * in_old_prediction, axis=-1)
            ratio = tf.exp(prob - old_prob)  # hehe, they are log probs, so we subtract
            
            # this is the VPG objective
            #ll = -(prob * advantage)
            
            # this is PPO objective
            ll = -K.minimum(ratio*advantage, K.clip(ratio, 0.8, 1.2)*advantage)
            return ll

        popt = Model([in_state, in_advantage, in_old_prediction], action_dist)
        popt.compile(Adam(5e-4), loss)
        popt.summary()

        self.pf = pf
        self.popt = popt
        self.p = p

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

    agent = PPOAgent(
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
                #print("ACT X:", act_x)
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
    
    # plot_data(lll)

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
        # print("ACT X:", act_x)
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