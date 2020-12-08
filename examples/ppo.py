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
"""Basic PPO implementation for LoLGym environment."""

import random
import time
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Activation, Lambda

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