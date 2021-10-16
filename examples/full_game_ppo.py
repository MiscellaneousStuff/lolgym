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
from tensorflow.keras.layers import Input, Dense, Activation, Lambda, LSTM

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

from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_string("config_path", "/mnt/c/Users/win8t/Desktop/pylol/config.txt", "Path to file containing GameServer and LoL Client directories")
flags.DEFINE_string("host", "192.168.0.16", "Host IP for GameServer, LoL Client and Redis")
flags.DEFINE_integer("epochs", 50, "Number of episodes to run the experiment for")
flags.DEFINE_float("step_multiplier", 1.0, "Run game server x times faster than real-time")
flags.DEFINE_bool("run_client", False, "Controls whether the game client is run or not")

class Controller(object):
    def __init__(self,
                 units=1,
                 gamma=0.99,
                 #batch_steps=None,
                 observation_space=None,
                 action_space=None):
        
        """
        if not batch_steps:
            raise ValueError("Controller needs batch step count specified")
        """

        self.units = units
        self.gamma = gamma

        self.observation_space = observation_space
        self.action_space = action_space

        self.init_policy_function(units=units)
        self.init_value_function(units=units)

        self.X = []
        self.Y = []
        self.V = []
        self.P = []

        self.n_agents = 0
        self.d_agents = 0
        self.cur_updating = True

    def plot_data(self, lll):
        plt.figure(figsize=(16, 8))
        plt.subplot(1,2,1)
        plt.plot([x[1] for x in lll], label="Mean Episode Reward")
        plt.plot([x[2] for x in lll], label="Epoch Loss")
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot([x[3] for x in lll], color='green', label="value Loss")
        plt.legend()

    def init_value_function(self, units):
        observation_space = self.observation_space

        # value function
        x = in1 = Input(observation_space.shape)
        x = Dense(units, activation='elu')(x)
        x = Dense(units, activation='elu')(x)
        x = Dense(1)(x)
        v = Model(in1, x)

        v.compile(Adam(1e-3), 'mse')
        v.summary()

        vf = K.function(v.layers[0].input, v.layers[-1].output)

        self.vf = vf
        self.v = v
    
    def init_policy_function(self, units):
        observation_space = self.observation_space
        action_space = self.action_space

        # policy function
        x = in_state = Input(observation_space.shape)
        x = Dense(units, activation='elu')(x)
        x = Dense(units, activation='elu')(x)
        x = Dense(action_space.n)(x)
        action_dist = Lambda(lambda x: tf.nn.log_softmax(x, axis=-1))(x)
        p = Model(in_state, action_dist)
        
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

        pf = K.function(p.layers[0].input,
                        [p.layers[-1].output,
                        tf.random.categorical(p.layers[-1].output, 1)[0]])
                        
        self.pf = pf
        self.popt = popt
        self.p = p

    def fit(self, batch_size=5, epochs=10, shuffle=True, verbose=0):
        X = self.X
        Y = self.Y
        V = self.V
        P = self.P

        self.d_agents += 1

        print("[FIT] FIT CHECK:", self.d_agents, self.n_agents)
        print(X)
        print(Y)
        print(V)
        print(P)

        if self.d_agents < self.n_agents:
            return None, None

        print("[FIT] TRAINING ON DATA")

        X, Y, V, P = [np.array(x) for x in [X, Y, V, P]]

        # Subtract value baseline to get advantage
        A = V - self.vf(X)[:, 0]

        loss = self.popt.fit([X, A, P], Y, batch_size=5, epochs=10, shuffle=True, verbose=0)
        loss = loss.history["loss"][-1]
        vloss = self.v.fit(X, V, batch_size=5, epochs=10, shuffle=True, verbose=0)
        vloss = vloss.history["loss"][-1]

        self.X = []
        self.Y = []
        self.V = []
        self.P = []
        
        self.d_agents = 0

        return loss, vloss

    def get_pred_act(obs):
        pred, act = [x[0] for x in self.pf(obs[None])]
        return pred, act

    def register_agent(self):
        self.n_agents += 1
        return self.n_agents

class PPOAgent(object):
    """Basic PPO implementation for LoLGym environment."""
    def __init__(self, controller=None, run_client=False):
        if not controller:
            raise ValueError("PPOAgent needs to be provided an external controller")
        
        self.controller = controller
        self.agent_id = controller.register_agent()

        print("PPOAgent:", self.agent_id, "Controller:", self.controller)

        env = gym.make("LoLGame-v0")
        env.settings["map_name"] = "Old Summoners Rift"
        env.settings["human_observer"] = run_client # Set to true to run league client
        env.settings["host"] = FLAGS.host # Set this using "hostname -i" ip on Linux
        env.settings["players"] = "Ezreal.BLUE,Ezreal.PURPLE"
        env.settings["config_path"] = FLAGS.config_path
        env.settings["step_multiplier"] = FLAGS.step_multiplier

        self.env = env

    def convert_action(self, raw_obs, act):
        action_space = self.controller.action_space

        act_x = 8 if act else 0
        act_y = 4
        target_pos = point.Point(raw_obs[0].observation["me_unit"].position_x,
                                    raw_obs[0].observation["me_unit"].position_y)
        act = [
            [1, point.Point(act_x, act_y)],
            _NO_OP # _SPELL + [[0], target_pos]
        ]

        return act

    def save_pair(self, obs, act):
        action_space = self.controller.action_space
        self.controller.X.append(np.copy(obs))
        act_mask = np.zeros((action_space.n))
        act_mask[act] = 1.0
        self.controller.Y.append(act_mask)

    def close(self):
        self.env.close()

    def run(self, max_steps):
        obs = self.env.reset()
        
        # Spawning agents at Y = 7000 due to Google Colab camera centering where agent 1 spawns
        # Ensure escaping agent is slightly left of the enemy agent so going left is the best policy
        self.env.teleport(1, point.Point(7100.0, 7000.0))
        self.env.teleport(2, point.Point(7500.0, 7000.0))
        raw_obs = obs
        obs = np.array(raw_obs[0].observation["enemy_unit"].distance_to_me)[None]
        rews = []
        steps = 0

        while True:
            steps += 1
            pred, act = [x[0] for x in self.controller.pf(obs[None])]
            act = np.argmax(pred)

            act = self.convert_action(raw_obs, act)

            obs, rew, done, _ = self.env.step(act)
            raw_obs = obs
            obs = np.array(raw_obs[0].observation["enemy_unit"].distance_to_me)[None]
            
            rew = +(raw_obs[0].observation["enemy_unit"].distance_to_me / 1000.0)

            done = done[0]
            rews.append(rew)

            if done or steps == max_steps:
                break

        print("Ran %d steps, got %f reward" % (len(rews), np.sum(rews)))

    def train(self, epochs, batch_steps, episode_steps, experiment_name):
        final_out = "" # Used to store outputs

        lll = []

        for epoch in range(epochs):
            st = time.perf_counter()
            # X, Y, V, P = [], [], [], []
            ll = []
            while len(self.controller.X) < batch_steps:
                obs = self.env.reset()
                self.env.teleport(1, point.Point(7100.0, 7000.0))
                self.env.teleport(2, point.Point(7500.0, 7000.0))
                raw_obs = obs
                # print("RAW OBS:", raw_obs[0].observation["me_unit"])
                obs = np.array(raw_obs[0].observation["enemy_unit"].distance_to_me)[None]
                rews = []
                steps = 0
                while True:
                    steps += 1

                    # Prediction, action, save prediction
                    print("[AGENT " + str(self.agent_id) + "]: obs[None] :=", obs[None])
                    pred, act = [x[0] for x in self.controller.pf(obs[None])]
                    self.controller.P.append(pred)

                    # Save this state action pair
                    self.save_pair(obs, act)

                    # Get action
                    act = self.convert_action(raw_obs, act)

                    # Take the action and save the reward
                    obs, rew, done, _ = self.env.step(act)
                    raw_obs = obs
                    obs = np.array(raw_obs[0].observation["enemy_unit"].distance_to_me)[None]
                    
                    # print(pred, act)
                    rew = +(raw_obs[0].observation["enemy_unit"].distance_to_me / 1000.0)
                    
                    done = done[0]
                    rews.append(rew)

                    if done or steps == episode_steps:
                        ll.append(np.sum(rews))
                        for i in range(len(rews)-2, -1, -1):
                            rews[i] += rews[i+1] * self.controller.gamma
                        self.controller.V.extend(rews)
                        break
            
            loss, vloss = self.controller.fit()

            if loss != None and vloss != None:
                lll.append((epoch, np.mean(ll), loss, vloss, len(self.controller.X), len(ll), time.perf_counter() - st))
                print("%3d  ep_rew:%9.2f  loss:%7.2f   vloss:%9.2f  counts: %5d/%3d tm: %.2f s" % lll[-1])
                self.env.broadcast_msg("Episode No: %3d  Episode Reward: %9.2f" % (lll[-1][0], lll[-1][1]))
                sign = "+" if lll[-1][1] >= 0 else ""
                final_out += sign + str(lll[-1][1])
        
        self.controller.plot_data(lll)

        with open(experiment_name + "_" + str(self.controller.units) + "_units_" + str(uuid.uuid4()) + ".txt", "w") as f:
            f.write(final_out)

def main(unused_argv):
    units = 1 # <= try changing this next...
    gamma = 0.99
    epochs = FLAGS.epochs # epochs = 50
    batch_steps = 25
    episode_steps = batch_steps
    experiment_name = "run_away"
    run_client = FLAGS.run_client

    # Declare observation space, action space and model controller
    observation_space = Box(low=0, high=24000, shape=(1,), dtype=np.float32)
    action_space = Discrete(2)
    controller = Controller(units, gamma, observation_space, action_space)
    # controller = Controller(units, gamma, batch_steps, observation_space, action_space)

    # Declare, train and run agent
    agent = PPOAgent(controller=controller, run_client=run_client)
    agent.train(epochs=epochs,
                batch_steps=batch_steps,
                episode_steps=episode_steps,
                experiment_name=experiment_name)
    agent.run(max_steps=episode_steps)

    agent.close()

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)
