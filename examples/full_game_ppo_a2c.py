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
"""Example of basic full game environment implementing A2C PPO."""

import uuid

import random
import time
import numpy as np

import threading
from queue import Queue

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
flags.DEFINE_integer("count", 1, "Number of games to run at once")
flags.DEFINE_string("config_path", "/mnt/c/Users/win8t/Desktop/pylol/config_dirs.txt", "Path to file containing GameServer and LoL Client directories")
flags.DEFINE_string("host", "192.168.0.16", "Host IP for GameServer, LoL Client and Redis")

def plot_data(lll):
    plt.figure(figsize=(16, 8))
    plt.subplot(1,2,1)
    plt.plot([x[1] for x in lll], label="Mean Episode Reward")
    plt.plot([x[2] for x in lll], label="Epoch Loss")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot([x[3] for x in lll], color='green', label="value Loss")
    plt.legend()

def run_thread():
    print("hi")

def main(unused_argv):
    """Run an agent."""

    threads = []
    for _ in range(FLAGS.count-1):
        t = threading.Thread(target=run_thread)
        threads.append(t)
        t.start()
    
    run_thread()

    for t in threads:
        t.join()

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)