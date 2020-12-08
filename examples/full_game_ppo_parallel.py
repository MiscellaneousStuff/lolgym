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
"""Example of a basic full game environment implementing PPO parallel."""

import subprocess

from absl import flags
from absl import app

import os

from pylol.lib import run_parallel

FLAGS = flags.FLAGS
flags.DEFINE_integer("count", 1, "Number of games to run at once")
flags.DEFINE_string("config_path", "/mnt/c/Users/win8t/Desktop/pylol/config_dirs.txt", "Path to file containing GameServer and LoL Client directories")
flags.DEFINE_string("host", "192.168.0.16", "Host IP for GameServer, LoL Client and Redis")

def main(unused_argv):
    parallel = run_parallel.RunParallel()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    args = ["python3", dir_path + "/full_game_ppo.py",
            "--config_path", FLAGS.config_path,
            "--host", FLAGS.host]

    try:
        parallel.run((subprocess.Popen, args) for _ in range(FLAGS.count))
    except KeyboardInterrupt:
        print("CLOSE EVERYTHING :D")

def entry_point():
    app.run(main)

if __name__ == "__main__":
    app.run(main)