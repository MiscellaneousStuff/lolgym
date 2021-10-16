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
"""Module setuptools script."""

from setuptools import setup


description = """LoLGym - OpenAI Gym Environments for League of Legends v4.20

LoLGym is the OpenAI Gym implementation of the League of Legends v4.20 Learning
Environment (using a modified version of the LeagueSandbox's GameServer project,
not the original server from Riot.)

Read the README at https://github.com/MiscellaneousStuff/pylol for more information.
"""

setup(
    name='lolgym',
    version='1.0.0',
    description='OpenAI Gym implementation of PyLoL module',
    long_description=description,
    author='MiscellaneousStuff',
    author_email='raokosan@gmail.com',
    license='MIT License',
    keywords='League of Legends',
    url='https://github.com/MiscellaneousStuff/lolgym',
    packages=[
        'lolgym',
        'lolgym.envs',
        'examples'
    ],
    install_requires=['gym', 'absl-py', 'numpy', 'tensorflow', 'pylol-rl'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)