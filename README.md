# PyLoL OpenAI Gym Environments

OpenAI Gym Environments for the League of Legends v4.20
PyLoL environment.

## Installation

You can install LoLGym from a local clone of the git repo:

```shell
git clone https://github.com/MiscellaneousStuff/lolgym.git
pip3 install --upgrade lolgym/
```
## Usage

You need the following minimum code to run any LoLGym environment:

Import gym and this package:

    import gym
    import lolgym.envs

Import and initialize absl.flags (required due to `pylol` dependency)

    import sys
    from absl import flags
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

Create and initialize the specific environment.

## Available Environments

### LoLGame

The full League of Legends v4.20 game environment. Initialize as follows:

    env = gym.make('LoLGame-v0')
    todo: options go here...
    
Versions:
- `LoLGame-v0`: The full game with complete access to action and observation
space.

#### Notes
- The action space for this environment doesn't require the call to `functionCall`
like `pylol` does. You only need to call it with an array of action and arguments.
For example:

    todo: action function call example goes here...

- This environment doesn't specify the observation_space and action_space members
like traditional `gym` environments. Instead, it provides access to the `observation_spec`
and `action_spec` objects from the `pylol` environment.

### General Notes
*  Per the Gym environment specifications, the reset function returns an observation,
and the step function returns a tuple (observation, reward, done, info), where info is
an empty dictionary and the observation is the observatoin object from the pylol 
environment. The reward is the same as observation.reward, and done is equal true if
observation.step_type is LAST.
* Aside from `step()` and `reset()`, the enviroments define a `save_replay()`,
method that accepts a single parameter `replay_dir`, which is the name of the replay
directory to save the GameServer replays inside of.
* All the environments have the following additional properties:
    - `episode`: The current episode number
    - `num_step`: The total number of steps taken
    - `episode_reward`: The total reward received for this episode
    - `total_reward`: The total reward received for all episodes
* The examples folder contains examples of using the environments.
