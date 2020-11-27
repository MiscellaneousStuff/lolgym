# PyLoL OpenAI Gym Environments

OpenAI Gym Environments for the League of Legends v4.20
PyLoL environment.

## Installation

You can install LoLGym from a local clone of the git repo:

```shell
git clone https://github.com/MiscellaneousStuff/lolgym.git
pip3 install -e lolgym/
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

    env = gym.make("LoLGame-v0")
    env.settings["map_name"] = "New Summoners Rift" # Set the map
    env.settings["human_observer"] = False # Set to true to run league client
    env.settings["host"] = "localhost" # Set this to a local ip
    env.settings["players"] = "Nidalee.BLUE,Lucian.PURPLE"
    
The `players` setting specifies which champions are in the game and what
team they are playing on. The `pysc2` environment expects them to be in
a comma-separated list of Champion.TEAM items with that exact capitalization.

Versions:
- `LoLGame-v0`: The full game with complete access to action and observation
space.

#### Notes
- The action space for this environment doesn't require the call to `functionCall`
like `pylol` does. You only need to call it with an array of action and arguments.
For example:

        _SPELL = actions.FUNCTIONS.spell.id
        _EZREAL_Q = [0]
        _TARGET = point.Point(8000, 8000)
        acts = [[_SPELL, _EZREAL_Q, _TARGET] for _ in range(env.n_agents)]
        obs_n, reward_n, done_n, _ = env.step(acts)

    The environment will not check whether an action is valid before passing it
    along to the `pysc2` environment so make sure you've checked what actions are
    available from `obs.observation["available_actions"]`.

- This environment doesn't specify the `observation_space` and `action_space` members
like traditional `gym` environments. Instead, it provides access to the `observation_spec`
and `action_spec` objects from the `pylol` environment.

### General Notes
* Per the Gym environment specifications, the reset function returns an observation,
and the step function returns a tuple (observation_n, reward_n, done_n, info_n), where
info_n is a list of empty dictionaries. However, because `lolgym` is a multi-agent environment
each item is a list of items, i.e. `observation_n` is an observation for each agent, `reward_n`
is the reward for each agent, `done_n` is whether any of the `observation.step_type` is `LAST`.
* Aside from `step()` and `reset()`, the environments define a `save_replay()`
method, that accepts a single parameter `replay_dir`, which is the name of the replay
directory to save the `GameServer` replays inside of.
* All the environments have the following additional properties:
    - `episode`: The current episode number
    - `num_step`: The total number of steps taken
    - `episode_reward`: The total reward received for this episode
    - `total_reward`: The total reward received for all episodes
* The examples folder contains examples of using the various environments.
