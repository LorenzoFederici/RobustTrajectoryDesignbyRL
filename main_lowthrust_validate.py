# CHECK THE ENVIRONMENT #

import gym
import gym_lowthrust
from stable_baselines.common.env_checker import check_env

env = gym.make('lowthrust-v0')

# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)