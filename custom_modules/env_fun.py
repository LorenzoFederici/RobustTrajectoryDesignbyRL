import gym
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor

# The multiprocessing implementation requires a function that 
# can be called inside 
# the process to instantiate a gym env
def make_env(env_id, rank, seed=0, monitor=True, filename=None, **kwargs):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for PRNG
    :param monitor: (bool) monitor the training process
    :param filename: (string) the location to save an optional log file
    :param kwargs: the kwargs to pass to the environment class
    """
    def _init():
        env = gym.make(env_id, **kwargs)
        env.seed(seed + rank) # Important: use a different seed for each environment
        if monitor:
            env = Monitor(env, filename=filename, allow_early_resets=True)
        return env

    set_global_seeds(seed)
    return _init