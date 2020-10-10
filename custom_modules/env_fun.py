import gym
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor

# The multiprocessing implementation requires a function that 
# can be called inside 
# the process to instantiate a gym env
def make_env(env_id, rank, seed=0, filename=None, *argv):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.__init__(*argv)
        # Important: use a different seed for each environment
        env.seed(seed + rank)
        env = Monitor(env, filename=filename, allow_early_resets=True)

        return env

    set_global_seeds(seed)
    return _init