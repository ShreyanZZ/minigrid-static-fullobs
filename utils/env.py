import gymnasium as gym
from .randomness import StochasticTransitionWrapper
from .rewardwrapper import ShapedRewardWrapper
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper 
from .unsafe_state import CollisionFullyObsWrapper

class FixedSeedWrapper(gym.Wrapper):
    """
    Forces the environment to always reset with a specific seed,
    ignoring whatever seed is passed to reset().
    """
    def __init__(self, env, seed):
        super().__init__(env)
        self._forced_seed = seed

    def reset(self, **kwargs):
        # We completely ignore the 'seed' arg coming from the training loop
        # and always inject our forced seed.
        return self.env.reset(seed=self._forced_seed, options=kwargs.get('options'))
    
def make_env(env_key, seed=None, render_mode="rgb_array",slip=0.03):
    env = gym.make(env_key, render_mode=render_mode)
    if seed is not None:
        env = FixedSeedWrapper(env, seed)
    # print(env_key)
    env = StochasticTransitionWrapper(env, prob=slip)
    env = ShapedRewardWrapper(env)
    env = CollisionFullyObsWrapper(env)
    env.reset(seed=seed)
    return env
