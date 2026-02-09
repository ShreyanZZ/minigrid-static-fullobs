import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper 

def make_env(env_key, seed=None, render_mode="rgb_array"):
    env = gym.make(env_key, render_mode=render_mode)
    # print(env_key)
    env=FullyObsWrapper(env)
    env=ImgObsWrapper(env)
    env.reset(seed=seed)
    return env
