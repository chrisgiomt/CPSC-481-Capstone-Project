import gymnasium as gym
import numpy as np

from gym.envs.registration import register

env = gym.make('LunarLander-v2', render_mode='human')
env.reset()

window_width, window_height = env.render().shape

print(window_width)
print(window_height)

env.cloes()