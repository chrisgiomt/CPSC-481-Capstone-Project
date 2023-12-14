from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
# conda install -c conda-forge swig box2d-py
import gymnasium as gym
import os

env = gym.make("LunarLander-v2", render_mode="human")
env.reset()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        obs, reward, done, truncated, info = env.step(env.action_space.sample())

env.close()