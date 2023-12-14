from stable_baselines3 import A2C
# conda install -c conda-forge swig box2d-py
import gymnasium as gym
import os

models_dir = "models_final/A2C_100k"
logdir = "logs_100k"


if not os.path.exists(models_dir):
   os.makedirs(models_dir)

if not os.path.exists(logdir):
   os.makedirs(logdir)

env = gym.make("LunarLander-v2", render_mode="human")
env.reset()

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 100000
episodes = 10

for i in range(episodes):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f"{models_dir}/{TIMESTEPS * i}")

env.close()