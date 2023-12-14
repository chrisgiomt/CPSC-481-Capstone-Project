import gymnasium as gym
from gym.envs.registration import register
from stable_baselines3 import A2C


env = gym.make('LunarLander-v2', continuous=False, render_mode="human")  # continuous: LunarLanderContinuous-v2
env.reset()

model_path = f"models_final/A2C_100k/400000.zip"
model = A2C.load(model_path, env=env)

episodes = 10

obs, info = env.reset()
for ep in range(episodes):
    obs, info = env.reset()
    done = False

    while not done:
        env.render()
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, truncated, info = env.step(action)
        print(rewards)

env.close()