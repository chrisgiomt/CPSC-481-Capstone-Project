import gymnasium as gym
import numpy as np
from pyboy import PyBoy, WindowEvent

class MarioGymEnv(gym.Env):
    def __init__(self, rom_path):
        super(MarioGymEnv, self).__init__()
        self.pyboy = PyBoy(rom_path, window_type="headless", window_scale=3, game_wrapper=True)
        self.action_space = gym.spaces.Discrete(5)  # Replace with the actual action space size
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def reset(self):
        self.pyboy.load_rom()
        # Additional setup for the game if needed
        return self.pyboy.get_screen().screen_ndarray()

    def step(self, action):
        # Map Gym action space to PyBoy inputs
        pyboy_action = self.map_gym_action(action)
        self.pyboy.send_input(pyboy_action)

        # Execute one PyBoy tick
        self.pyboy.tick()

        # Get the current game state
        observation = self.pyboy.get_screen().screen_ndarray()
        reward = 0  # Replace with the actual reward calculation
        done = self.pyboy.tick()  # Replace with the actual termination condition
        info = {}

        return observation, reward, done, info

    def render(self, mode='human'):
        return self.pyboy.get_screen().screen_ndarray()

    def close(self):
        self.pyboy.stop()

    def map_gym_action(self, gym_action):
        # Map Gym action space to PyBoy inputs
        # Customize this method based on your game controls
        if gym_action == 0:
            return WindowEvent.PRESS_ARROW_RIGHT
        elif gym_action == 1:
            return WindowEvent.PRESS_ARROW_LEFT
        elif gym_action == 2:
            return WindowEvent.PRESS_ARROW_UP
        elif gym_action == 3:
            return WindowEvent.PRESS_ARROW_DOWN
        elif gym_action == 4:
            return WindowEvent.PRESS_BUTTON_A
        else:
            return None  # Handle additional actions as needed

