import gym
from stable_baselines3 import PPO, SAC
from wrapper.wrapper import F110_Wrapped
from f110_gym.envs.base_classes import Integrator
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
import utility.map_utility as map_utility
from utility.linear_schedule import linear_schedule
from itertools import product
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.env_checker import check_env
class OneMap:
    def __init__(self, map_name="Monza", load_model=False, total_timesteps=50_000, grid_search=False):
        self.timestep = 0.01
        self.total_timesteps = total_timesteps
        self.learning_rate = linear_schedule(0.0003)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.verbose = 1
        self.device = 'cpu'
        self.eval_env = None
        self.tensorboard_path = "./train_test/"
        self.load_model = load_model
        path = map_utility.get_map(map_name)
        map_path = map_utility.get_formatted_map(path)
        map_ext = map_utility.map_ext

        

        self.env = gym.make('f110_gym:f110-v0', map=map_path, map_ext=map_ext, num_agents=1, timestep=self.timestep)
                                 
        # Wrap basic gym with RL functions
        self.eval_env = F110_Wrapped(self.env, random_map=False)
        self.eval_env.set_map_path(path)
        self.eval_env.seed(np.random.randint(pow(2, 31) - 1))

        #check_env(self.eval_env, warn=False, skip_render_check=True)

        

    def run(self):


        model = PPO("MlpPolicy", self.eval_env, 
                     gae_lambda=self.gae_lambda, 
                     verbose=self.verbose, 
                     tensorboard_log=self.tensorboard_path,
                     learning_rate=self.learning_rate,
                     gamma=self.gamma,
                     device=self.device)
        
        if self.load_model:
            model = PPO.load(self.tensorboard_path + "best_model", env=self.eval_env)

        eval_callback = EvalCallback(self.eval_env, best_model_save_path='./train_test/',
                             log_path='./train_test/', eval_freq=1000,
                             deterministic=True, render=False)


        model.learn(total_timesteps=self.total_timesteps, progress_bar=True, callback=eval_callback)
        # Define the parameter grid
  


if __name__ == "__main__":
    train_on_one_map = OneMap()
    train_on_one_map.run()
