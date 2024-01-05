import gym
from stable_baselines3 import PPO
from wrapper.wrapper import F110_Wrapped
from f110_gym.envs.base_classes import Integrator
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
import utility.map_utility as map_utility
from utility.linear_schedule import linear_schedule

class TrainOnOneMap:
    def __init__(self, map_name="Monza"):
        self.timestep = 0.01
        self.total_timesteps = 1_000
        self.learning_rate = linear_schedule(0.0003)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.verbose = 1
        self.device = 'cpu'
        self.eval_env = None
        
        path = map_utility.get_map(map_name)
        map_path = map_utility.get_formatted_map(path)
        map_ext = map_utility.map_ext

        self.eval_env = gym.make('f110_gym:f110-v0', map=map_path, map_ext=map_ext, num_agents=1, timestep=self.timestep, integrator=Integrator.RK4)
        # Wrap basic gym with RL functions
        self.eval_env = F110_Wrapped(self.eval_env, random_map=False)
        self.eval_env.set_map_path(path)
        self.eval_env.seed(np.random.randint(pow(2, 31) - 1))

        

    def run(self):

        try:
            model = PPO.load("./train_test/best_model", self.eval_env, device='cpu')
        except:
            model = PPO("MlpPolicy", self.eval_env, learning_rate=self.learning_rate, gamma=self.gamma, gae_lambda=self.gae_lambda, verbose=self.verbose, device=self.device)

        eval_callback = EvalCallback(self.eval_env, best_model_save_path='./train_test/', log_path='./train_test/', eval_freq=1000, deterministic=True, render=False)

        model.learn(total_timesteps=self.total_timesteps, callback=eval_callback)

if __name__ == "__main__":
    train_on_one_map = TrainOnOneMap()
    train_on_one_map.run()
