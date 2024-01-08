import gym
from stable_baselines3 import PPO
from wrapper.wrapper import F110_Wrapped
from f110_gym.envs.base_classes import Integrator
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
import utility.map_utility as map_utility
from utility.linear_schedule import linear_schedule

class BaseRandomMap:
    def __init__(self, load_model=False, total_timesteps = 50_000):
        self.timestep = 0.01
        self.total_timesteps = total_timesteps
        self.learning_rate = linear_schedule(0.0003)
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.load_model = load_model
        self.verbose = 1
        self.device = 'cpu'
        self.eval_env = None
        self.tensorboard_path = "./train_test/"


        self.eval_env = gym.make('f110_gym:f110-v0', num_agents=1, timestep=self.timestep, integrator=Integrator.RK4)
        # Wrap basic gym with RL functions
        self.eval_env = F110_Wrapped(self.eval_env, random_map=True)
        self.eval_env.seed(np.random.randint(pow(2, 31) - 1))


    def run(self):

        model = PPO("MlpPolicy", self.eval_env, 
                     gae_lambda=self.gae_lambda, 
                     verbose=self.verbose, 
                     tensorboard_log=self.tensorboard_path)
        
        if self.load_model:
            model = PPO.load(self.tensorboard_path + "best_model", env=self.eval_env)


        model.learn(total_timesteps=self.total_timesteps)

        model.save(self.tensorboard_path + "best_model")

if __name__ == "__main__":
    train_on_one_map = BaseRandomMap()
    train_on_one_map.run()


