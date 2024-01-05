import gym
from stable_baselines3 import PPO, A2C
from wrapper.wrapper import F110_Wrapped
import yaml
from argparse import Namespace
from f110_gym.envs.base_classes import Integrator
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from typing import Callable
from stable_baselines3.common.callbacks import EvalCallback
import utility.map_utility as map_utility

#adaptive learning rate
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func



timestep = 0.01
tensorboard_path = 'runs'
total_timesteps = 1_000

#load random map using utility_map

path = map_utility.get_one_random_map()
map_path = map_utility.get_formatted_map(path)
print("map path =", map_path)
map_ext = map_utility.map_ext

print(map_path)


eval_env  = gym.make('f110_gym:f110-v0', map=map_path, map_ext=map_ext, num_agents=1, timestep=timestep, integrator=Integrator.RK4)
        # wrap basic gym with RL functions

# wrap evaluation environment
eval_env = F110_Wrapped(eval_env)
eval_env.set_map_path(path)
#eval_env = ThrottleMaxSpeedReward(eval_env,0,1,2.5,2.5)
#eval_env = RandomF1TenthMap(eval_env, 1)
eval_env.seed(np.random.randint(pow(2, 31) - 1))

try:
    model = PPO.load("./train_test/best_model", eval_env,  device='cpu')
except:
    model = PPO("MlpPolicy", eval_env,  learning_rate=linear_schedule(0.0003), gamma=0.99, gae_lambda=0.95, verbose=1, device='cpu')


eval_callback = EvalCallback(eval_env, best_model_save_path='./train_test/',
                             log_path='./train_test/', eval_freq=1000,
                             deterministic=True, render=False)


model.learn(total_timesteps=total_timesteps, callback=eval_callback)


del model # remove to demonstrate saving and loading

model = PPO.load("./train_test/best_model", eval_env,  device='cpu')

episode = 0
while episode < 1000:
    episode += 1
    obs = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = eval_env.step(action)
        if done:
            print("Lap done", episode)
        eval_env.render(mode='human')
