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
from utility.map_utility import get_map, get_formatted_map, map_ext
from utility.linear_schedule import linear_schedule
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from utility.SaveOnBestTrainingRewardCallback import SaveTheBestAndRestart
timestep = 0.01
tensorboard_path = './train_test/'


map = "Zandvoort"
path = get_map(map)
map_path = get_formatted_map(path)

eval_env  = gym.make('f110_gym:f110-v0', map=map_path, map_ext=map_ext, num_agents=1, timestep=timestep, integrator=Integrator.RK4)

        # wrap basic gym with RL functions

# wrap evaluation environment
eval_env = F110_Wrapped(eval_env, random_map=True)
#eval_env.set_map_path(path)


# eval_env = Monitor(eval_env, tensorboard_path, allow_early_resets=True)
# eval_env.unwrapped.render_mode = 'human_fast'
# eval_env.unwrapped.seed






#eval_env = ThrottleMaxSpeedReward(eval_env,0,1,2.5,2.5)
#eval_env = RandomF1TenthMap(eval_env, 1)
eval_env.seed(np.random.randint(pow(2, 31) - 1))

total_timesteps = 1_000_0

for i in range (0, 10):
    try:
        model = PPO.load("./train_test/best_model", eval_env, learning_rate=0.0001)
    except:
        model = PPO("MlpPolicy", eval_env,  learning_rate=0.005, gamma=0.99, gae_lambda=0.95, verbose=2,  tensorboard_log=tensorboard_path)


    eval_callback = EvalCallback(eval_env, best_model_save_path='./train_test/',
                                log_path='./train_test/', eval_freq=1000,
                                deterministic=True, render=False)

    #model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=eval_callback)
# model.save("./train_test/best_model")

del model # remove to demonstrate saving and loading



eval_env  = gym.make('f110_gym:f110-v0', map=map_path, map_ext=map_ext, num_agents=1, timestep=timestep, integrator=Integrator.RK4)
eval_env = F110_Wrapped(eval_env, random_map=False)
eval_env.set_map_path(path)


eval_callback = EvalCallback(eval_env, best_model_save_path='./train_single_map/',
                                log_path='./train_single_map/', eval_freq=1000,
                                deterministic=True, render=False)

model = PPO.load("./train_test/best_model", eval_env)

#

#model.learn(total_timesteps=1_000, progress_bar=True, callback=eval_callback, reset_num_timesteps=False)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()


episode = 0
while episode < 100:

    episode += 1
    obs = vec_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        if done:
            print("-->", episode)
        eval_env.render(mode='human_fast')
