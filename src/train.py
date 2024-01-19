import gym
from stable_baselines3 import PPO, A2C
from wrapper.wrapper import F110_Wrapped
import yaml
from argparse import Namespace
from f110_gym.envs.base_classes import Integrator
import numpy as np
import os
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

eval_env = F110_Wrapped(eval_env, random_map=True)

eval_env.seed(1773449316)

skip_training = False

max_timesteps = 200_000
min_timesteps = 50_000

max_learning_rate = 0.0005
min_learning_rate = 0.00005
num_of_steps = 6

device = 'cpu'

timesteps_list = np.logspace(np.log10(min_timesteps), np.log10(max_timesteps), num=num_of_steps, endpoint=True, base=10.0, dtype=int, axis=0)
learning_rate_list = np.logspace(np.log10(max_learning_rate), np.log10(min_learning_rate), num=num_of_steps, endpoint=True, base=10.0, dtype=None, axis=0)

print("timestemp" , timesteps_list)
print("leaning rate", learning_rate_list)

if not skip_training:

    for timesteps, learning_rate in zip(timesteps_list, learning_rate_list):
            #if best_model exists, load it
        # if os.path.exists("./train_test/best_model.zip"):
        #     print("Loading Existing Model")
        #     model = PPO.load("./train_test/best_model", eval_env, learning_rate=learning_rate)
        if os.path.exists("./train_test/best_global_model.zip"):
            print("Loading Existing Model")
            model = PPO.load("./train_test/best_global_model", eval_env, learning_rate=linear_schedule(learning_rate), device=device)
        else:
            model = PPO("MlpPolicy",
                        eval_env,
                        gamma=0.99,
                        learning_rate=linear_schedule(learning_rate),
                        gae_lambda=0.95,
                        verbose=0, 
                        tensorboard_log=tensorboard_path,
                        device=device
                        )


        eval_callback = EvalCallback(eval_env, best_model_save_path='./train_test/',
                                    log_path='./train_test/', eval_freq= int(timesteps/20),
                                    deterministic=True, render=False)
        


        model.learn(total_timesteps=timesteps, progress_bar=True, callback=eval_callback)

        del model 

        model = PPO.load("./train_test/best_model", eval_env, device=device)

        mean_reward, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=20)


        #save in file the mean reward, if the file does not exist, create it
        if not os.path.exists("./train_test/mean_reward.txt"):
            with open("./train_test/mean_reward.txt", "w") as f:
                f.write(f"{mean_reward}")
                model.save("./train_test/best_global_model")
        else:
            #overwrite the file with the new mean reward if it is better
            with open("./train_test/mean_reward.txt", "r") as f:
                best_mean_reward = float(f.read())
            if mean_reward > best_mean_reward:
                with open("./train_test/mean_reward.txt", "w") as f:
                    f.write(f"{mean_reward}")
                model.save("./train_test/best_global_model")
                print("Saved Best Model")

        del model 



    


eval_env  = gym.make('f110_gym:f110-v0', map=map_path, map_ext=map_ext, num_agents=1, timestep=timestep, integrator=Integrator.RK4)
eval_env = F110_Wrapped(eval_env, random_map=False)
eval_env.set_map_path(path)

model = PPO.load("./train_test/best_global_model", eval_env, device=device)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)

print(mean_reward)

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
        eval_env.render(mode='human')
