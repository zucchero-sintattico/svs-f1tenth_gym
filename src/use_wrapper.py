import gym
from stable_baselines3 import PPO, A2C
from env_wrapper.wrapper import F110_Wrapped
import yaml
from argparse import Namespace
from f110_gym.envs.base_classes import Integrator
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env



timestep = 0.01
tensorboard_path = 'runs'
total_timesteps = 1_000_000

PATH = 'ppo'
with open('src/map/example_map/config_example_map.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

print(conf.sx, conf.sy,conf.stheta)

eval_env  = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=timestep, integrator=Integrator.RK4)
        # wrap basic gym with RL functions

# wrap evaluation environment
eval_env = F110_Wrapped(eval_env)
#eval_env = ThrottleMaxSpeedReward(eval_env,0,1,2.5,2.5)
#eval_env = RandomF1TenthMap(eval_env, 1)
eval_env.seed(np.random.randint(pow(2, 31) - 1))


model = PPO("MlpPolicy",
                    eval_env,
                    verbose=1,
                    tensorboard_log=tensorboard_path, device='mps')

try:
    model = PPO.load(PATH, eval_env,  device='mps')
except:
    pass

model.learn(total_timesteps=total_timesteps)
model.save(PATH)

del model # remove to demonstrate saving and loading

model = PPO.load(PATH)

episode = 0
while episode < 1000:
    episode += 1
    obs = eval_env.reset(random=False)
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = eval_env.step(action)
        if done:
            print("Lap done", episode)
        eval_env.render(mode='human')
