import gym
from stable_baselines3 import PPO
from wrapper.wrapper import F110_Wrapped
from argparse import Namespace
from f110_gym.envs.base_classes import Integrator
import utility.map_utility as map_utility
from random_train import RandomTrain
from train_on_one_map import TrainOnOneMap

#RandomTrain().run()

TrainOnOneMap("YasMarina").run()


timestep = 0.01

path = map_utility.get_map("YasMarina")
map_path = map_utility.get_formatted_map(path)
map_ext = map_utility.map_ext


eval_env  = gym.make('f110_gym:f110-v0', map=map_path, map_ext=map_ext, num_agents=1, timestep=timestep, integrator=Integrator.RK4)
eval_env = F110_Wrapped(eval_env, random_map=False)
eval_env.set_map_path(path)


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
