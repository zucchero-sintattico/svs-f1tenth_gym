import gym
from stable_baselines3 import PPO
from car_node.car_node.wrapper import F110_Wrapped
from f110_gym.envs.base_classes import Integrator
from stable_baselines3.common.evaluation import evaluate_policy
from car_node.car_node.map_utility import get_map, get_formatted_map, map_ext
from utility.linear_schedule import linear_schedule

timestep = 0.01
tensorboard_path = './train_test/'
map = "Zandvoort"
path = get_map(map)
map_path = get_formatted_map(path)
device = 'cpu'

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
