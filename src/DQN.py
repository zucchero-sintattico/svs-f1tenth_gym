from collections import namedtuple, deque
import math
import random
import F110GymBase
import yaml
from argparse import Namespace
import numpy as np
from F110GymBase import PlanningStrategy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import torch.nn.init as init


with open('src/map/example_map/config_example_map.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)



class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        return x



class CustomPlanningStrategy(PlanningStrategy):
    def plan(self, observation):

        #print(observation['scans'][0], len(observation['scans'][0]))

        action = np.array([[0., 0.8]])
        return action


map = 'example_map'
num_agents = 1
timestep = 0.0001
custom_strategy = CustomPlanningStrategy()

f110_gym = F110GymBase.F110GymBase(map, num_agents, conf, custom_strategy, timestep)

f110_gym.start()