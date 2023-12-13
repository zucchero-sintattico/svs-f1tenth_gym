import F110GymBase
import yaml
from argparse import Namespace
import numpy as np
from F110GymBase import PlanningStrategy

with open('src/map/example_map/config_example_map.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)



class CustomPlanningStrategy(PlanningStrategy):
    def plan(self, observation):
        action = np.array([[0., 0.8]])
        return action


map = 'example_map'
num_agents = 1
timestep = 0.01
custom_strategy = CustomPlanningStrategy()

f110_gym = F110GymBase.F110GymBase(map, num_agents, conf, custom_strategy, timestep)

f110_gym.start()