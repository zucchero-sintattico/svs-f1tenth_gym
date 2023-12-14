import gym
import numpy as np
from f110_gym.envs.base_classes import Integrator


class PlanningStrategy:
    def plan(self, observation):
        raise NotImplementedError("Subclasses must implement the plan method")

class DefaultPlanningStrategy(PlanningStrategy):
    def plan(self, observation):
        # Default planning strategy, you can modify this as needed
        action = np.array([[0., 0.0]])
        return action


class F110GymBase:

    def __init__(self, map, num_agents, conf, planning_strategy, timestep=0.001):
        self.map = map
        self.num_agents = num_agents
        self.env  = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=timestep, integrator=Integrator.RK4)
        self.obs, self.step_reward, self.done, self.info = self.env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
        self.env.render()
        self.conf = conf
        self.lap_time = 0.
        self.planning_strategy = planning_strategy


    def plan(self, observation):
        return self.planning_strategy.plan(observation)


    def start (self):
        while not self.done:
            # get action based on the observation
            actions = self.plan(self.obs)

            # stepping through the environment
            self.obs, self.step_reward, self.done, self.info = self.env.step(actions)

            


            if self.done:
                self.obs, self.step_reward, self.done, self.info = self.env.reset(np.array([[self.conf.sx, self.conf.sy, self.conf.stheta]]))

            self.env.render(mode='human')

            self.lap_time += self.step_reward

        print('Lap time: ', self.lap_time)














