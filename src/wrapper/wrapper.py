import gym
import numpy as np
from gym import spaces
from pathlib import Path
import yaml
from argparse import Namespace
from pyglet.gl import GL_POINTS
import utility.map_utility as map_utility



def get_distance_from_closest_point(x_list, w_list, x, y, index):
    closest_x = x_list[index]
    closest_y = w_list[index]
    distance = np.sqrt(np.power(x - closest_x, 2) + np.power(y - closest_y, 2))
    return distance


def convert_range(value, input_range, output_range):
    # converts value(s) from range to another range
    # ranges ---> [min, max]
    (in_min, in_max), (out_min, out_max) = input_range, output_range
    in_range = in_max - in_min
    out_range = out_max - out_min
    return (((value - in_min) * out_range) / in_range) + out_min


class F110_Wrapped(gym.Wrapper):
    """
    This is a wrapper for the F1Tenth Gym environment intended
    for only one car, but should be expanded to handle multi-agent scenarios
    """

    def __init__(self, env, random_map=False):
        super().__init__(env)

        # normalised action space, steer and speed
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float)

        # normalised observations, just take the lidar scans
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1080,), dtype=np.float)

        # store allowed steering/speed/lidar ranges for normalisation
        self.s_min = self.env.params['s_min']
        self.s_max = self.env.params['s_max']
        self.v_min = self.env.params['v_min']
        self.v_max = self.env.params['v_max']
        self.lidar_min = 0
        self.lidar_max = 30  # see ScanSimulator2D max_range

        # store car dimensions and some track info
        self.car_length = self.env.params['length']
        self.car_width = self.env.params['width']
        self.track_width = 3.2  # ~= track width, see random_trackgen.py

        # radius of circle where car can start on track, relative to a centerpoint
        self.start_radius = (self.track_width / 2) - ((self.car_length + self.car_width) / 2)  # just extra wiggle room

        self.step_count = 0

        # set threshold for maximum angle of car, to prevent spinning
        self.max_theta = 100
        self.count = 0

        self.map_path = None
        self.random_map = random_map

        self.race_line_color = [255, 0, 0]
        self.race_line_x = []
        self.race_line_y = []

    def set_raceliens(self):
        if self.map_path is not None:
            raceline = map_utility.get_raceline(self.map_path)
            self.race_line_x, self.race_line_y, _ = map_utility.get_x_y_theta_from_raceline(raceline)
            self.race_line_x = self.race_line_x.tolist()
            self.race_line_y = self.race_line_y.tolist()

    def set_map_path(self, map_path):
        self.map_path = map_path
        self.set_raceliens()


    def start_position(self):
        if self.map_path is not None:
            x, y, t = map_utility.get_start_position(self.map_path)
            self.set_raceliens()
            return x, y, t
        else:
            raise Exception("Map path not set")



    def step(self, action):
        action_convert = self.un_normalise_actions(action)
        observation, _, done, info = self.env.step(np.array([action_convert]))

        self.step_count += 1

        next_x = self.race_line_x[self.count]
        next_y = self.race_line_y[self.count]

        if self.env.renderer:
            if done:
                self.race_line_color = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]
            self.env.renderer.batch.add(1, GL_POINTS, None, ('v3f/stream', [next_x * 50, next_y * 50, 0.]),
                                        ('c3B/stream', self.race_line_color))
        reward = 0
        reward = reward + 0.9

        if self.count < len(self.race_line_x) - 1:
            X, Y = observation['poses_x'][0], observation['poses_y'][0]

            dist = np.sqrt(np.power((X - next_x), 2) + np.power((Y - next_y), 2))
            if dist < 2:
                self.count += 1

            if dist < 2.5:
                complete = (self.count / len(self.race_line_x)) * 0.5
                reward += complete

        else:
            self.count = 0
            reward += 100
            print("Lap Done")

        if observation['collisions'][0]:
            self.count = 0
            reward = -2

        if abs(observation['poses_theta'][0]) > self.max_theta:
            done = True

        if self.env.lap_counts[0] > 0:
            self.count = 0
            reward += 1
            if self.env.lap_counts[0] > 1:
                reward += 1
                self.env.lap_counts[0] = 0

        return self.normalise_observations(observation['scans'][0]), reward, bool(done), info

    def reset(self):
        if self.random_map:
            path = map_utility.get_one_random_map()
            map_path = map_utility.get_formatted_map(path)
            print("Rand Map :", map_path)
            map_ext = map_utility.map_ext
            self.update_map(map_path, map_ext, update_render=True)
            self.set_map_path(path)

        x, y, t = self.start_position()
        observation, _, _, _ = self.env.reset(np.array([[x, y, t]]))
        return self.normalise_observations(observation['scans'][0])

    def update_map(self, map_name, map_extension, update_render=True):
        self.env.map_name = map_name
        self.env.map_ext = map_extension
        self.env.update_map(f"{map_name}.yaml", map_extension)
        if update_render and self.env.renderer:
            self.env.renderer.close()
            self.env.renderer = None

    def seed(self, seed):
        self.current_seed = seed
        np.random.seed(self.current_seed)
        print(f"Seed -> {self.current_seed}")

    def un_normalise_actions(self, actions):
        # convert actions from range [-1, 1] to normal steering/speed range
        steer = convert_range(actions[0], [-1, 1], [self.s_min, self.s_max])
        speed = convert_range(actions[1], [-1, 1], [self.v_min, self.v_max])
        return np.array([steer, speed], dtype=np.float)

    def normalise_observations(self, observations):
        # convert observations from normal lidar distances range to range [-1, 1]
        return convert_range(observations, [self.lidar_min, self.lidar_max], [-1, 1])
