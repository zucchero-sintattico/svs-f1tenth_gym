import gym
import numpy as np
from gym import spaces
from pyglet.gl import GL_POINTS
import utility.map_utility as map_utility
from typing import List
import csv

def logger(map, event, reword, lap_time):
    """
    Logs the map, event, reward, and lap time in a CSV file.

    :param map: The name of the map.
    :param event: The event that occurred.
    :param reward: The reward received.
    :param lap_time: The lap time.
    """
    map = map.split("/")[-1]
    reword = round(reword, 6)
    lap_time = round(lap_time, 6)
    with open('log.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([map, event, reword, lap_time])

def convert_range(value, input_range, output_range):
    """
    Converts a value from one range to another range.

    :param value: The value to be converted.
    :param input_range: The input range [min, max].
    :param output_range: The output range [min, max].
    :return: The converted value.
    """
    (in_min, in_max), (out_min, out_max) = input_range, output_range
    in_range = in_max - in_min
    out_range = out_max - out_min
    return (((value - in_min) * out_range) / in_range) + out_min

class F110_Wrapped(gym.Wrapper):
    """
    This is a wrapper for the F1Tenth Gym environment intended
    for only one car, but should be expanded to handle multi-agent scenarios.
    """

    def __init__(self, env, random_map=False):
        super().__init__(env)
        self.optimize_speed = False

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
        self.count = 0
        self.step_for_episode = 0

        # set threshold for maximum angle of car, to prevent spinning
        self.max_theta = 100


        self.map_path = None
        self.random_map = random_map

        self.race_line_color = [255, 0, 0]
        self.race_line_x = []
        self.race_line_y = []
        self.race_line_theta = []

        self.episode_returns = []
        self.is_rendering = False
        self.last_position = {'x': None, 'y': None}
        self.number_of_base_reward_give = 10
        self.one_lap_done = False

    def get_total_steps(self) -> int:
        """
        Returns the total number of steps taken in the environment.

        :return: The total number of steps.
        """
        return self.step_count

    def get_episode_rewards(self) -> List[float]:
        """
        Returns the rewards of all the episodes.

        :return: The rewards of all the episodes.
        """
        return self.episode_returns

    def set_optimize_speed(self, optimize_speed: bool) -> None:
        """
        Sets the optimize speed flag.

        :param optimize_speed: The flag to optimize speed.
        """
        self.optimize_speed = optimize_speed

    def set_raceliens(self):
        """
        Sets the race line coordinates based on the map path.
        """
        if self.map_path is not None:
            raceline = map_utility.get_raceline(self.map_path)
            self.race_line_x, self.race_line_y, self.race_line_theta = map_utility.get_x_y_theta_from_raceline(raceline)
            self.race_line_x = self.race_line_x.tolist()
            self.race_line_y = self.race_line_y.tolist()
            self.race_line_theta = self.race_line_theta.tolist()

    def set_map_path(self, map_path):
        """
        Sets the map path.

        :param map_path: The path of the map.
        """
        self.map_path = map_path
        self.set_raceliens()

    def start_position(self):
        """
        Returns the start position of the car on the track.

        :return: The start position (x, y, theta).
        """
        if self.map_path is not None:
            x, y, t = map_utility.get_start_position(self.map_path)
            self.set_raceliens()
            return x, y, t
        raise Exception("Map path not set")

    def step(self, action):
        """
        Performs a step in the environment.

        :param action: The action to take.
        :return: The next observation, reward, done flag, and additional info.
        """
        def episode_end(reason = None, rew = 0):
            if reason is not None:
                print("Episode End ->", reason, self.map_path)

            done = True
            self.count = 0
            self.episode_returns = []
            self.step_for_episode = 0
            self.one_lap_done = False
            return done, rew

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

        aceleration_reward = action_convert[1]

        if self.optimize_speed:
            if aceleration_reward > 2:
                reward += aceleration_reward
            else:
                reward += 2

        reward = reward * 0.01

        if self.count < len(self.race_line_x) - 1:
            X, Y = observation['poses_x'][0], observation['poses_y'][0]

            dist = np.sqrt(np.power((X - next_x), 2) + np.power((Y - next_y), 2))
            if dist < 2:
                self.count = self.count + 1
                reward +=  0.01
                self.number_of_base_reward_give = 0

            if dist < 2.5:
                self.number_of_base_reward_give += 1

                if self.number_of_base_reward_give < 100:
                    reward += 0.05 
                else:
                    reward -= 1

            if dist > 3: 
                reward -= 1

        else:  
            if self.optimize_speed:
                steps_goal = self.count       
                if  not self.one_lap_done:
                    steps_done = self.step_for_episode          
                elif self.one_lap_done:
                    steps_done = self.step_for_episode / 2   

                k = (steps_done - steps_goal)/steps_goal

                reward += (1-k) * 100 

            print("----------------- Lap Done ----------------->", self.map_path, self.step_for_episode * 0.01)

            self.count = 0

            if self.one_lap_done:
                logger(self.map_path, "lap_done", sum(self.episode_returns), self.step_for_episode * 0.01)
                self.episode_returns = []
                self.step_for_episode = 0
                self.one_lap_done = False
            else:
                self.one_lap_done = True

        reward = round(reward, 6)

        if observation['collisions'][0]:
            logger(self.map_path, "collisions", sum(self.episode_returns), self.step_for_episode * 0.01)
            done, reward = episode_end(rew = -30)

        if self.step_for_episode > 50_000:
            logger(self.map_path, "too_slow", sum(self.episode_returns), self.step_for_episode * 0.01)
            done, reward = episode_end("Too long", -10)

        self.episode_returns.append(reward)
        self.step_for_episode += 1

        return self.normalise_observations(observation['scans'][0]), reward, bool(done), info

    def reset(self):
        """
        Resets the environment and returns the initial observation.

        :return: The initial observation.
        """
        if self.random_map:
            path = map_utility.get_one_random_map()
            map_path = map_utility.get_formatted_map(path)
            map_ext = map_utility.map_ext
            self.update_map(map_path, map_ext, update_render=True)
            self.set_map_path(path)

        # Select a random point from the race line
        race_line_x = self.race_line_x
        race_line_y = self.race_line_y
        race_line_theta = self.race_line_theta
        random_index = np.random.randint(len(race_line_x))
        x = race_line_x[random_index]
        y = race_line_y[random_index]
        t = race_line_theta[random_index]

        # Update the race line to start from the selected point
        race_line_x = race_line_x[random_index:] + race_line_x[:random_index]
        race_line_y = race_line_y[random_index:] + race_line_y[:random_index]
        race_line_theta = race_line_theta[random_index:] + race_line_theta[:random_index]

        self.race_line_x = race_line_x
        self.race_line_y = race_line_y
        self.race_line_theta = race_line_theta

        self.episode_returns = []
        self.step_for_episode = 0
        self.last_position = {'x': x, 'y': y}

        observation, _, _, _ = self.env.reset(np.array([[x, y, t]]))
        return self.normalise_observations(observation['scans'][0])

    def update_map(self, map_name, map_extension, update_render=False):
        """
        Updates the map of the environment.

        :param map_name: The name of the map.
        :param map_extension: The extension of the map file.
        :param update_render: Flag to update the rendering.
        """
        self.env.map_name = map_name
        self.env.map_ext = map_extension
        self.env.update_map(f"{map_name}.yaml", map_extension)

    def seed(self, seed):
        """
        Sets the seed for random number generation.

        :param seed: The seed value.
        """
        self.current_seed = seed
        np.random.seed(self.current_seed)
        print(f"Seed -> {self.current_seed}")

    def un_normalise_actions(self, actions):
        """
        Converts actions from the range [-1, 1] to the normal steering/speed range.

        :param actions: The actions to be converted.
        :return: The converted actions.
        """
        steer = convert_range(actions[0], [-1, 1], [self.s_min, self.s_max])
        speed = convert_range(actions[1], [-1, 1], [self.v_min, self.v_max])
        return np.array([steer, speed], dtype=np.float)

    def normalise_observations(self, observations):
        """
        Converts observations from the normal lidar distances range to the range [-1, 1].

        :param observations: The observations to be converted.
        :return: The converted observations.
        """
        return convert_range(observations, [self.lidar_min, self.lidar_max], [-1, 1])

