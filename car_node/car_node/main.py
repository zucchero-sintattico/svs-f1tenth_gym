#!/usr/bin/env python
# from __future__ import print_function
import sys
import math
import numpy as np
import rclpy
from rclpy.node import Node

#ROS Imports
from sensor_msgs.msg import Image, LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


import gym
from stable_baselines3 import PPO
from car_node.wrapper import F110_Wrapped
from f110_gym.envs.base_classes import Integrator
from stable_baselines3.common.evaluation import evaluate_policy
from car_node.map_utility import get_map, get_formatted_map, map_ext
from car_node.wrapper import convert_range
import torch

timestep = 0.01
tensorboard_path = './train_test/'
map = "BrandsHatch"
path = get_map(map)
map_path = get_formatted_map(path)
device = 'cpu'

class WallFollow(Node):
    """ Implement Wall Following on the car
    """
    def __init__(self):
        super().__init__('wall_follow_node')
        #Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        self.eval_env  = gym.make('f110_gym:f110-v0', map=map_path, map_ext=map_ext, num_agents=1, timestep=timestep, integrator=Integrator.RK4)
        self.eval_env = F110_Wrapped(self.eval_env, random_map=False)
        self.eval_env.set_map_path(path)
        self.model = PPO.load("/sim_ws/src/car_node/car_node/train_test/best_global_model", self.eval_env, device=device)
        # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
        # print(mean_reward)
        # Enjoy trained agent
        self.vec_env = self.model.get_env()
        obs = self.vec_env.reset()

        self.lidar_sub = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

    def pid_control(self, error, velocity):

        angle = 0 # model.predict()

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "laser"
        drive_msg.drive.steering_angle = angle
        drive_msg.drive.speed = velocity
        # prev_error = error

        # remove the oldest error
        prev_error = np.delete(prev_error, 0)
        # add the newest error
        prev_error = np.append(prev_error, error)

        self.drive_pub.publish(drive_msg)

    def followLeft(self, data, leftDist):
        #Follow left wall as per the algorithm 
        #TODO:implement
        return 0.0 

    def lidar_callback(self, data):
        # steer = convert_range(data.ranges, [-1, 1], [-1, 1])

        # data = np.array(data.ranges, dtype=np.float64)
        # data = data.reshape(1, -1)
        # data = torch.tensor(data.ranges, dtype=torch.float32)
        d = np.zeros((1,1080))
        
        self.get_logger().info(f'{data.shape}')
        self.get_logger().info(f'{data}')
        
        # speed = convert_range(actions[1], [-1, 1], [self.v_min, self.v_max])
        # return np.array([steer, speed], dtype=np.float)

        action, _states = self.model.predict(d, deterministic=True)
        # obs, rewards, dones, info = self.vec_env.step(action)
        # if done:
        #     print("-->", episode)
        # eval_env.render(mode='human')
        

        self.get_logger().info(f'{action}')
        # self.get_logger().info(f'{data.ranges}')
        self.get_logger().info(f'{len(data.ranges)}')

def main(args=None):
    rclpy.init(args=args)
    node = WallFollow()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()



if __name__=='__main__':
	# main(sys.argv)
    main()







# episode = 0
# while episode < 100:

#     episode += 1
#     obs = vec_env.reset()
#     done = False
#     while not done:
#         action, _states = model.predict(obs, deterministic=True)
#         obs, rewards, dones, info = vec_env.step(action)
#         if done:
#             print("-->", episode)
#         eval_env.render(mode='human')
