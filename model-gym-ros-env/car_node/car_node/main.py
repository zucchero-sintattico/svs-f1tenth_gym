#!/usr/bin/env python
# from __future__ import print_function
import numpy as np
import rclpy
from rclpy.node import Node

#ROS Imports
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


import gym
from stable_baselines3 import PPO
from car_node.wrapper import F110_Wrapped
from f110_gym.envs.base_classes import Integrator
from car_node.map_utility import get_map, get_formatted_map, map_ext
from car_node.wrapper import convert_range

timestep = 0.01
tensorboard_path = './train_test/'
map = "BrandsHatch"
path = get_map(map)
map_path = get_formatted_map(path)
device = 'cpu'

STEER_SCALE = 0.5
SPEED_SCALE = 0.7

class PPOModelEvaluator(Node):
    
    def __init__(self):
        super().__init__('wall_follow_node')
        
        #Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # Model & Env
        self.eval_env  = gym.make('f110_gym:f110-v0', map=map_path, map_ext=map_ext, num_agents=1, timestep=timestep, integrator=Integrator.RK4)
        self.eval_env = F110_Wrapped(self.eval_env, random_map=False)
        self.eval_env.set_map_path(path)
        self.model = PPO.load("/sim_ws/src/car_node/car_node/train_test/best_global_model", self.eval_env, device=device)
        self.vec_env = self.model.get_env()

        # ROS 
        self.lidar_sub = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

    def lidar_callback(self, data):
        d = np.array(data.ranges, dtype=np.float64)
        d = convert_range(d, [data.angle_min, data.angle_max], [-1, 1])
        
        action, _states = self.model.predict(d, deterministic=True)
        action = self.eval_env.un_normalise_actions(action)
        
        self.get_logger().info(f'{action[0], action[1]}')
        
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "laser"
        drive_msg.drive.steering_angle = action[0] * STEER_SCALE
        drive_msg.drive.speed = action[1] * SPEED_SCALE
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PPOModelEvaluator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
