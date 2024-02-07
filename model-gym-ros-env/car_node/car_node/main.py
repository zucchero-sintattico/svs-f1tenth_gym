#!/usr/bin/env python
# from __future__ import print_function
import numpy as np
import rclpy
from rclpy.node import Node

#ROS Imports
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

from stable_baselines3 import PPO

device = 'cpu'

STEERING_RANGE_MIN = -0.4189
STEERING_RANGE_MAX = 0.4189
SPEED_RANGE_MIN = -5
SPEED_RANGE_MAX = 20
MODEL_INTERVAL = [-1, 1]

class PPOModelEvaluator(Node):

    def __init__(self):
        super().__init__('car_node')

        #Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        self.model = PPO.load("/sim_ws/src/car_node/car_node/models/YasMarina_optimize_model.zip", device=device)

        # ROS 
        self.lidar_sub = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

    def lidar_callback(self, data):
        d = np.array(data.ranges, dtype=np.float64)
        d = convert_range(d, [data.range_min, data.range_max], MODEL_INTERVAL)

        action = self.model.predict(d, deterministic=True)[0]
        steer = convert_range(action[0], MODEL_INTERVAL, [STEERING_RANGE_MIN, STEERING_RANGE_MAX])
        speed = convert_range(action[1], MODEL_INTERVAL, [SPEED_RANGE_MIN, SPEED_RANGE_MAX])

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "laser"
        drive_msg.drive.steering_angle = steer
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PPOModelEvaluator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
