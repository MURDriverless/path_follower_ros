#!/usr/bin/env python

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from mur_common.msg import cone_msg as ConeData
from mur_common.msg import actuation_msg as ActuationData
from mur_common.msg import planner_msg as PlannerData
from src.pid_pure_pursuit import PIDPurePursuit


class PathFollowerNode:
    left_cone_colour = "blue"
    right_cone_colour = "yellow"

    def __init__(self):
        self.state = None
        self.path_nodes = None
        self.left_cones = None
        self.right_cones = None
        self.controller = PIDPurePursuit()
        self.odom_sub = rospy.Subscriber("~odom", Odometry, self.odom_callback)
        self.cone_sub = rospy.Subscriber("mur_slam", ConeData, self.cone_callback)
        self.planner_sub = rospy.Subscriber("mur_planner", PlannerData, self.planner_callback)
        self.actuation_pub = rospy.Publisher("mur_actuation", ActuationData)

    def update(self):
        # Compute control action
        acc_threshold, steering = self.controller.control(self.state, self.path_nodes)
        # Construct message
        msg = ActuationData()
        msg.acceleration_threshold = acc_threshold
        msg.steering = steering
        # Publish actuation command
        self.actuation_pub.publish(acc_threshold, steering)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        v = np.hypot(vx, vy)
        # From https://github.com/tsrour/aionr6-mpc-ros/blob/master/ltvcmpc_controller/scripts/mpcc_control_node
        # line 85-87
        z_measure = msg.pose.pose.orientation.z
        w_measure = msg.pose.pose.orientation.w
        yaw = 2 * np.arcsin(abs(z_measure)) * np.sign(z_measure) * np.sign(w_measure)
        # Update the current state measurements
        self.state = np.array([x, y, v, yaw])

    def cone_callback(self, msg):
        xs = msg.x
        ys = msg.y
        colours = msg.colour
        left_cones = []
        right_cones = []
        # Place each cone in either left or right cones depending on their colour
        for i in range(len(xs)):
            if colours[i] == self.left_cone_colour:
                left_cones.append([xs[i], ys[i]])
            elif colours[i] == self.right_cone_colour:
                right_cones.append([xs[i], ys[i]])
        # Create a numpy array as it is more compact and gets accessed faster (supposedly)
        self.left_cones = np.array(left_cones)
        self.right_cones = np.array(right_cones)

    def planner_callback(self, msg):
        self.path_nodes = msg.path_nodes


def run_node():
    rospy.init_node("mur_follower")
    rospy.spin()
