#!/usr/bin/env python

import numpy as np
import math
import rospy
from nav_msgs.msg import Odometry
from mur_common.msg import cone_msg as ConeData
from mur_common.msg import actuation_msg as ActuationData
from mur_common.msg import path_msg as PathData
from pid_pure_pursuit import PIDPurePursuit


class PathFollower:
    left_cone_colour = "BLUE"
    right_cone_colour = "YELLOW"

    def __init__(self):
        self.controller = PIDPurePursuit()
        # From Odometry
        self.state = None
        # From SLAM
        self.left_cones = None
        self.right_cones = None
        # From Path Planner
        self.path_nodes = None
        # For Actuation
        self.actuation_pub = None
        self.state = np.array([0, 0, 0, 0])
        self.left_cones = []
        self.right_cones = []

    def set_actuation_pub(self, actuation_pub):
        self.actuation_pub = actuation_pub

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        v = math.sqrt(vx**vx + vy**vy)
        # From https://github.com/tsrour/aionr6-mpc-ros/blob/master/ltvcmpc_controller/scripts/mpcc_control_node
        # line 85-87
        z_measure = msg.pose.pose.orientation.z
        w_measure = msg.pose.pose.orientation.w
        yaw = 2 * np.arcsin(abs(z_measure)) * \
            np.sign(z_measure) * np.sign(w_measure)
        # Update the current state measurements
        self.state = np.array([x, y, v, yaw])

    def publishControl(self):
        # Compute control action
        if (len(self.left_cones) == 0 or len(self.right_cones) == 0):
            return
        acc_threshold, steering = self.controller.control(
            self.state, self.path_nodes)
        # Construct message
        message = ActuationData()
        message.acceleration_threshold = acc_threshold
        message.steering = steering
        # Publish actuation command
        self.actuation_pub.publish(message)
        return

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
        self.path_nodes = [(x, y, v) for (x, y, v) in zip(msg.x, msg.y, msg.v)]


def run_node():
    # Initialise node
    rospy.init_node("mur_follower")

    follower = PathFollower()

    # Odometry subscriber
    rospy.Subscriber("/mur/slam/Odom", Odometry, follower.odom_callback)

    # Cone data subscriber
    rospy.Subscriber("mur/slam/cones", ConeData, follower.cone_callback)

    # Path Planner subscriber
    rospy.Subscriber("/mur/planner/path", PathData, follower.planner_callback)

    # Actuation publisher
    actuation_pub = rospy.Publisher(
        "mur/control/actuation", ActuationData, queue_size=10)
    follower.set_actuation_pub(actuation_pub)

    # Run the node forever
    rate = rospy.Rate(20)
    while(not rospy.is_shutdown()):
        follower.publishControl()
        rate.sleep()
    rospy.spin()


if __name__ == "__main__":
    run_node()
