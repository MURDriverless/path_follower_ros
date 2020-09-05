#!/usr/bin/env python

import rospy
from src.pid_pure_pursuit import PIDPurePursuit
from src.msgs import SlamData, PlannerData, ActuationData


class PathFollowerNode:
    def __init__(self):
        self.state = None
        self.path = None
        self.controller = PIDPurePursuit()
        self.slam_sub = rospy.Subscriber("mur_slam", SlamData, self.slam_callback)
        self.planner_sub = rospy.Subscriber("mur_planner", PlannerData, self.planner_callback)
        self.actuation_pub = rospy.Publisher("mur_actuation", ActuationData)

    def update(self):
        # Compute control action
        acc_threshold, steering = self.controller.control(self.state, self.path.nodes)
        # Construct message
        msg = ActuationData()
        msg.acceleration_threshold = acc_threshold
        msg.steering = steering
        # Publish actuation command
        self.actuation_pub.publish(acc_threshold, steering)

    def slam_callback(self, msg):
        self.state = msg.slam

    def planner_callback(self, msg):
        self.path = msg.planner


def run_node():
    rospy.init_node("mur_follower")
    rospy.spin()
