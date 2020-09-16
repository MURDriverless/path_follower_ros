#!/usr/bin/env python

import numpy as np
import math
import rospy
from nav_msgs.msg import Odometry
from mur_common.msg import cone_msg as ConeData
from mur_common.msg import actuation_msg as ActuationData
from mur_common.msg import path_msg as PathData
from pid_pure_pursuit import PIDPurePursuit
from pyrobotics_pure_pursuit import PyRoboticsPurePursuit
from nav_msgs.msg import Path
from cubic_spline import Spline2D

from geometry_msgs.msg import PoseStamped, Pose


class PathFollower:
    left_cone_colour = "BLUE"
    right_cone_colour = "YELLOW"

    def __init__(self):
        # self.controller = PIDPurePursuit()
        self.controller = PyRoboticsPurePursuit([0.0, 0.0, 0.0, 0.0])
        # From Odometry
        self.state = None
        # From SLAM
        self.left_cones = []
        self.right_cones = []
        # From Path Planner
        self.path_nodes = []
        # For Actuation
        self.actuation_pub = None
        self.state = np.array([0, 0, 0, 0])
        self.pathPub = rospy.Publisher("Path", Path, queue_size=10)

    def set_actuation_pub(self, actuation_pub):
        self.actuation_pub = actuation_pub

    def publishPath(self):
        msg = Path()
        poses = []
        ps = PoseStamped()
        ps.pose.position.x = self.state[0]
        ps.pose.position.y = self.state[1]
        poses.append(ps)
        for pose in self.path_nodes:
            ps = PoseStamped()
            ps.pose.position.x = pose[0]
            ps.pose.position.y = pose[1]
            ps.header.frame_id = "map"
            ps.header.seq = 1
            ps.header.stamp = rospy.Time.now()
            poses.append(ps)
        msg.poses = poses
        msg.header.seq = 1
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        self.pathPub.publish(msg)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        # From https://github.com/tsrour/aionr6-mpc-ros/blob/master/ltvcmpc_controller/scripts/mpcc_control_node
        # line 85-87
        z_measure = msg.pose.pose.orientation.z
        w_measure = msg.pose.pose.orientation.w
        yaw = 2 * np.arcsin(abs(z_measure)) * \
            np.sign(z_measure) * np.sign(w_measure)
        # v = vx * math.cos(yaw) + vy * math.sin(yaw)
        v = np.hypot(vx, vy)
        # Update the current state measurements
        self.state = np.array([x, y, v, yaw])

    def publishControl(self):
        # Compute control action
        if (len(self.left_cones) == 0 or len(self.right_cones) == 0):
            return
        if (len(self.path_nodes) == 0):
            return
        acc_threshold, steering = self.controller.control(
            self.state, self.path_nodes)

        if acc_threshold > 0 and acc_threshold > 0.3:
            acc_threshold = 0.3

        # Construct message
        print("Steering message")
        print(acc_threshold, steering)
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
                left_cones.append((xs[i], ys[i]))
            elif colours[i] == self.right_cone_colour:
                right_cones.append((xs[i], ys[i]))
        # Create a numpy array as it is more compact and gets accessed faster (supposedly)
        self.left_cones = np.array(left_cones)
        self.right_cones = np.array(right_cones)

    def planner_callback(self, msg):
        spline = Spline2D(msg.x, msg.y)
        max_distance = spline.t[-1]
        step_size = 0.1
        interval = np.arange(0, max_distance, step_size)
        positions = [spline.interpolate(t) for t in interval]
        self.path_nodes = [(positions[i][0], positions[i][1], 0.5) for i in range(len(interval))]
        # self.path_nodes = [(x, y, v) for (x, y, v) in zip(msg.x, msg.y, msg.v)]
        # self.plan_path()
        # print(self.path_nodes)
        # pass

    def plan_path(self):
        associates = []
        possible = []
        for rcone in self.right_cones:
            print(rcone)

            tmp = (rcone[0], rcone[1])
            possible.append(tmp)
            lcone = self.getNearestFriendo(rcone)
            tmp = (lcone[0], lcone[1])
            associates.append(tmp)
        midpoints = []
        print("Associates")
        print(associates)

        for i, cone in enumerate(self.right_cones):
            midpoints.append(((cone[0] + associates[i][0])/2,
                              (cone[1] + associates[i][1])/2))
        midpoints = list(midpoints)
        midpoints.sort(key=lambda pt: abs(
            self.state[1]-pt[1]) + abs(self.state[0]-pt[0]), reverse=False)
        # nodes = []
        self.path_nodes = []
        for point in midpoints:
            if self.infront(point):
                continue
            # nodes.append((point[0], point[1], 4.0))
            self.path_nodes.append((point[0], point[1], 0.5))

        # nodes_x = [node[0] for node in nodes]
        # nodes_y = [node[1] for node in nodes]
        # path_spline = Spline2D(nodes_x, nodes_y)
        # max_distance = 3
        # step_size = 0.2
        # interval = np.arange(0, max_distance, step_size)
        # points = [path_spline.interpolate(i) for i in interval]
        # self.path_nodes = [(point[0], point[1], 4.0) for point in points]
        self.publishPath()

    def getNearestFriendo(self, rcone):
        min_dist = float('inf')
        min_cone = None
        for lcone in self.left_cones:
            dist = math.sqrt((rcone[0] - lcone[0]) **
                             2 + (rcone[1] - lcone[1])**2)
            if dist < min_dist:
                min_dist = dist
                min_cone = lcone
        return min_cone

    def infront(self, cone):
        x = self.state[0]
        y = self.state[1]
        dist = math.sqrt((x-cone[0])**2 + (y-cone[1])**2)
        if dist < 0.3:
            return True
        theta = self.state[3]
        print("theta:", theta)

        angle = pi_2_pi(math.atan2(cone[0]-x, cone[1] - y))
        print("Angle:", angle)

        if pi_2_pi(angle - theta - math.pi/2) < math.pi/2:
            return False
        return True


def pi_2_pi(angle):
    return (angle + np.pi) % (2*np.pi)-np.pi


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
        # follower.plan_path()
        follower.publishControl()
        rate.sleep()
    rospy.spin()


if __name__ == "__main__":
    run_node()
