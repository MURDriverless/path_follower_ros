"""
Path tracking simulation with pure pursuit steering and PID speed control.
author: Atsushi Sakai (@Atsushi_twi)
        Guillaume Jacquenot (@Gjacquenot)
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from pid import pid
from rospy import Publisher, Time
from geometry_msgs.msg import PoseStamped

# Parameters
k = 0.1  # look forward gain
Lfc = 3.5  # [m] look-ahead distance

dt = 0.05  # [s] time tick
WB = 2.951  # [m] wheel base of vehicle

Kp = 1
Ki = 0.01
Kd = 0.5
PID = pid(Kp, Ki, Kd, 0.05, 5)

show_animation = True


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def update(self, a, delta):
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WB * math.tan(delta) * dt
        self.v += a * dt
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def raw_update(self, state):
        self.x = state[0]
        self.y = state[1]
        self.v = state[2]
        self.yaw = state[3]
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)


class States:

    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []

    def append(self, t, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.t.append(t)


def proportional_control(target, current):
    # a = Kp * (target - current)
    #
    # return a
    return PID.control(target - current)


class TargetCourse:

    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state_obj):

        # To speed up nearest point search, doing it at only first time.
        if self.old_nearest_point_index is None:
            # search nearest point index
            dx = [state_obj.rear_x - icx for icx in self.cx]
            dy = [state_obj.rear_y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state_obj.calc_distance(self.cx[ind],
                                                          self.cy[ind])
            while True:
                distance_next_index = state_obj.calc_distance(self.cx[ind + 1],
                                                              self.cy[ind + 1])
                if distance_this_index < distance_next_index or (ind+1) > len(self.cx):
                    break
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        Lf = k * state_obj.v + Lfc  # update look ahead distance

        # search look ahead target point index
        while Lf > state_obj.calc_distance(self.cx[ind], self.cy[ind]):
            # If we've reached the end of the track (since our track is circular,
            # this will be the starting point), set the lookahead point to be
            # approximately one cone away from the starting point
            if (ind + 1) >= len(self.cx):
                ind = 1
                break
            ind += 1

        return ind, Lf


def pure_pursuit_steer_control(state_obj, trajectory, pind):
    ind, Lf = trajectory.search_target_index(state_obj)

    if pind >= ind:
        ind = pind

    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
    else:  # toward goal
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1

    alpha = math.atan2(ty - state_obj.rear_y, tx - state_obj.rear_x) - state_obj.yaw

    delta = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0)

    return delta, ind


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """
    Plot arrow
    """

    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


class PyRoboticsPurePursuit:
    def __init__(self, state):
        self.target_node_pub = Publisher(
            "TargetNode", PoseStamped, queue_size=10)
        self.state_obj = State(state[0], state[1], state[3], state[2])
        self.target_ind = 0
        self.dist_travelled = 0.0
        self.just_started = True

    def control(self, state, nodes):
        # If planner is not ready, don't increment distance
        if len(nodes) > 0:
            self.dist_travelled += np.hypot(self.state_obj.x - state[0], self.state_obj.y - state[1])
        self.state_obj.raw_update(state)
        cx = [node[0] for node in nodes]
        cy = [node[1] for node in nodes]
        target_course = TargetCourse(cx, cy)

        last_index = len(nodes) - 1

        # If we have covered at least 20 metres, we are already quite far in the lap
        if self.dist_travelled >= 20.0:
            self.just_started = False

        # If we have finished mapping, whereby the last point and the first point are identical,
        # we want to make the car stop at the origin
        if nodes[last_index][0] == nodes[0][0] and nodes[last_index][1] == nodes[0][1]:
            # We set the second point as our "end point", because if we set the starting point instead,
            # if we are 0.5 behind the finish line, the car will not move. Tested on the default track in ROS,
            # 0.5m before the second point puts us at the starting point
            dist_to_end = np.hypot(cx[1] - self.state_obj.x, cy[1] - self.state_obj.y)

            self.target_ind, Lf = target_course.search_target_index(self.state_obj)

            # Additional check so that the car still moves when starting out
            # (further inspection will reveal that just_started is not used at all
            # as by the time we have fully mapped the track, we're past 20m)
            #
            # If the distance from car to the "finish" target point is lesser than 0.5m,
            # we can stop the car. Remember to flush the PID errors so the car stops completely
            if self.just_started is False and (dist_to_end <= 0.5):
                ai = proportional_control(0.0, state[2])
                PID.ei = 0.0
                PID.ep = 0.0
            else:
                ai = proportional_control(nodes[self.target_ind][2], state[2])
        else:
            ai = proportional_control(nodes[self.target_ind][2], state[2])

        di, self.target_ind = pure_pursuit_steer_control(self.state_obj, target_course, self.target_ind)

        # Publish lookahead point
        ps = PoseStamped()
        ps.pose.position.x = target_course.cx[self.target_ind]
        ps.pose.position.y = target_course.cy[self.target_ind]
        # ps.pose.position.x = state[0]
        # ps.pose.position.y = state[1]
        ps.header.frame_id = "map"
        ps.header.seq = 1
        ps.header.stamp = Time.now()
        self.target_node_pub.publish(ps)

        return ai, di
