"""
Path tracking simulation with Stanley steering control and PID speed control.
author: Atsushi Sakai (@Atsushi_twi)
Ref:
    - [Stanley: The robot that won the DARPA grand challenge](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
    - [Autonomous Automobile Path Tracking](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import cubic_spline as cubic_spline_planner
from pid import pid
from rospy import Publisher, Time
from geometry_msgs.msg import PoseStamped



k = 0.5  # control gain
# Kp = 3.0  # speed proportional gain
dt = 0.05  # [s] time difference
L = 2.9  # [m] Wheel base of vehicle
max_steer = np.radians(45.0)  # [rad] max steering angle

Kp = 1
Ki = 0.01
Kd = 0.5
PID = pid(Kp, Ki, Kd, 0.05, 5)

show_animation = True


class State(object):
    """
    Class representing the state of a vehicle.
    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        """Instantiate the object."""
        super(State, self).__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, acceleration, delta):
        """
        Update the state of the vehicle.
        Stanley Control uses bicycle model.
        :param acceleration: (float) Acceleration
        :param delta: (float) Steering
        """
        delta = np.clip(delta, -max_steer, max_steer)

        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / L * np.tan(delta) * dt
        self.yaw = normalize_angle(self.yaw)
        self.v += acceleration * dt

    def raw_update(self, state):
        # delta = np.clip(delta, -max_steer, max_steer)
        self.x = state[0]
        self.y = state[1]
        self.yaw = state[3]
        self.yaw = normalize_angle(self.yaw)
        self.v = state[2]


def pid_control(target, current):
    """
    Proportional control for the speed.
    :param target: (float)
    :param current: (float)
    :return: (float)
    """
    return PID.control(target - current)


def stanley_control(state, cx, cy, cyaw, last_target_idx):
    """
    Stanley steering control.
    :param state: (State object)
    :param cx: ([float])
    :param cy: ([float])
    :param cyaw: ([float])
    :param last_target_idx: (int)
    :return: (float, int)
    """
    current_target_idx, error_front_axle = calc_target_index(state, cx, cy)

    if last_target_idx >= current_target_idx:
        current_target_idx = last_target_idx

    # theta_e corrects the heading error
    theta_e = normalize_angle(cyaw[current_target_idx] - state.yaw)
    # theta_d corrects the cross track error
    theta_d = np.arctan2(k * error_front_axle, 0.01 + state.v)
    # Steering control
    delta = theta_e + theta_d
    delta = np.clip(delta, -max_steer, max_steer)

    return delta, current_target_idx


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def calc_target_index(state, cx, cy):
    """
    Compute index in the trajectory list of the target.
    :param state: (State object)
    :param cx: [float]
    :param cy: [float]
    :return: (int, float)
    """
    # Calc front axle position
    fx = state.x + L * np.cos(state.yaw)
    fy = state.y + L * np.sin(state.yaw)

    # Search nearest point index
    dx = [fx - icx for icx in cx]
    dy = [fy - icy for icy in cy]
    d = np.hypot(dx, dy)
    target_idx = np.argmin(d)

    # Project RMS error onto front axle vector
    front_axle_vec = [-np.cos(state.yaw + np.pi / 2),
                      -np.sin(state.yaw + np.pi / 2)]
    error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

    return target_idx, error_front_axle


class PyRoboticsStanley:
    def __init__(self, state):
        self.target_node_pub = Publisher("TargetNode", PoseStamped, queue_size=10)
        self.state_obj = State(state[0], state[1], state[3], state[2])
        self.dist_travelled = 0.0
        self.just_started = True

    def control(self, state, nodes):
        if len(nodes) > 0:
            self.dist_travelled += np.hypot(self.state_obj.x - state[0], self.state_obj.y - state[1])
        self.state_obj.raw_update(state)
        cx = [node[0] for node in nodes]
        cy = [node[1] for node in nodes]
        cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(cx, cy, ds=0.1)
        target_idx, _ = calc_target_index(self.state_obj, cx, cy)
        last_index = len(nodes) - 1

        if self.dist_travelled >= 20.0:
            self.just_started = False

        # Finished mapping
        if nodes[last_index][0] == nodes[0][0] and nodes[last_index][1] == nodes[0][1]:
            dist_to_end = np.hypot(cx[last_index] - self.state_obj.x, cy[last_index] - self.state_obj.y)

            if self.just_started is False and (dist_to_end <= 1.0):
                ai = pid_control(0.0, state[2])
            else:
                ai = pid_control(1.0, state[2])
        else:
            ai = pid_control(1.0, state[2])

        di, target_idx = stanley_control(self.state_obj, cx, cy, cyaw, target_idx)

        ps = PoseStamped()
        ps.pose.position.x = cx[target_idx]
        ps.pose.position.y = cy[target_idx]
        # ps.pose.position.x = state[0]
        # ps.pose.position.y = state[1]
        ps.header.frame_id = "map"
        ps.header.seq = 1
        ps.header.stamp = Time.now()
        self.target_node_pub.publish(ps)

        return ai, di


#
# def main():
#     import json
#     """Plot an example of Stanley steering control on a cubic spline."""
#     #  target course
#     # ax = [0.0, 100.0, 100.0, 50.0, 60.0]
#     # ay = [0.0, 0.0, -30.0, -20.0, 0.0]
#     with open("fsg_track.json", "r") as json_file:
#         map_dict = json.load(json_file)
#
#     ax = map_dict["X"][0:-2]
#     ay = map_dict["Y"][0:-2]
#
#     cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
#         ax, ay, ds=0.1)
#
#     target_speed = 15.0 / 3.6  # [m/s]
#
#     max_simulation_time = 150.0
#
#     # Initial state
#     state = State(x=-0.0, y=-3.0, yaw=np.radians(20.0), v=0.0)
#     # state = State(x=cx[0], y=cy[0], yaw=np.radians(20.0), v=0.0)
#
#     last_idx = len(cx) - 1
#     time = 0.0
#     x = [state.x]
#     y = [state.y]
#     yaw = [state.yaw]
#     v = [state.v]
#     t = [0.0]
#     target_idx, _ = calc_target_index(state, cx, cy)
#
#     dist_travelled = 0.0
#     just_started = True
#
#     while max_simulation_time >= time:
#         dist_to_end = np.hypot(cx[last_idx]-state.x, cy[last_idx]-state.y)
#
#         dist_travelled = np.hypot(state.x-cx[0], state.y-cy[0])
#         if (dist_travelled >= 20.0):
#             just_started = False
#
#         if just_started is False and (dist_to_end <= 1.0):
#             ai = pid_control(0.0, state.v)
#         else:
#             ai = pid_control(target_speed, state.v)
#         di, target_idx = stanley_control(state, cx, cy, cyaw, target_idx)
#         state.update(ai, di)
#
#         time += dt
#
#         x.append(state.x)
#         y.append(state.y)
#         yaw.append(state.yaw)
#         v.append(state.v)
#         t.append(time)
#
#         if show_animation:  # pragma: no cover
#             plt.cla()
#             plt.plot(map_dict["X_i"], map_dict["Y_i"], "y-")
#             plt.plot(map_dict["X_o"], map_dict["Y_o"], "y-")
#             # for stopping simulation with the esc key.
#             plt.gcf().canvas.mpl_connect('key_release_event',
#                     lambda event: [exit(0) if event.key == 'escape' else None])
#             plt.plot(cx, cy, "-r", label="course")
#             plt.plot(x, y, "-b", label="trajectory")
#             plt.plot(x[-1], y[-1], "xb", label="current position")
#             plt.plot(cx[target_idx], cy[target_idx], "xg", label="target")
#             plt.axis("equal")
#             plt.grid(True)
#             plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
#             plt.pause(0.001)
#
#     # Test
#     assert last_idx >= target_idx, "Cannot reach goal"
#
#     if show_animation:  # pragma: no cover
#         plt.plot(cx, cy, ".r", label="course")
#         plt.plot(x, y, "-b", label="trajectory")
#         plt.legend()
#         plt.xlabel("x[m]")
#         plt.ylabel("y[m]")
#         plt.axis("equal")
#         plt.grid(True)
#
#         plt.subplots(1)
#         plt.plot(t, [iv * 3.6 for iv in v], "-r")
#         plt.xlabel("Time[s]")
#         plt.ylabel("Speed[km/h]")
#         plt.grid(True)
#         plt.show()
#
#
# if __name__ == '__main__':
#     main()