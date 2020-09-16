import math
from sanitise_output import constrain_output, convert_acceleration_to_threshold
import rospy
from rospy import Publisher
from geometry_msgs.msg import PoseStamped
# from visualization_msgs.msg import MarkerArray, Marker
from pid import pid


# Vehicle parameters
WB = 2.951  # Length from front wheel to back


# PID parameters
Kp = 2
Ki = 0.1
Kd = 0.7
PID = pid(Kp, Ki, Kd, 1/20)


# Pure Pursuit
Lfv = 0.1
Lfc = 5


def acceleration_control(target_speed, current_speed):
    # # Simple P controller for now
    # return Kp * (target_speed - current_speed)
    return PID.control(target_speed - current_speed)


def steering_control(x, y, yaw, target_x, target_y, Lf):
    rear_x, rear_y = calculate_rear_position(x, y, yaw)
    alpha = math.atan2(target_y - rear_y, target_x - rear_x) - yaw
    steering = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0)
    # return steering if alpha > 0 else -steering
    return steering


def calculate_rear_position(x, y, yaw):
    # Convert the current position of the car in the centre vehicle frame,
    # to the rear vehicle frame
    rear_x = x - ((WB / 2.0) * math.cos(yaw))
    rear_y = y - ((WB / 2.0) * math.sin(yaw))
    return rear_x, rear_y


def calculate_rear_to_target(x, y, yaw, target_x, target_y):
    rear_x, rear_y = calculate_rear_position(x, y, yaw)
    # Calculate the difference from rear to target the Euclidean distance
    dx = rear_x - target_x
    dy = rear_y - target_y
    return math.hypot(dx, dy)


def search_target_index(x, y, v, yaw, nodes, previous_index):
    nodes = nodes[previous_index:]
    # Get look-ahead distance
    Lf = Lfv * v + Lfc
    # Calculate the distance from current car position to the first node
    index = 0
    point_x, point_y = nodes[index][0], nodes[index][1]
    index_distance = calculate_rear_to_target(x, y, yaw, point_x, point_y)
    # Iterate through each point in the planned path until we get the furthest point
    # according to Lf, or if we are at the end-point.
    while index_distance < Lf:
        if (index + 1) >= len(nodes):
            break
        # Since we are still in iteration, increment the index
        index += 1
        # Recalculate the distance to check if we have reached the furthest point
        point_x, point_y = nodes[index][0], nodes[index][1]
        index_distance = calculate_rear_to_target(x, y, yaw, point_x, point_y)

    return index, Lf


class PIDPurePursuit:
    def __init__(self):
        self.target_node_pub = Publisher("TargetNode", PoseStamped, queue_size=10)
        self.previous_index = 0

    def control(self, state, nodes):
        # Expected state to have the format: [x, y, v, yaw, ...]
        x, y, v, yaw = state[0], state[1], state[2], state[3]

        # Calculates previous index
        dx = [x - node[0] for node in nodes]
        dy = [y - node[1] for node in nodes]
        squared_distances = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        min_squared_dist = min(squared_distances)
        self.previous_index = squared_distances.index(min_squared_dist) + self.previous_index

        # Search for the node that is the furthest given our look ahead distance
        target_index, Lf = search_target_index(x, y, v, yaw, nodes, self.previous_index)
        target_node = nodes[target_index]

        ps = PoseStamped()
        ps.pose.position.x = target_node[0]
        ps.pose.position.y = target_node[1]
        ps.header.frame_id = "map"
        ps.header.seq = 1
        ps.header.stamp = rospy.Time.now()
        self.target_node_pub.publish(ps)

        # Control law for forward acceleration. Target velocity is target_node[2],
        # where 2 describes the index of linear velocity in PathNode.msg
        acceleration = acceleration_control(0.5, v)

        # Control law for steering
        steering = steering_control(
            x, y, yaw, target_node[0], target_node[1], Lf)

        acceleration, steering = constrain_output(acceleration, steering)
        acc_threshold = convert_acceleration_to_threshold(acceleration)

        return acc_threshold, steering
