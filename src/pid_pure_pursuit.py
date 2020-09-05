import math
from src.sanitise_output import constrain_output, convert_acceleration_to_threshold


# Vehicle parameters
WB = 2.95  # Length from front wheel to back


# PID parameters
Kp = 1


# Pure Pursuit
Lfv = 0.1
Lfc = 2


def acceleration_control(target_speed, current_speed):
    # Simple P controller for now
    return Kp * (target_speed - current_speed)


def steering_control(x, y, yaw, target_x, target_y, Lf):
    rear_x, rear_y = calculate_rear_position(x, y, yaw)
    alpha = math.atan2(target_y - rear_y, target_x - rear_x) - yaw
    return math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0)


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


def search_target_index(x, y, v, yaw, nodes):
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
    @staticmethod
    def control(state, nodes):
        # Expected state to have the format: [x, y, v, yaw, ...]
        x, y, v, yaw = state[0], state[1], state[2], state[3]

        # Search for the node that is the furthest given our look ahead distance
        target_index, Lf = search_target_index(x, y, v, yaw, nodes)
        target_node = nodes[target_index]

        # Control law for forward acceleration. Target velocity is target_node[2],
        # where 2 describes the index of linear velocity in PathNode.msg
        acceleration = acceleration_control(target_node[2], v)

        # Control law for steering
        steering = steering_control(x, y, yaw, target_node[0], target_node[1], Lf)

        acceleration, steering = constrain_output(acceleration, steering)
        acc_threshold = convert_acceleration_to_threshold(acceleration)

        return acc_threshold, steering
