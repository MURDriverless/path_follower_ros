import numpy as np

# Vehicle constraints

g = 9.81
max_acceleration = 1.2 * g
max_deceleration = -1.8 * g

max_steering = 0.8


def constrain_output(acceleration, steering):
    if acceleration >= max_acceleration:
        acceleration = max_acceleration
    elif acceleration <= max_deceleration:
        acceleration = max_deceleration
    if np.abs(steering) >= max_steering:
        steering = max_steering - 0.001

    return acceleration, steering


def convert_acceleration_to_threshold(acceleration):
    if acceleration > 0.0:
        return acceleration / max_acceleration
    else:
        return acceleration / max_deceleration
