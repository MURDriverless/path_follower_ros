# Vehicle constraints

g = 9.81
max_acceleration = 1.2 * g
max_deceleration = -1.8 * g


def constrain_output(acceleration, steering):
    if acceleration >= max_acceleration:
        acceleration = max_acceleration
    elif acceleration <= max_deceleration:
        acceleration = max_deceleration

    return acceleration, steering


def convert_acceleration_to_threshold(acceleration):
    if acceleration > 0.0:
        return acceleration / max_acceleration
    else:
        return acceleration / max_deceleration
