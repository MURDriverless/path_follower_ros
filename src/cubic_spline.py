import numpy as np
import bisect


class Spline:
    def __init__(self, t, x):
        self.t = t
        self.x = x

        # For n points, we have n-1 segments
        number_of_segments = len(x) - 1
        tf_vector = np.diff(t)

        self.spline_coefficients = [self.get_spline_coefficients(x[i], 0, x[i + 1], 0, tf_vector[i])
                                    for i in range(number_of_segments)]

    def search_spline_index(self, t):
        # bisect() does not merely find the nearest index of the element 't'. It finds
        # the first index to the where we have to insert the element 't' to keep it sorted
        # (if it sounds weird, check bisect's documentation).
        # That means, if I have t = [0.0, 5.0, 10.0], bisect(4.5) will return 1, not 0.
        # So, to get the index of the starting spline, we need to subtract 1 from index.
        index = bisect.bisect(self.t, t) - 1
        # Additionally, if we are interpolating at the end index, there are no splines
        # after the end index. So at the end index, set index -= 1 to use the spline just
        # one index before the end index. Otherwise, just return index
        return index - 1 if index == (len(self.t)-1) else index

    def interpolate(self, t):
        """
        Given input 't', calculate the interpolated x if it is within the interpolation range
        Args:
            t (float): Current input to interpolate result
        Returns:
            float: Value of x which is interpolated at time 't' of the input argument
        """
        # Check if t is within the interpolation range
        if t < self.t[0]:
            # return None
            t = self.t[0]
        elif t > self.t[-1]:
            # return None
            t = self.t[-1]

        index = self.search_spline_index(t)
        coefficients = self.spline_coefficients[index]
        t = t - self.t[index]
        t2 = t ** 2
        t3 = t2 * t
        return coefficients[0] + coefficients[1] * t + coefficients[2] * t2 + coefficients[3] * t3

    def interpolate_first_derivative(self, t):
        """Similar to self.interpolate(), but calculates first derivative of the cubic spline region"""
        if t < self.t[0]:
            # print(f"{t} is below minimum range")
            # return None
            t = self.t[0]
        elif t > self.t[-1]:
            # print(f"{t} is above maximum range")
            # return None
            t = self.t[-1]

        index = self.search_spline_index(t)
        coefficients = self.spline_coefficients[index]
        t = t - self.t[index]
        t2 = t ** 2
        return coefficients[1] + 2*coefficients[2]*t + 3*coefficients[3]*t2

    @staticmethod
    def get_spline_coefficients(x_init, xdot_init, x_final, xdot_final, tf):
        """
        Calculate the spline coefficients a0, a1, a2 and a3
        Args:
            x_init (float): f(0), initial f(t) value at the start of the spline
            xdot_init (float): f'(0), initial f'(t) value at the start of the spline
            x_final (float): f(tf), final f(t) value at the end of the spline
            xdot_final (float): f'(tf), final f'(t) value at the end of the spline
            tf (float): The 't' value for x_final
        Returns:
            np.ndarray: [a0, a1, a2, a3]
        """
        tf_2 = tf ** 2
        tf_3 = tf_2 * tf

        # We will get our segment coefficient using the equation AX = B
        # 'A' is a 4x4 matrix on the left-most side containing values in terms of a0, a1, a2 and a3
        # 'X' is a 4x1 matrix containing our segment coefficients [a0; a1; a2; a3;]
        # 'C' is a 4x1 matrix on the right-hand side containing [x(0); x'(0); x(tf); x'(tf)]
        A = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, tf, tf_2, tf_3],
            [0, 1, 2*tf, 3*tf_2]
        ])
        B = np.array([x_init, xdot_init, x_final, xdot_final])

        # Solve for X
        return np.linalg.solve(A, B)


class Spline2D:
    def __init__(self, x, y):
        self.t = self.get_t_vector(x, y)
        self.sx = Spline(self.t, x)
        self.sy = Spline(self.t, y)

    @staticmethod
    def get_t_vector(x, y):
        # Assuming x and y are n points, dx & dy capture the difference between successive points
        dx = np.diff(x)
        dy = np.diff(y)
        magnitude = np.hypot(dx, dy)
        # The time vector is expressed as the projection of dx and dy onto a single axis using hypotenuse,
        # and we use cumulative sum to build up a time vector.
        cumulative_sum = np.cumsum(magnitude)
        # Add the value '0' to index 0
        return np.insert(cumulative_sum, 0, 0.0000)

    def interpolate(self, t):
        x = self.sx.interpolate(t)
        y = self.sy.interpolate(t)
        return x, y

    def interpolate_first_derivative(self, t):
        xd = self.sx.interpolate_first_derivative(t)
        yd = self.sy.interpolate_first_derivative(t)
        return xd, yd
