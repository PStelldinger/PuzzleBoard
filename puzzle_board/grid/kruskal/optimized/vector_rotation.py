import numpy as np


def rotate_point(point, angle):
    """
    Rotate a 2D point around the origin (0, 0) by a given angle, using integer coordinates.

    :param point: A tuple (x, y) representing the 2D point with integer coordinates.
    :param angle: The rotation angle in degrees.
    :return: A tuple representing the rotated point with integer coordinates.
    """
    # Convert angle to radians
    angle_rad = np.radians(angle)

    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    # Rotate the point and round to the nearest integer
    rotated_point = np.dot(rotation_matrix, point)
    rotated_point_int = np.rint(rotated_point).astype(int)

    return rotated_point_int
