import math
import numpy as np

def rotation_matrix(angle_rad: float, dim1: int, dim2: int) -> np.ndarray:
    """
    Returns a 3x3 rotation matrix for rotating in the plane defined by dim1 and dim2.

    :param angle_rad: Rotation angle in radians.
    :param dim1: Index (0-2) of one of the two dimensions involved in rotation.
    :param dim2: Index (0-2) of the other dimension involved in rotation.
    :return: 3x3 NumPy array representing the rotation matrix.
    :raises ValueError: If dim1 or dim2 are not in [0, 1, 2] or are equal.
    """
    if dim1 not in (0, 1, 2) or dim2 not in (0, 1, 2) or dim1 == dim2:
        raise ValueError("dim1 and dim2 must be different and in [0, 1, 2]")

    matrix = np.identity(3)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    matrix[dim1, dim1] = c
    matrix[dim1, dim2] = -s
    matrix[dim2, dim1] = s
    matrix[dim2, dim2] = c

    return matrix

