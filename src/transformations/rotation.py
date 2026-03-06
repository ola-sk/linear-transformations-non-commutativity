import math
import numpy as np

def rotation_matrix(angle_rad: float, dim0: int, dim1: int) -> np.ndarray:
    """
    Returns a 3x3 rotation matrix for rotating in the plane defined by dim0 and dim1.

    :param angle_rad: Rotation angle in radians.
    :param dim0: Index (0-2) of one of the two dimensions involved in rotation.
    :param dim1: Index (0-2) of the other dimension involved in rotation.
    :return: 3x3 NumPy array representing the rotation matrix.
    :raises ValueError: If dim1 or dim2 are not in [0, 1, 2] or are equal.
    """
    if dim0 not in (0, 1, 2) or dim1 not in (0, 1, 2) or dim0 == dim1:
        raise ValueError("dim1 and dim2 must be different and in [0, 1, 2]")

    matrix = np.identity(3)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    matrix[dim0, dim0] = c
    matrix[dim0, dim1] = -s
    matrix[dim1, dim0] = s
    matrix[dim1, dim1] = c

    return matrix

