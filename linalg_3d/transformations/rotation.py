from math import cos, sin

import numpy as np

from linalg_3d import Vector3D


def rotation_matrix(
    angle_rad: float,
    dim0: int,
    dim1: int,
    origin: Vector3D = Vector3D([0, 0, 0]),
) -> np.ndarray:
    """
    Generates a 3D rotation matrix for rotating around the plane defined by two
    specified axes in 3-dimensional space. The rotation is performed by a given
    angle in radians and optionally around a specific origin.

    Parameters
    ----------
    angle_rad : float
        The angle of rotation in radians.
    dim0 : int
        The first axis defining the plane of rotation. Must be 0, 1, or 2.
    dim1 : int
        The second axis defining the plane of rotation. Must be 0, 1, or 2, and
        must be different from `dim0`.
    origin : Vector3D, optional
        The origin around which rotation is performed, default is a zero vector
        [0, 0, 0].

    Returns
    -------
    ndarray
        A 3x3 rotation matrix representing the rotation in the specified plane
        and angle.

    Raises
    ------
    ValueError
        If `dim0` or `dim1` is not in [0, 1, 2], or if they are the same value.
    """
    if dim0 not in (0, 1, 2) or dim1 not in (0, 1, 2) or dim0 == dim1:
        raise ValueError("dim1 and dim2 must be different and in [0, 1, 2]")

    matrix = np.identity(3)
    c, s = cos(angle_rad), sin(angle_rad)
    matrix[dim0, dim0] = c
    matrix[dim0, dim1] = -s
    matrix[dim1, dim0] = s
    matrix[dim1, dim1] = c
    return matrix

