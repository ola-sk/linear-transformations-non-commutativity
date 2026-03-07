import numpy as np


def reflection_about_axis_matrix(axis: int) -> np.ndarray:
    """
    Generates a reflection matrix about the specified axis.

    This function returns a 3x3 reflection matrix for a specified axis in 3D space.
    Reflection is performed on a plane orthogonal to the specified axis,
    which results in negating the coordinates of points along the remaining two axes.

    Parameters
    ----------
    axis : int
        The axis about which reflection is performed. Must be one of the following:
        0 (reflection about the yz-plane), 1 (reflection about the xz-plane), or
        2 (reflection about the xy-plane).

    Returns
    -------
    np.ndarray
        A 3x3 numpy array representing the reflection matrix.

    Raises
    ------
    ValueError
        If the specified axis is not 0, 1, or 2.
    """
    if axis == 0:
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    if axis == 1:
        return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    if axis == 2:
        return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    raise ValueError("axis must be 0, 1 or 2.")


def reflection_about_plane(dim0: int, dim1: int) -> np.ndarray:
    if dim0 == 0 and dim1 == 1:
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    if dim0 == 0 and dim1 == 2:
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    if dim0 == 1 and dim1 == 2:
        return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    raise ValueError("dim0 and dim1 must be different and in [0, 1, 2]")

