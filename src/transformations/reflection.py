import numpy as np


def reflection_matrix(axis: int) -> np.ndarray:
    """
    Returns a 3x3 reflection matrix across the specified axis: 0, 1 or 2 index.

    In 2D reflections we set up the identity matrix and reverse every
    dimension other than the one around which we reflect. This generalises to 3D.

    Reflection is a 180-degree rotation in a plane perpendicular to the axis of reflection.

    :param axis: Axis to reflect across 0-indexed.
    :return: 3x3 NumPy array representing the reflection matrix.
    :raises ValueError: If the axis is not 0, 1 or 2.
    """
    if axis == 0:
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    elif axis == 1:
        return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    elif axis == 2:
        return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    else:
        raise ValueError("axis must be '0, 1 or 2.")
