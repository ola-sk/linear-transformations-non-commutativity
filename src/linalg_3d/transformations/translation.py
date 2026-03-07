import numpy as np

from linalg_3d import Vector3D


def translation_vector(dim0: float = 0.0, dim1: float = 0.0, dim2: float = 0.0) -> Vector3D:
    """
    Generates a translation vector in 3D space.

    This function creates a 3-dimensional translation vector based on the
    specified x, y, and z values (called dim0, dim1, dim2 respectively).
     The resulting vector is represented as a 1D NumPy array of shape (3,)
     (view as Vector3D) in the 3D coordinate system.

    Usage: add this vector to a 3D point/vector to translate it.

    Parameters
    ----------
    dim0 : float, optional
        The x-coordinate of the translation vector. Defaults to 0.0.
    dim1 : float, optional
        The y-coordinate of the translation vector. Defaults to 0.0.
    dim2 : float, optional
        The z-coordinate of the translation vector. Defaults to 0.0.

    Returns
    -------
    numpy.ndarray
        A 1D NumPy array representing the translation vector in 3D
        space, where the elements correspond to the x, y, and z
        components respectively.
    """
    return Vector3D([dim0, dim1, dim2])

