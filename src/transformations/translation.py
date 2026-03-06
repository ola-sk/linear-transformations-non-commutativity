import numpy as np

from ..Vector3D import Vector3D


def translation_vector(dim0: float = 0.0, dim1: float = 0.0, dim2: float = 0.0) -> np.ndarray:
    """
    Returns a 1D numpy ndarray of shape (3,) cast to Vector3D class.

    Usage: add this vector to a 3D point/vector to translate it.

    :param dim0: Translation along the 0th (x) axis (default 0.0).
    :param dim1: Translation along the 1st (y) axis (default 0.0).
    :param dim2: Translation along the 2nd (z) axis (default 0.0).
    :return: A 3-element Vector3D object describing translation in 3D space.
    """
    return Vector3D([dim0, dim1, dim2])

