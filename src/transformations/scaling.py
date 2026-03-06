import numpy as np

def scaling_matrix(scale_dim0: float = 1.0, scale_dim1: float = 1.0, scale_dim2: float = 1.0) -> np.ndarray:
    """
    Returns a 3x3 scaling matrix.

    :param scale_dim0: Scaling factor along the 0-axis  (default 1.0).
    :param scale_dim1: Scaling factor along the 1-axis  (default 1.0).
    :param scale_dim2: Scaling factor along the 2-axis (default 1.0).
    :return: 3x3 NumPy array representing the scaling matrix.
    """
    return np.diag([scale_dim0, scale_dim1, scale_dim2])

