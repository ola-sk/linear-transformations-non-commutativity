import numpy as np


def scaling_matrix(scale_dim0: float = 1.0, scale_dim1: float = 1.0, scale_dim2: float = 1.0) -> np.ndarray:
    """
    Generate a scaling matrix for 3D transformations based on the provided scale factors for each dimension.

    The function creates a diagonal matrix used to scale 3D coordinates. Each scale factor
    corresponds to a specific axis (x, y, z) and determines the magnitude of scaling applied
    to that axis. If no values are provided, the default scale factors are set to 1.0, resulting
    in an identity matrix.

    Parameters
    ----------
    scale_dim0 : float, optional
        Scaling factor for the x-axis. Defaults to 1.0.
    scale_dim1 : float, optional
        Scaling factor for the y-axis. Defaults to 1.0.
    scale_dim2 : float, optional
        Scaling factor for the z-axis. Defaults to 1.0.

    Returns
    -------
    np.ndarray
        A 3x3 diagonal matrix representing the scaling transformation, with the scale
        factors on its diagonal.
    """
    return np.diag([scale_dim0, scale_dim1, scale_dim2])

