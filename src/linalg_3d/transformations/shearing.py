"""
3D shearing transformations.

Analogy:
Imagine a tall building standing on the xz plane with its height along y.
When strong wind applies shear force, higher parts of the building shift more
than lower parts, while the vertical direction itself remains the reference
that drives the sideways displacement. In matrix terms, one dimension drives
proportional displacement in one or more other dimensions.
There is an assumption that the "walls" of the building can stretch
indefinitely as it shears, but the values in dimension of the height of
the building stay the same. In other words, if we project the height
of the sheared building onto an axis perpendicular to the ground on which
the building stands, it always stays the same: before and after the shearing.

This module provides a generic shearing matrix builder plus helpers for shear
parallel to each principal 3D plane (xy, xz, yz).
"""

import numpy as np


def _validate_dim(dim: int, name: str) -> None:
    if dim not in (0, 1, 2):
        raise ValueError(f"{name} must be in [0, 1, 2]")


def shearing_matrix(source_dim: int, target_dim: int, shear_factor: float) -> np.ndarray:
    """
    Generate a shearing matrix for 3D transformations.

    This function creates a 3x3 identity matrix and applies a shear factor between the specified
    `source_dim` and `target_dim`. The `source_dim` and `target_dim` must be different, and the
    shear factor will affect the corresponding element in the matrix for shear transformation
    in 3D space.

    Parameters
    ----------
    source_dim : int
        The source dimension to apply the shear transformation. Must be 0, 1, or 2.
    target_dim : int
        The target dimension that the source will be sheared relative to. Must be
        0, 1, or 2 and different from `source_dim`.
    shear_factor : float
        The factor of shearing applied between the source and target dimensions.

    Returns
    -------
    np.ndarray
        A 3x3 transformation matrix representing the shearing operation.

    Raises
    ------
    ValueError
        If `source_dim` and `target_dim` are the same.
    """
    _validate_dim(source_dim, "source_dim")
    _validate_dim(target_dim, "target_dim")
    if source_dim == target_dim:
        raise ValueError("source_dim and target_dim must be different")

    matrix = np.identity(3)
    matrix[target_dim, source_dim] = shear_factor
    return matrix


def shear_parallel_to_xy_plane(shear_factor_dim0: float = 0.0, shear_factor_dim1: float = 0.0) -> np.ndarray:
    """
    Generates a shear transformation matrix for a 3D space where the shear occurs
    parallel to the XY plane. The amount of shear is defined by the input factors
    along the X and Y directions.

    Parameters
    ----------
    shear_factor_dim0 : float, optional
        The shear factor along the X-axis in the XY plane. Defaults to 0.0.
    shear_factor_dim1 : float, optional
        The shear factor along the Y-axis in the XY plane. Defaults to 0.0.

    Returns
    -------
    numpy.ndarray
        A 3x3 matrix representing the shear transformation, where shear
        occurs parallel to the XY plane.
    """
    matrix = np.identity(3)
    matrix[0, 2] = shear_factor_dim0
    matrix[1, 2] = shear_factor_dim1
    return matrix


def shear_parallel_to_xz_plane(shear_factor_dim0: float = 0.0, shear_factor_dim2: float = 0.0) -> np.ndarray:
    """
    Generates a shear matrix for transformation parallel to the XZ-plane. The function
    computes a 3x3 transformation matrix that applies shear along the X and Z axes
    based on the specified shear factors.

    Parameters
    ----------
    shear_factor_dim0 : float, optional
        Shear factor along the X-axis. Determines the amount of shearing in the direction
        of the X-axis. Defaults to 0.0.

    shear_factor_dim2 : float, optional
        Shear factor along the Z-axis. Determines the amount of shearing in the direction
        of the Z-axis. Defaults to 0.0.

    Returns
    -------
    np.ndarray
        A 3x3 numpy array representing the shear transformation matrix for the XZ-plane.

    """
    matrix = np.identity(3)
    matrix[0, 1] = shear_factor_dim0
    matrix[2, 1] = shear_factor_dim2
    return matrix


def shear_parallel_to_yz_plane(shear_factor_dim1: float = 0.0, shear_factor_dim2: float = 0.0) -> np.ndarray:
    """
    Generates a 3x3 shear transformation matrix for shearing parallel to the YZ plane.

    The resulting matrix can be used to transform coordinates in three-dimensional space
    by applying a shear effect along the X-axis with respect to the Y and Z dimensions.

    Parameters
    ----------
    shear_factor_dim1 : float, optional
        Shearing factor for the Y dimension with respect to the X-axis.
    shear_factor_dim2 : float, optional
        Shearing factor for the Z dimension with respect to the X-axis.

    Returns
    -------
    numpy.ndarray
        A 3x3 shear transformation matrix for shearing in the YZ plane.
    """
    matrix = np.identity(3)
    matrix[1, 0] = shear_factor_dim1
    matrix[2, 0] = shear_factor_dim2
    return matrix

