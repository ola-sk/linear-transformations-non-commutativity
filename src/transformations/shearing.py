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
    Returns a 3x3 shear matrix where one source dimension shears one target dimension.

    For a vector v, the transformed component is:
    v[target_dim] = v[target_dim] + shear_factor * v[source_dim].


    :param source_dim: Source dimension index (0-based) driving the shear.
    :param target_dim: Target dimension index (0-based) being sheared.
    :param shear_factor: Shear factor from source_dim into target_dim.
    :return: 3x3 NumPy array representing the shearing matrix.
    :raises ValueError: If dimensions are outside [0, 1, 2] or equal.
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
    Returns a 3x3 shear matrix parallel to the xy plane (z drives shear in x and y).

    Mapping: (x, y, z) -> (x + shear_factor_dim0 * z, y + shear_factor_dim1 * z, z).

    :param shear_factor_dim0: Shear factor applied to dimension 0 from dimension 2.
    :param shear_factor_dim1: Shear factor applied to dimension 1 from dimension 2.
    :return: 3x3 NumPy array representing the shearing matrix.
    """
    matrix = np.identity(3)
    matrix[0, 2] = shear_factor_dim0
    matrix[1, 2] = shear_factor_dim1
    return matrix


def shear_parallel_to_xz_plane(shear_factor_dim0: float = 0.0, shear_factor_dim2: float = 0.0) -> np.ndarray:
    """
    Returns a 3x3 shear matrix parallel to the xz plane (y drives shear in x and z).

    Mapping: (x, y, z) -> (x + shear_factor_dim0 * y, y, z + shear_factor_dim2 * y).

    :param shear_factor_dim0: Shear factor applied to dimension 0 from dimension 1.
    :param shear_factor_dim2: Shear factor applied to dimension 2 from dimension 1.
    :return: 3x3 NumPy array representing the shearing matrix.
    """
    matrix = np.identity(3)
    matrix[0, 1] = shear_factor_dim0
    matrix[2, 1] = shear_factor_dim2
    return matrix


def shear_parallel_to_yz_plane(shear_factor_dim1: float = 0.0, shear_factor_dim2: float = 0.0) -> np.ndarray:
    """
    Returns a 3x3 shear matrix parallel to the yz plane (x drives shear in y and z).

    Mapping: (x, y, z) -> (x, y + shear_factor_dim1 * x, z + shear_factor_dim2 * x).

    :param shear_factor_dim1: Shear factor applied to dimension 1 from dimension 0.
    :param shear_factor_dim2: Shear factor applied to dimension 2 from dimension 0.
    :return: 3x3 NumPy array representing the shearing matrix.
    """
    matrix = np.identity(3)
    matrix[1, 0] = shear_factor_dim1
    matrix[2, 0] = shear_factor_dim2
    return matrix
