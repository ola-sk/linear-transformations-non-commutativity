"""
This module provides functionality for generating a cube's edges as a set of
line segments in 3D space.

The cube is centered at the origin, and its size is adjustable through a
parameter that specifies the edge length. It leverages the `Vector3D`
and `LineSegment` classes for representing points and edges in 3D, and
uses rotation transformations to create and position the edges.
"""

import numpy as np
from linalg_3d import Vector3D
from linalg_3d import LineSegment
from linalg_3d.transformations import rotation_matrix

def fish(size: float = 1.0) -> list[LineSegment]:
    """
    Generates the edges of a cube centered at the origin.

    This function computes the edges of a cube by defining the front face edges,
    transforming them to generate the back face, and then calculating the side
    edges. The resulting edges are represented as instances of `LineSegment` objects.
    The cube is centered at the origin, and its size is determined by the `size`
    parameter, where each edge is of length `size` units.

    Parameters
    ----------
    size : float, optional
        The length of the edges of the cube. Defaults to 1.0.

    Returns
    -------
    list of LineSegment
        A list of `LineSegment` objects representing the edges of the cube. Each
        edge connects two vertices of the cube.
    """
    half = size / 2
    bottom_right = Vector3D([half, -half, half])
    top_right = bottom_right + Vector3D([0, size, 0])

    front_right_edge = LineSegment(
        bottom_right,
        top_right
    )
    front_face = [
        front_right_edge,
        front_right_edge.transform(rotation_matrix(np.pi / 2, 0, 1)),
        front_right_edge.transform(rotation_matrix(np.pi, 0, 1)),
        front_right_edge.transform(rotation_matrix(3 * np.pi / 2, 0, 1))
    ]
    side_face = [
        edge.transform(rotation_matrix(np.pi / 2, 1, 2)) for edge in front_face
    ]
    cube_edges = front_face + side_face
    return cube_edges