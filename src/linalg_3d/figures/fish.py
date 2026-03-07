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
from linalg_3d.transformations import rotation_matrix, reflection_about_plane

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
    # draw in the XY plane assuming Z=half; the front (head) of the
    # fish should be in Y-positive direction.
    mouth = Vector3D([0, size, size])
    topfront_head = mouth + Vector3D([size/2, -size/3, 0])
    topback_head = mouth + Vector3D([8*size/9, -size, 0])
    middle_top = Vector3D([size, -7*size/9, 8*size/9])
    tail_base_top = Vector3D([size/9, -size, -size/2])
    tail_fin_top = Vector3D([2*size/3, -size, -2*size/3])
    tail_fin2_top = Vector3D([7*size/9, -size, -size])
    tail_fin_center_top = Vector3D([size/3, -size, -size])
    tail_fin_center = Vector3D([0, -size, -2*size/3])

    fish_top = [
        LineSegment(mouth, topfront_head),
        LineSegment(topfront_head, topback_head),
        LineSegment(topback_head, middle_top),
        LineSegment(middle_top, tail_base_top),
        LineSegment(tail_base_top, tail_fin_top),
        LineSegment(tail_fin_top, tail_fin2_top),
        LineSegment(tail_fin2_top, tail_fin_center_top),
        LineSegment(tail_fin_center_top, tail_fin_center),
    ]
    fish_bottom = [edge.transform(reflection_about_plane(1, 2)) for edge in fish_top]
    eye_vector = Vector3D([1*size/8, size/3, size])
    eye = LineSegment(eye_vector, eye_vector)

    fish = fish_top + fish_bottom + [eye]
    return fish