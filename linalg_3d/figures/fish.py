"""
This module provides functionality for generating a stylised fish as a set
of line segments in 3D space.

The top half of the fish is defined manually from mouth to tail, and the
bottom half is produced by reflecting those segments across the X-axis.
An eye point is included as a degenerate (zero-length) line segment.
"""

from linalg_3d import Vector3D
from linalg_3d import LineSegment
from linalg_3d.transformations import reflection_about_plane


def fish(size: float = 1.0) -> list[LineSegment]:
    """
    Generate the edges of a stylised fish centred near the origin.

    Parameters
    ----------
    size : float, optional
        Overall scale factor for the fish.  Defaults to 1.0.

    Returns
    -------
    list[LineSegment]
        Line segments forming the fish outline (top half, bottom half,
        and an eye).
    """
    # Draw in the XY plane assuming Z=half; the front (head) of the
    # fish faces the Y-positive direction.
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