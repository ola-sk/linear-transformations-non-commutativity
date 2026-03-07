"""
This module provides functionality for generating a 3D star as a set of
line segments in 3D space.

The star is constructed as a spiked shape: six spike tips are placed along
the positive and negative X, Y, and Z axes, and each spike tip is connected
to the four nearest equatorial vertices of an inner octahedron.  The inner
octahedron edges are also included so the figure reads clearly as a solid
3D star rather than disconnected spikes.

The figure is centred at the origin and its size is adjustable.
"""

import numpy as np
from linalg_3d import Vector3D, LineSegment
from linalg_3d.transformations import rotation_matrix


def star(size: float = 1.0) -> list[LineSegment]:
    """
    Generate the edges of a six-pointed 3D star centred at the origin.

    The star has six spike tips along ±X, ±Y, ±Z at distance ``size``
    from the origin, and an inner octahedron whose vertices sit at
    ``size * 0.4`` from the origin.  Each spike tip is connected to its
    four neighbouring inner vertices, and all 12 inner-octahedron edges
    are included, giving a total of **36 edges**.

    Construction
    ------------
    1. Define the +Y spike tip, then generate all six tips by rotating
       the initial tip around the coordinate axes.
    2. Build the inner octahedron vertices at 40 % of the spike radius
       using the same rotation approach.
    3. For each spike tip, connect it to the four inner vertices that do
       **not** lie on the same axis (i.e. the four equatorial neighbours).
    4. Collect all 12 edges of the inner octahedron.

    Parameters
    ----------
    size : float, optional
        Distance from the origin to each spike tip.  Defaults to 1.0.

    Returns
    -------
    list[LineSegment]
        A list of ``LineSegment`` objects representing the star's edges.
    """
    inner_ratio = 0.4  # inner octahedron sits at 40 % of spike length

    # --- Build the six axis-aligned direction vectors ----------------------
    base = Vector3D([0, size, 0])

    tip_pos_y = base
    tip_neg_y = Vector3D(rotation_matrix(np.pi, 1, 2) @ base)
    tip_pos_x = Vector3D(rotation_matrix(-np.pi / 2, 0, 1) @ base)
    tip_neg_x = Vector3D(rotation_matrix(np.pi / 2, 0, 1) @ base)
    tip_pos_z = Vector3D(rotation_matrix(np.pi / 2, 1, 2) @ base)
    tip_neg_z = Vector3D(rotation_matrix(-np.pi / 2, 1, 2) @ base)

    tips = [tip_pos_x, tip_neg_x, tip_pos_y, tip_neg_y, tip_pos_z, tip_neg_z]

    # Inner octahedron vertices — same directions, shorter.
    inner = [Vector3D(np.asarray(t) * inner_ratio) for t in tips]
    # inner indices: 0=+X  1=-X  2=+Y  3=-Y  4=+Z  5=-Z

    # --- Spike edges -------------------------------------------------------
    # Each tip connects to the 4 inner vertices NOT on its own axis.
    # Axis groups (pairs sharing an axis): (0,1)=X  (2,3)=Y  (4,5)=Z
    axis_pairs = [{0, 1}, {2, 3}, {4, 5}]

    edges: list[LineSegment] = []
    for tip_idx, tip in enumerate(tips):
        own_pair = next(p for p in axis_pairs if tip_idx in p)
        neighbours = [j for j in range(6) if j not in own_pair]
        for n in neighbours:
            edges.append(LineSegment(tip, inner[n]))

    # --- Inner octahedron edges --------------------------------------------
    for i in range(6):
        for j in range(i + 1, 6):
            own_pair = next(p for p in axis_pairs if i in p)
            if j not in own_pair:
                edges.append(LineSegment(inner[i], inner[j]))

    return edges

