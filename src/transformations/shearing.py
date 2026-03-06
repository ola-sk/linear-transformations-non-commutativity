import numpy as np

def shear_parallel_to_xz_plane(shear_factor_dim0: float = 0.0, shear_factor_dim2: float = 0.0) -> np.ndarray:
    """
    Returns a 3x3 shear matrix for 3D shearing transformation.

    As shear occurs in 3D we can define a plane parallel to which shear occurs.
    Let's illustrate it on 3 orthogonal axis: x, y and z, and a building:
    high buildings are prone to shear forces caused by wind.
    The building stands on the xz plane and has a certain height in y dimension.
    We assume that as the building shears, its height in y dimension remains the same.
    No matter how big the shear is, the building's walls are assumed
    to stretch indefinitely, not changing its "height" in the y-axis direction.

    Here the y dimension is the one affecting the shear in other
    dimensions, i.e. the plane xz.

    Shear factors define how much the values in y dimension
    affect the shear in other dimensions. As a 3D point (a, b, c) maps to:
    (a + shear_factor_dim0 * b, b, c + shear_factor_dim2 * b).
    We can see that b above is affecting each dimension:
    shear is directly proportional to the value of b.

    Shear factors are defined separately for each dimension in the plane.
    The "higher" we go in y-direction, the shear is more pronounced.
    Those factors are a measure of how much.

    :param shear_factor_dim0: Shear factor along the 0th dimension (0-based; i.e. x-axis).
    :param shear_factor_dim2: Shear factor along the 2nd dimension z-axis.
    :return: 3x3 NumPy array representing the shearing matrix.
    """
    return np.array([
        [1, shear_factor_dim0, 0],
        [0, 1, 0],
        [0, shear_factor_dim2, 1]
    ])

