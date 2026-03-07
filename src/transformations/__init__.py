# This file marks the 'transformations' directory as a Python package.

# The transformations package contains modular (atomic) transformation matrices for linear algebra operations.
from .rotation import rotation_matrix
from .reflection import reflection_about_axis_matrix
from .scaling import scaling_matrix
from .translation import translation_vector
from .shearing import (
    shearing_matrix,
    shear_parallel_to_xy_plane,
    shear_parallel_to_xz_plane,
    shear_parallel_to_yz_plane,
)
