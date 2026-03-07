import unittest
import numpy as np

from linalg_3d.transformations.shearing import (
    shearing_matrix,
    shear_parallel_to_xy_plane,
    shear_parallel_to_xz_plane,
    shear_parallel_to_yz_plane,
)

class TestShearingMatrix(unittest.TestCase):
    def test_shearing_matrix_generic(self):
        # source_dim=2 (z), target_dim=0 (x), shear_factor=2
        # v[0] = v[0] + 2 * v[2]
        matrix = shearing_matrix(2, 0, 2.0)
        expected = np.identity(3)
        expected[0, 2] = 2.0
        np.testing.assert_array_equal(matrix, expected)

    def test_shear_parallel_to_xy_plane(self):
        # z drives shear in x and y
        matrix = shear_parallel_to_xy_plane(1.0, 2.0)
        expected = np.identity(3)
        expected[0, 2] = 1.0
        expected[1, 2] = 2.0
        np.testing.assert_array_equal(matrix, expected)

    def test_shear_parallel_to_xz_plane(self):
        # y drives shear in x and z
        matrix = shear_parallel_to_xz_plane(1.0, 2.0)
        expected = np.identity(3)
        expected[0, 1] = 1.0
        expected[2, 1] = 2.0
        np.testing.assert_array_equal(matrix, expected)

    def test_shear_parallel_to_yz_plane(self):
        # x drives shear in y and z
        matrix = shear_parallel_to_yz_plane(1.0, 2.0)
        expected = np.identity(3)
        expected[1, 0] = 1.0
        expected[2, 0] = 2.0
        np.testing.assert_array_equal(matrix, expected)

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            shearing_matrix(0, 0, 1.0)
        with self.assertRaises(ValueError):
            shearing_matrix(0, 3, 1.0)
        with self.assertRaises(ValueError):
            shearing_matrix(-1, 1, 1.0)

    def test_zero_factor_is_identity(self):
        matrix = shearing_matrix(0, 1, 0.0)
        np.testing.assert_array_equal(matrix, np.identity(3))

    def test_determinant_is_one(self):
        matrix = shearing_matrix(1, 0, 5.0)
        self.assertAlmostEqual(np.linalg.det(matrix), 1.0)

    def test_shear_applied_to_vector(self):
        """Shearing source=2(z) -> target=0(x) with factor 3:
        [1, 2, 4] -> [1 + 3*4, 2, 4] = [13, 2, 4]."""
        from linalg_3d.vector3d import Vector3D
        v = Vector3D([1, 2, 4])
        m = shearing_matrix(2, 0, 3.0)
        result = m @ v
        np.testing.assert_array_almost_equal(result, [13, 2, 4])

    def test_inverse_shear(self):
        """Shearing by +k then -k should give identity."""
        m1 = shearing_matrix(1, 0, 2.5)
        m2 = shearing_matrix(1, 0, -2.5)
        np.testing.assert_array_almost_equal(m1 @ m2, np.identity(3))

if __name__ == '__main__':
    unittest.main()
