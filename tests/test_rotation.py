import unittest
import numpy as np
import math

from linalg_3d.transformations.rotation import rotation_matrix

class TestRotationMatrix(unittest.TestCase):
    def test_rotation_z_axis(self):
        # Rotation in xy plane (dim0=0, dim1=1) by 90 degrees
        angle = math.pi / 2
        matrix = rotation_matrix(angle, 0, 1)
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_rotation_y_axis(self):
        # Rotation in xz plane (dim0=0, dim1=2) by 90 degrees
        angle = math.pi / 2
        matrix = rotation_matrix(angle, 0, 2)
        expected = np.array([
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, 0]
        ])
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_rotation_x_axis(self):
        # Rotation in yz plane (dim0=1, dim1=2) by 90 degrees
        angle = math.pi / 2
        matrix = rotation_matrix(angle, 1, 2)
        expected = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_rotation_zero_angle(self):
        matrix = rotation_matrix(0, 0, 1)
        expected = np.identity(3)
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            rotation_matrix(0, 0, 0) # same dim
        with self.assertRaises(ValueError):
            rotation_matrix(0, 0, 3) # dim out of range
        with self.assertRaises(ValueError):
            rotation_matrix(0, -1, 1) # dim out of range

    def test_full_360_returns_identity(self):
        matrix = rotation_matrix(2 * math.pi, 0, 1)
        np.testing.assert_array_almost_equal(matrix, np.identity(3))

    def test_inverse_rotation(self):
        """Rotating by θ then -θ should give identity."""
        angle = math.pi / 3
        m1 = rotation_matrix(angle, 1, 2)
        m2 = rotation_matrix(-angle, 1, 2)
        np.testing.assert_array_almost_equal(m1 @ m2, np.identity(3))

    def test_determinant_is_one(self):
        matrix = rotation_matrix(math.pi / 5, 0, 2)
        self.assertAlmostEqual(np.linalg.det(matrix), 1.0)

    def test_orthogonality(self):
        """R @ R^T should equal identity for any rotation."""
        matrix = rotation_matrix(1.23, 0, 1)
        np.testing.assert_array_almost_equal(matrix @ matrix.T, np.identity(3))

    def test_negative_angle(self):
        angle = -math.pi / 4
        matrix = rotation_matrix(angle, 0, 1)
        c, s = math.cos(angle), math.sin(angle)
        expected = np.identity(3)
        expected[0, 0] = c
        expected[0, 1] = -s
        expected[1, 0] = s
        expected[1, 1] = c
        np.testing.assert_array_almost_equal(matrix, expected)

if __name__ == '__main__':
    unittest.main()
