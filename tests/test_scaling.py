import unittest
import numpy as np

from linalg_3d.transformations.scaling import scaling_matrix

class TestScalingMatrix(unittest.TestCase):
    def test_default_scaling(self):
        matrix = scaling_matrix()
        expected = np.identity(3)
        np.testing.assert_array_equal(matrix, expected)

    def test_uniform_scaling(self):
        matrix = scaling_matrix(2.0, 2.0, 2.0)
        expected = np.array([
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2]
        ])
        np.testing.assert_array_equal(matrix, expected)

    def test_non_uniform_scaling(self):
        matrix = scaling_matrix(1.0, 2.0, 3.0)
        expected = np.array([
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]
        ])
        np.testing.assert_array_equal(matrix, expected)

    def test_zero_scaling(self):
        matrix = scaling_matrix(0, 0, 0)
        expected = np.zeros((3, 3))
        np.testing.assert_array_equal(matrix, expected)

    def test_negative_scaling(self):
        matrix = scaling_matrix(-1, -1, -1)
        expected = np.diag([-1.0, -1.0, -1.0])
        np.testing.assert_array_equal(matrix, expected)

    def test_determinant(self):
        matrix = scaling_matrix(2, 3, 4)
        self.assertAlmostEqual(np.linalg.det(matrix), 24.0)

    def test_identity_when_all_ones(self):
        matrix = scaling_matrix(1.0, 1.0, 1.0)
        np.testing.assert_array_equal(matrix, np.identity(3))

    def test_scaling_applied_to_vector(self):
        from linalg_3d.vector3d import Vector3D
        v = Vector3D([1, 2, 3])
        m = scaling_matrix(2, 0.5, 3)
        result = m @ v
        np.testing.assert_array_almost_equal(result, [2, 1, 9])

if __name__ == '__main__':
    unittest.main()
