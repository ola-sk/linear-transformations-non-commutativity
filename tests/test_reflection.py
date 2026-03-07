import unittest
import numpy as np

from linalg_3d.transformations.reflection import reflection_about_axis_matrix

class TestReflectionMatrix(unittest.TestCase):
    def test_reflection_axis_0(self):
        # Reflection across axis 0 (x-axis)
        # Expected: [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
        matrix = reflection_about_axis_matrix(0)
        expected = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        np.testing.assert_array_equal(matrix, expected)

    def test_reflection_axis_1(self):
        # Reflection across axis 1 (y-axis)
        # Expected: [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
        matrix = reflection_about_axis_matrix(1)
        expected = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
        np.testing.assert_array_equal(matrix, expected)

    def test_reflection_axis_2(self):
        # Reflection across axis 2 (z-axis)
        # Expected: [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
        matrix = reflection_about_axis_matrix(2)
        expected = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_equal(matrix, expected)

    def test_invalid_axis(self):
        with self.assertRaises(ValueError):
            reflection_about_axis_matrix(3)
        with self.assertRaises(ValueError):
            reflection_about_axis_matrix(-1)

    def test_double_reflection_is_identity(self):
        """Reflecting twice about the same axis should yield identity."""
        for axis in (0, 1, 2):
            m = reflection_about_axis_matrix(axis)
            np.testing.assert_array_almost_equal(m @ m, np.identity(3))

    def test_determinant_is_positive_one(self):
        """Reflecting about an axis negates two components, so det = +1."""
        for axis in (0, 1, 2):
            m = reflection_about_axis_matrix(axis)
            self.assertAlmostEqual(np.linalg.det(m), 1.0)

    def test_preserves_reflected_axis_component(self):
        """The component along the reflection axis should be unchanged."""
        from linalg_3d.vector3d import Vector3D
        v = Vector3D([3, 4, 5])
        for axis in (0, 1, 2):
            m = reflection_about_axis_matrix(axis)
            result = m @ v
            self.assertAlmostEqual(result[axis], v[axis])

if __name__ == '__main__':
    unittest.main()
