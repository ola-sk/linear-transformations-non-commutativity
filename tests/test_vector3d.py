import unittest
import math
import numpy as np

from linalg_3d.vector3d import Vector3D


class TestVector3DConstruction(unittest.TestCase):
    """Construction, validation and basic properties."""

    def test_from_list(self):
        v = Vector3D([1, 2, 3])
        np.testing.assert_array_equal(v, [1.0, 2.0, 3.0])

    def test_from_tuple(self):
        v = Vector3D((4, 5, 6))
        np.testing.assert_array_equal(v, [4.0, 5.0, 6.0])

    def test_from_ndarray(self):
        v = Vector3D(np.array([7.0, 8.0, 9.0]))
        np.testing.assert_array_equal(v, [7.0, 8.0, 9.0])

    def test_dtype_is_float(self):
        v = Vector3D([1, 2, 3])
        self.assertEqual(v.dtype, float)

    def test_shape_is_3(self):
        v = Vector3D([1, 2, 3])
        self.assertEqual(v.shape, (3,))

    def test_is_ndarray_subclass(self):
        v = Vector3D([1, 2, 3])
        self.assertIsInstance(v, np.ndarray)

    def test_wrong_size_raises(self):
        with self.assertRaises(ValueError):
            Vector3D([1, 2])
        with self.assertRaises(ValueError):
            Vector3D([1, 2, 3, 4])

    def test_non_numeric_raises(self):
        with self.assertRaises((TypeError, ValueError)):
            Vector3D(["a", "b", "c"])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            Vector3D([])


class TestVector3DArithmetic(unittest.TestCase):
    """Standard vector arithmetic via NumPy ops."""

    def test_addition(self):
        v = Vector3D([1, 2, 3]) + Vector3D([4, 5, 6])
        np.testing.assert_array_equal(v, [5, 7, 9])

    def test_subtraction(self):
        v = Vector3D([4, 5, 6]) - Vector3D([1, 2, 3])
        np.testing.assert_array_equal(v, [3, 3, 3])

    def test_scalar_multiplication(self):
        v = Vector3D([1, 2, 3]) * 2
        np.testing.assert_array_equal(v, [2, 4, 6])

    def test_negation(self):
        v = -Vector3D([1, -2, 3])
        np.testing.assert_array_equal(v, [-1, 2, -3])

    def test_dot_product(self):
        result = np.dot(Vector3D([1, 0, 0]), Vector3D([0, 1, 0]))
        self.assertAlmostEqual(result, 0.0)

    def test_cross_product(self):
        result = np.cross(Vector3D([1, 0, 0]), Vector3D([0, 1, 0]))
        np.testing.assert_array_almost_equal(result, [0, 0, 1])


class TestVector3DMagnitude(unittest.TestCase):

    def test_unit_vector(self):
        v = Vector3D([1, 0, 0])
        self.assertAlmostEqual(v.magnitude(), 1.0)

    def test_345_triangle(self):
        v = Vector3D([3, 4, 0])
        self.assertAlmostEqual(v.magnitude(), 5.0)

    def test_zero_vector_magnitude(self):
        v = Vector3D([0, 0, 0])
        self.assertAlmostEqual(v.magnitude(), 0.0)


class TestVector3DNormalize(unittest.TestCase):

    def test_normalize_unit(self):
        v = Vector3D([3, 4, 0]).normalize()
        self.assertAlmostEqual(v.magnitude(), 1.0)
        np.testing.assert_array_almost_equal(v, [0.6, 0.8, 0.0])

    def test_normalize_preserves_direction(self):
        v = Vector3D([0, 0, 5])
        n = v.normalize()
        np.testing.assert_array_almost_equal(n, [0, 0, 1])

    def test_normalize_zero_raises(self):
        with self.assertRaises(ValueError):
            Vector3D([0, 0, 0]).normalize()


class TestVector3DRepr(unittest.TestCase):

    def test_str(self):
        s = str(Vector3D([1, 2, 3]))
        self.assertIn("Vector3D", s)

    def test_repr_roundtrip(self):
        v = Vector3D([1.5, -2.5, 3.0])
        reconstructed = eval(repr(v))
        np.testing.assert_array_almost_equal(v, reconstructed)


if __name__ == "__main__":
    unittest.main()

