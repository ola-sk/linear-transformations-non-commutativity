import unittest
import numpy as np

from linalg_3d.transformations.translation import translation_vector
from linalg_3d.vector3d import Vector3D

class TestTranslationVector(unittest.TestCase):
    def test_default_translation(self):
        v = translation_vector()
        self.assertIsInstance(v, Vector3D)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(v, expected)

    def test_custom_translation(self):
        v = translation_vector(1.0, -2.0, 3.5)
        self.assertIsInstance(v, Vector3D)
        expected = np.array([1.0, -2.0, 3.5])
        np.testing.assert_array_equal(v, expected)

    def test_zero_translation(self):
        v = translation_vector(0.0, 0.0, 0.0)
        np.testing.assert_array_equal(v, [0, 0, 0])

    def test_negative_translation(self):
        v = translation_vector(-5.0, -10.0, -15.0)
        np.testing.assert_array_equal(v, [-5.0, -10.0, -15.0])

    def test_translation_applied_to_segment(self):
        from linalg_3d.line_segment import LineSegment
        seg = LineSegment(Vector3D([1, 2, 3]), Vector3D([4, 5, 6]))
        t = translation_vector(10, 20, 30)
        result = seg.translate(t)
        np.testing.assert_array_almost_equal(result.start, [11, 22, 33])
        np.testing.assert_array_almost_equal(result.end, [14, 25, 36])

    def test_opposite_translations_cancel(self):
        t1 = translation_vector(3.0, -4.0, 5.0)
        t2 = translation_vector(-3.0, 4.0, -5.0)
        np.testing.assert_array_almost_equal(t1 + t2, [0, 0, 0])

if __name__ == '__main__':
    unittest.main()
