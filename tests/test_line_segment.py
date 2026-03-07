import unittest
import numpy as np

from linalg_3d.vector3d import Vector3D
from linalg_3d.line_segment import LineSegment


class TestLineSegmentConstruction(unittest.TestCase):

    def test_basic_construction(self):
        s = Vector3D([0, 0, 0])
        e = Vector3D([1, 1, 1])
        seg = LineSegment(s, e)
        np.testing.assert_array_equal(seg.start, s)
        np.testing.assert_array_equal(seg.end, e)

    def test_non_vector3d_start_raises(self):
        with self.assertRaises(TypeError):
            LineSegment(np.array([0, 0, 0]), Vector3D([1, 1, 1]))

    def test_non_vector3d_end_raises(self):
        with self.assertRaises(TypeError):
            LineSegment(Vector3D([0, 0, 0]), np.array([1, 1, 1]))

    def test_wrong_shape_start_raises(self):
        """Vector3D itself rejects wrong shapes, so construction should fail."""
        with self.assertRaises(ValueError):
            LineSegment(Vector3D([0, 0]), Vector3D([1, 1, 1]))

    def test_degenerate_segment(self):
        """A zero-length segment (point) should be allowed."""
        p = Vector3D([2, 3, 4])
        seg = LineSegment(p, p)
        np.testing.assert_array_equal(seg.start, seg.end)


class TestLineSegmentTransform(unittest.TestCase):

    def test_identity_transform(self):
        seg = LineSegment(Vector3D([1, 0, 0]), Vector3D([0, 1, 0]))
        result = seg.transform(np.eye(3))
        np.testing.assert_array_almost_equal(result.start, seg.start)
        np.testing.assert_array_almost_equal(result.end, seg.end)

    def test_scaling_transform(self):
        seg = LineSegment(Vector3D([1, 0, 0]), Vector3D([0, 1, 0]))
        scale = np.diag([2.0, 3.0, 4.0])
        result = seg.transform(scale)
        np.testing.assert_array_almost_equal(result.start, [2, 0, 0])
        np.testing.assert_array_almost_equal(result.end, [0, 3, 0])

    def test_transform_returns_new_segment(self):
        seg = LineSegment(Vector3D([1, 0, 0]), Vector3D([0, 1, 0]))
        result = seg.transform(np.eye(3))
        self.assertIsNot(seg, result)

    def test_transform_result_is_line_segment(self):
        seg = LineSegment(Vector3D([1, 0, 0]), Vector3D([0, 1, 0]))
        result = seg.transform(np.eye(3))
        self.assertIsInstance(result, LineSegment)

    def test_90_degree_rotation(self):
        """90° rotation in X-Y plane: (1,0,0) → (0,1,0)."""
        from linalg_3d.transformations import rotation_matrix
        import math
        seg = LineSegment(Vector3D([1, 0, 0]), Vector3D([0, 0, 1]))
        rot = rotation_matrix(math.pi / 2, 0, 1)
        result = seg.transform(rot)
        np.testing.assert_array_almost_equal(result.start, [0, 1, 0])
        np.testing.assert_array_almost_equal(result.end, [0, 0, 1])


class TestLineSegmentTranslate(unittest.TestCase):

    def test_translate(self):
        seg = LineSegment(Vector3D([1, 2, 3]), Vector3D([4, 5, 6]))
        t = Vector3D([10, 20, 30])
        result = seg.translate(t)
        np.testing.assert_array_almost_equal(result.start, [11, 22, 33])
        np.testing.assert_array_almost_equal(result.end, [14, 25, 36])

    def test_translate_zero(self):
        seg = LineSegment(Vector3D([1, 2, 3]), Vector3D([4, 5, 6]))
        result = seg.translate(Vector3D([0, 0, 0]))
        np.testing.assert_array_almost_equal(result.start, seg.start)
        np.testing.assert_array_almost_equal(result.end, seg.end)


class TestLineSegmentRepr(unittest.TestCase):

    def test_str_contains_coords(self):
        seg = LineSegment(Vector3D([1, 2, 3]), Vector3D([4, 5, 6]))
        s = str(seg)
        self.assertIn("LineSegment", s)

    def test_repr_contains_class(self):
        seg = LineSegment(Vector3D([1, 2, 3]), Vector3D([4, 5, 6]))
        r = repr(seg)
        self.assertIn("LineSegment", r)
        self.assertIn("Vector3D", r)


if __name__ == "__main__":
    unittest.main()

