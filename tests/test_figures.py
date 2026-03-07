"""Tests for figure generators (cube, fish, star)."""

import unittest
import numpy as np

from linalg_3d.line_segment import LineSegment
from linalg_3d.vector3d import Vector3D
from linalg_3d.figures.cube import cube
from linalg_3d.figures.fish import fish
from linalg_3d.figures.star import star


class TestCubeFigure(unittest.TestCase):

    def test_returns_list_of_line_segments(self):
        edges = cube(1.0)
        self.assertIsInstance(edges, list)
        for e in edges:
            self.assertIsInstance(e, LineSegment)

    def test_cube_has_12_edges(self):
        edges = cube(1.0)
        self.assertEqual(len(edges), 12)

    def test_cube_centered_at_origin(self):
        """All vertex coordinates should be ±half."""
        edges = cube(2.0)
        coords = set()
        for e in edges:
            for pt in (e.start, e.end):
                coords.update(pt.tolist())
        # With size=2, half=1, so all coords are ±1
        for c in coords:
            self.assertAlmostEqual(abs(c), 1.0, places=10)

    def test_custom_size(self):
        edges = cube(4.0)
        coords = set()
        for e in edges:
            for pt in (e.start, e.end):
                coords.update(pt.tolist())
        for c in coords:
            self.assertAlmostEqual(abs(c), 2.0, places=10)

    def test_each_edge_has_correct_length(self):
        size = 3.0
        edges = cube(size)
        for e in edges:
            length = np.linalg.norm(np.asarray(e.end) - np.asarray(e.start))
            self.assertAlmostEqual(length, size, places=10)


class TestFishFigure(unittest.TestCase):

    def test_returns_list_of_line_segments(self):
        edges = fish(1.0)
        self.assertIsInstance(edges, list)
        for e in edges:
            self.assertIsInstance(e, LineSegment)

    def test_fish_has_segments(self):
        """Fish should have top + bottom + eye = 8 + 8 + 1 = 17 segments."""
        edges = fish(1.0)
        self.assertEqual(len(edges), 17)

    def test_fish_scales_with_size(self):
        small = fish(1.0)
        big = fish(2.0)
        # The mouth vertex y-coordinate scales linearly with size
        self.assertAlmostEqual(small[0].start[1] * 2, big[0].start[1])


class TestStarFigure(unittest.TestCase):

    def test_returns_list_of_line_segments(self):
        edges = star(1.0)
        self.assertIsInstance(edges, list)
        for e in edges:
            self.assertIsInstance(e, LineSegment)

    def test_star_has_36_edges(self):
        """6 tips × 4 neighbours = 24 spike edges + 12 inner edges = 36."""
        edges = star(1.0)
        self.assertEqual(len(edges), 36)

    def test_spike_tips_at_correct_distance(self):
        size = 2.0
        edges = star(size)
        # Collect all unique vertices
        verts = set()
        for e in edges:
            verts.add(tuple(np.round(e.start, 10)))
            verts.add(tuple(np.round(e.end, 10)))
        distances = sorted({round(np.linalg.norm(v), 10) for v in verts})
        # Two distinct distances: inner (0.4 * size) and outer (size)
        self.assertEqual(len(distances), 2)
        self.assertAlmostEqual(distances[0], 0.4 * size, places=8)
        self.assertAlmostEqual(distances[1], size, places=8)

    def test_star_symmetric_about_origin(self):
        """For every vertex v, -v should also be a vertex."""
        edges = star(1.5)
        verts = set()
        for e in edges:
            verts.add(tuple(np.round(e.start, 10)))
            verts.add(tuple(np.round(e.end, 10)))
        for v in list(verts):
            neg = tuple(-np.array(v))
            neg_rounded = tuple(np.round(neg, 10))
            self.assertIn(neg_rounded, verts,
                          f"Vertex {v} has no symmetric counterpart {neg_rounded}")


if __name__ == "__main__":
    unittest.main()

