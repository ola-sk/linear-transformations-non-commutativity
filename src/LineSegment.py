import numpy as np
from Vector3D import Vector3D

class LineSegment:
    start: Vector3D
    end: Vector3D

    def __init__(self, start: Vector3D, end: Vector3D) -> None:
        if not isinstance(start, Vector3D):
            raise TypeError(f"start must be a Vector3D instance, got {type(start)}")
        if not isinstance(end, Vector3D):
            raise TypeError(f"end must be a Vector3D instance, got {type(end)}")
        if start.shape != (3,):
            raise ValueError(f"start must have shape (3,), got {start.shape}")
        if end.shape != (3,):
            raise ValueError(f"end must have shape (3,), got {end.shape}")
        self.start = start
        self.end = end

    def __str__(self) -> str:
        return f"LineSegment(start={np.asarray(self.start).tolist()}, end={np.asarray(self.end).tolist()})"

    def __repr__(self) -> str:
        return (
            f"LineSegment(Vector3D({np.asarray(self.start).tolist()}), "
            f"Vector3D({np.asarray(self.end).tolist()}))"
        )

    def transform(self, matrix: np.ndarray) -> 'LineSegment':
        """
        Transforms the current line segment by applying a transformation matrix. The transformation
        is performed using matrix multiplication for both the start and end points of the
        line segment, treating them as row vectors.

        :param matrix: A 2D NumPy array representing the transformation matrix to apply. The
            matrix should be compatible for transformation with the points of the line segment.
        :return: A new instance of the `LineSegment` class with the transformed start and end
            points.
        """
        # self.start and self.end are 1D np.ndarray (they are of type
        # Vector3D, which is a view of np.ndarray).
        # 1D ndarray used as right operand, e.g. matrix @ self.start is
        # treated as a column vector
        return LineSegment(Vector3D(matrix @ self.start), Vector3D(matrix @ self.end))

    def interpolate(self, num_points: int = 100) -> np.ndarray:
        return np.linspace(self.start, self.end, num_points)