import numpy as np

from Vector3D import Vector3D
from LineSegment import LineSegment

if __name__ == "__main__":
    v1 = Vector3D([1, 2, 3])
    v2 = Vector3D([4, 5, 6])
    line = LineSegment(v1, v2)
    print(line)

    #  a simple stretching by 2 in a single direction X(0th)
    transformation_matrix = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
    print(line.transform(transformation_matrix))

