import numpy as np
from typing import cast


class Vector3D(np.ndarray):
    """
    Represents a vector (displacement/direction) in 3D space.

    Vectors represent a direction and magnitude, independent of position.
    They are origin-independent and represent displacements, not locations.

    Mathematical distinction:
    - Vector: Displacement/direction with magnitude (no fixed position)
    - Point: A location in space with specific coordinates

    Use Vector3D for:
    - Directions (e.g., velocity, force, normal vectors)
    - Displacements (e.g., translation amounts)
    - Mathematical operations that produce vectors

    For positions/locations, use Point3D instead.

    Performance characteristics:
    - Minimal memory overhead: Since there are no additional instance variables, a Vector3D has the same memory footprint as a np.ndarray.
    - No initialization penalty: The class adds minimal initialization logic via the constructor.
    - Vectorised operations: All NumPy optimized C-based operations still apply, so mathematical operations remain fast.
    - Type checking cost: The only cost comes at initialization time when the constructor validation logic is called, not during normal operations.

    Note: Like all NumPy array subclasses, validation only runs during direct construction.
    Operations like slicing, views, or array functions may bypass validation.

    Example:
        >>> v1 = Vector3D([1, 0, 0])  # Unit vector along x-axis
        >>> v2 = Vector3D([0, 1, 0])  # Unit vector along y-axis
        >>> v3 = v1 + v2              # Vector addition
        >>> magnitude = np.linalg.norm(v1)  # Vector magnitude
    """

    def __new__(
        cls,
        components: tuple[float | int, float | int, float | int] | list[float | int] | np.ndarray,
    ) -> 'Vector3D':
        """
        Constructor for Vector3D. Creates a vector from components.

        A vector represents a direction and magnitude, not a position in space.

        Parameters:
            components: Array-like object (tuple, list, or ndarray) with 3 numeric elements
                       representing the vector's components [x, y, z].

        Returns:
            New Vector3D array (numpy array of shape (3,) and dtype float).

        Raises:
            ValueError: If the input shape is not (3, ).
            TypeError: If input elements cannot be converted to floating point types.

        Example:
            >>> v1 = Vector3D([1, 2, 3])  # Vector with components (1, 2, 3)
            >>> v2 = Vector3D([3.5, -2.1, 0.0])
        """
        try:
            components = np.asarray(components, dtype=float)

            # Check shape
            if components.shape != (3,):
                raise ValueError(f"Vector3D components must have shape (3,), got {components.shape}")

            return cast('Vector3D', components.view(cls))
        except (ValueError, TypeError) as e:
            if "could not convert" in str(e):
                raise TypeError(
                    f"Vector3D requires numeric elements that can be converted to float, "
                    f"got {type(components)} with elements of incompatible types")
            raise

    def magnitude(self) -> float:
        """
        Calculate the magnitude (length) of the vector.

        Returns:
            The Euclidean norm of the vector.

        Example:
            >>> v = Vector3D([3, 4, 0])
            >>> v.magnitude()
            5.0
        """
        return float(np.linalg.norm(self))

    def normalize(self) -> 'Vector3D':
        """
        Return a unit vector in the same direction.

        Returns:
            A new Vector3D with magnitude 1 pointing in the same direction.

        Raises:
            ValueError: If the vector has zero magnitude.

        Example:
            >>> v = Vector3D([3, 4, 0])
            >>> unit_v = v.normalize()
            >>> unit_v.magnitude()
            1.0
        """
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize a zero vector")
        return Vector3D(self / mag)

    def __str__(self) -> str:
        """Return a string representation of the vector."""
        return f"Vector3D{(self.tolist())}"

    def __repr__(self) -> str:
        """Return a string representation that can be evaluated by
        eval() to reconstruct the object."""
        return f"Vector3D({self.tolist()})"

