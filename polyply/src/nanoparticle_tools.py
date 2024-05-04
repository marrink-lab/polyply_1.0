import numpy as np
from typing import Literal

def create_np_pattern(
    pattern: Literal[None, "Striped", "Janus", "TigerStriped", "Fibonacci"],
    core_numpy_coords: np.ndarray,
    length: int,
    minimum_threshold: float,
    maximum_threshold: float,
):
    """
    Output pattern for decorating the nanoparticle core.

    At the moment, we have striped, janus, and tigerstriped
    patterns to add onto the core surface
    """
    # identify only the surface atoms
    if pattern == None:  # when we have no patterns on surfaces
        core_values = {}
        for index, entry in enumerate(core_numpy_coords):
            core_values[index] = entry
        core_indices = [core_values]

    elif pattern == "Striped":  # when we have a striped patterns
        core_striped_values = {}
        core_ceiling_values = {}
        threshold = length / 3  # divide nanoparticle region into 3
        for index, entry in enumerate(core_numpy_coords):
            if (
                entry[2] > minimum_threshold + threshold
                and entry[2] < maximum_threshold - threshold
            ):
                core_striped_values[index] = entry
            else:
                core_ceiling_values[index] = entry
        core_indices = [core_striped_values, core_ceiling_values]

    elif pattern == "Janus":  # when we have a janus patterns
        core_top_values = {}
        core_bot_values = {}
        threshold = length / 2  # divide nanoparticle region into 2
        for index, entry in enumerate(core_numpy_coords):
            if entry[2] > minimum_threshold + threshold:
                core_top_values[index] = entry
            else:
                core_bot_values[index] = entry
        core_indices = [core_top_values, core_bot_values]

    # Need to test what this pattern comes up with
    elif pattern == "TigerStriped":
        num_points = len(core_numpy_coords)
        num_stripes = 10  # Example, you can adjust as needed
        phi = np.linspace(0, np.pi, num_points)
        theta = np.linspace(0, 2 * np.pi, num_points)
        phi, theta = np.meshgrid(phi, theta)

        stripe_pattern = np.sin(num_stripes * phi)

        x = np.sin(phi) * np.cos(theta) * stripe_pattern
        y = np.sin(phi) * np.sin(theta) * stripe_pattern
        z = np.cos(phi) * stripe_pattern

        striped_points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
        core_indices = [striped_points]

    elif pattern == "Fibonacci":
        num_points = len(core_numpy_coords)
        phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle
        indices = np.arange(num_points)
        z = 1 - (indices / float(num_points - 1)) * 2
        radius = np.sqrt(1 - z**2)
        theta = phi * indices
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        fibonacci_points = np.column_stack((x, y, z))
        core_indices = [fibonacci_points]
        
    return core_indices

def reorganize_points(points, R=1.5, random=True):
    """
    remove points at random and rearrange on the surface of the nanoparticle
    or alternatively, we can remove ligands that are R cartesian distance away from
    each other
    """
    if random == True:
        random_indices = np.random.choice(
            len(points), size=int(len(points) / 2), replace=False
        )
         
        # Remove the selected points
        remaining_points = np.delete(points, random_indices, axis=0)
        return remaining_points
    
    # Create a distance matrix
    dist_matrix = np.linalg.norm(points[:, None] - points, axis=-1)
    # Set the diagonal elements to a large value to avoid selecting the same point
    np.fill_diagonal(dist_matrix, np.inf)
    # Find indices where distance is less than or equal to R
    indices = np.where(dist_matrix >= R)
    # Get unique indices and filter points
    filtered_points = np.unique(np.concatenate((indices[0], indices[1])))
    return points[filtered_points]

def rotation_matrix_from_vectors(
    vec_a: np.ndarray, vec_b: np.ndarray, direction: str = "ccw"
) -> np.ndarray:
    """
    Find the rotation matrix that aligns vec_a to vec_b
    Args:
    vec_a:
        A 3D "source" vector
    vec_b:
        A 3D "destination" vector
    direction:
        'cw' (clockwise) or 'ccw' (counterclockwise) to specify the rotation direction
    Returns:
    rotation_matrix:
        A transformation matrix (3x3) which when applied to vec_a, aligns it with vec_b.
    """
    a, b = (vec_a / np.linalg.norm(vec_a)).reshape(3), (
        vec_b / np.linalg.norm(vec_b)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    if direction == "cw":
        rotation_matrix = np.eye(3) - kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    elif direction == "ccw":
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    else:
        raise ValueError("Invalid rotation direction. Use 'cw' or 'ccw'.")

    return rotation_matrix

