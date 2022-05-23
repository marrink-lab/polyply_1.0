# Copyright 2020 University of Groningen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import networkx as nx

from .linalg_functions import u_vect, rotate_from_vect, \
                        finite_difference_O5, finite_difference_O1

"""
Linear algebra class which constructs a rotating and moving
frame along a linear meta_molecule.
"""

def calc_tangents(curve):
    """
    Compute the tangent vectors for a discrete 'curve'. If enough
    data points are available the tangents are calculated using
    a fifth order finite difference, else the function reverts to
    a first order finite difference.

    Parameters
    ---------
    curve: numpy.ndarray
        A ndarray, of shape (N, 3), corresponding to the
        coordinates of a discrete curve

    Returns
    ---------
    dX: numpy.ndarray
        A ndarray, of shape (N, 3), corresponding
        to the tangent vectors of X
    """
    if len(curve) > 4:
        curve_tangents = finite_difference_O5(curve)
    else:
        curve_tangents = finite_difference_O1(curve)
    return curve_tangents

def gen_next_normal(current_frame, next_frame):
    """
    Generate a normal vector that determines the next frame and
    that is rotation minimizing with respect to the current orthonormal
    frame. Here we use the double reflection method by Wang et al.
    (DOI:10.1145/1330511.1330513)

    Parameters
    ---------
    current_frame: tuple
        containing information of the current orthonormal frame,
        i.e. the coordinate(r1), tangent(t1) and normal(n1) vector.
    next_frame: tuple
        containing information of the next frame that we want
        to define, i.e. the coordinate(r2) and tangent(t2) vector.

    Returns
    ---------
    normal: np.ndarray
        vector determining next_frame
    """
    r1, t1, n1 = current_frame
    r2, t2 = next_frame

    vec1 = r2 - r1
    norm1 = vec1 @ vec1
    Rl = n1 - (2/norm1) * (vec1 @ n1) * vec1
    Tl = t1 - (2/norm1) * (vec1 @ t1) * vec1
    vec2 = t2 - Tl
    norm2 = vec2 @ vec2
    n2 = Rl - (2/norm2) * (vec2 @ Rl) * vec2
    normal = u_vect(n2)
    return normal

def rotation_min_frame(curve):
    """
    Construct a rotation minimizing frame along
    a discrete 'curve' using the double reflection method.

    Parameters
    ---------
    curve: numpy.ndarray
        discrete curve coordinates, ndarray of shape (N, 3)

    Returns
    ---------
    tangent, normals, binormals: tuple
        All three ndarray, of shape (N, 3), corresponding
        to the reference vectors along curve
    """
    # Calculate tangents
    tangents = calc_tangents(curve)
    tangents = np.apply_along_axis(u_vect, axis=1, arr=tangents)

    # Initialize the reference/normal vector
    normals = np.zeros_like(curve)
    if np.any(tangents[0, :2]):
        normals[0] = [tangents[0, 1], -tangents[0, 0], 0.0]
    elif np.any(tangents[0, 2:]):
        normals[0] = [0.0, tangents[0, 2], -tangents[0, 1]]
    else:
            msg = ('\n Not able to create DNA strands'
                   'Check the provided DNA coordinates')
            raise IOError(msg)
    normals[0] = u_vect(normals[0])

    # Construct reference vectors along curve using double reflection
    for i in range(len(tangents) - 1):
        normals[i+1] = gen_next_normal((curve[i], tangents[i], normals[i]),
                                       (curve[i+1], tangents[i+1]))

    # Calculate the rotated binormals
    binormals = np.cross(tangents, normals)
    return tangents, normals, binormals

def close_frame(curve, tangents, normals, binormals, rotation_per_bp):
    """
    Uniformly distribute the corrective twist needed to close
    the moving frame on a cyclic curve. The Uniform distribution
    of excess curvatures, minimizes the total squared angular
    speed of the moving frame.

    Parameters
    ---------
    curve: numpy.ndarray
        ndarray of shape (N, 3)
    tangents: numpy.ndarray
        ndarray of shape (N, 3)
    normals: numpy.ndarray
        ndarray of shape (N, 3)
    binormals: numpy.ndarray
        ndarray of shape (N, 3)

    Returns
    ---------
    tangent, normals, coords: tuple
        All three ndarray, of shape (N, 3), corresponding
        to the reference vectors along cyclic curve
    """
    # Determine the minimal rotating normal vector wrt. last reference frame
    target_normal = gen_next_normal((curve[-1], tangents[-1], normals[-1]),
                                    (curve[0], tangents[0]))
    # Calculate rotation needed to match the normals
    phi = np.arccos(target_normal @ normals[0]) - rotation_per_bp

    # Distribute corrective curvature over curve
    correction_per_base = phi / (len(tangents) - 1)
    for ndx in range(1, len(tangents)):
        rotation_vector = correction_per_base * ndx * tangents[ndx]
        normals[ndx] = rotate_from_vect(normals[ndx], rotation_vector)
        binormals[ndx] = np.cross(tangents[ndx], normals[ndx])
    return tangents, normals, binormals

def dna_frame(meta_molecule, curve, rotation_per_bp):
    # Generate rotation minimizing frame on curve
    tangents, normals, binormals = rotation_min_frame(curve)

    # Rotate moving frame to introduce intrinsic DNA twist
    rotation_vectors = [vec * rotation_per_bp *
                        i for i, vec in enumerate(tangents, start=1)]
    for ndx, vector in enumerate(rotation_vectors):
        normals[ndx] = rotate_from_vect(normals[ndx], vector)
        binormals[ndx] = rotate_from_vect(binormals[ndx], vector)

    # Comply with boundary conditions if DNA is closed
    if nx.cycle_basis(meta_molecule):
        tangents, normals, binormals = close_frame(curve, tangents,
                                                   normals, binormals,
                                                   rotation_per_bp)
    return tangents, normals, binormals


class ConstructFrame:
    """
    Generate a rotating and moving frame over a meta_molecule.
    """

    def __init__(self, rotation_per_bp=0.59):
        self.rotation_per_bp = rotation_per_bp  # 34° in radian

    def run(self, meta_molecule):
        """
        Construct the moving frame. If the user provides base orientations,
        these are used. Otherwise, first a rotation minimizing frame is
        created along the 'meta_molecule' coordinates. An intrinsic twist
        is added to the moving frame to incorporate the DNA double-helix.
        Lastly, if the 'meta_molecule' is cyclical, a corrective twist
        is added in order to close the moving frame.

        Parameters
        ----------
        meta_molecule: :class:`polyply.src.MetaMolecule`

        Returns
        ----------
        moving_frame: numpy.ndarray
            ndarray of shape (N, (3, 3))

        """
        # Read positions from meta_molecule
        position_dictionary = nx.get_node_attributes(meta_molecule, "position")
        curve = np.array(list(position_dictionary.values()))

        if not nx.get_node_attributes(meta_molecule, "normal"):
            tangents, normals, binormals = dna_frame(meta_molecule, curve,
                                                     self.rotation_per_bp)
        else:
            # Read normals from meta_molecule
            normals = nx.get_node_attributes(meta_molecule, "normal")
            normals = np.asarray(list(normals.values()))
            normals = np.apply_along_axis(u_vect, axis=1, arr=normals)

            if not nx.get_node_attributes(meta_molecule, "tangent"):
                tangents = calc_tangents(curve)
                # Apply Gram–Schmidt orthogonalization
                tangents -= np.dot(normals, tangents) * normals
                tangents = np.apply_along_axis(u_vect, axis=1, arr=tangents)
            else:
                # Read tangents from meta_molecule
                tangents = nx.get_node_attributes(meta_molecule, "tangent")
                tangents = np.asarray(list(tangents.values()))
                tangents = np.apply_along_axis(u_vect, axis=1, arr=tangents)

            if not nx.get_node_attributes(meta_molecule, "binormals"):
                binormals = np.cross(tangents, normals)
            else:
                # Read binormals from meta_molecule
                binormals = nx.get_node_attributes(meta_molecule, "binormals")
                binormals = np.asarray(list(binormals.values()))
                binormals = np.apply_along_axis(u_vect, axis=1, arr=binormals)

        # Construct frame out of generated vectors
        moving_frame = [np.stack((i, j, k), axis=1)
                       for i, j, k in zip(normals, binormals, tangents)]
        return moving_frame
