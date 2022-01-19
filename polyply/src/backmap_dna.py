# Copyrig 2020 University of Groningen
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

from polyply import jit
from polyply.src.processor import Processor
from polyply.src.linalg_functions import u_vect, center_of_geometry, \
    rotate_from_vect, finite_difference_O5, finite_difference_O1

"""
Processor implementing a template based back
mapping to lower resolution coordinates for
a DNA strand meta molecule.
"""

def calc_tangents(curve):
    """
    Compute the tangent vectors for a discrete 3d curve.
    If enough data points are available the tangents are calculatd
    using a fifth order finite difference, else the function reverts to
    a first order finite difference.

    Parameters
    ---------
    curve: numpy.ndarray
        A ndarray, of shape (N, 3), corresponding to the
        coordinates of a discrete 3d curve

    Returns
    ---------
    dX: numpy.ndarray
        A ndarray, of shape (N-1, 3), corresponding
        to the tangent vectors of X
    """
    if len(curve) > 4:
        curve_tangents = finite_difference_O5(curve)
    else:
        curve_tangents = finite_difference_O1(curve)
    return curve_tangents

def _gen_base_frame(base, template):
    """
    Given a 'base' type and a lower resolution 'template',
    construct the intrinsic reference needed for aligning
    the base.

    Parameters
    ---------
    base: str
       base type
    template: dict[collections.abc.Hashable, np.ndarray]
        dict of positions referring to the lower resolution atoms of node

    Returns
    ---------
    numpy.ndarray
        the base reference frame
    """

    anchor_library = {"DA": [("N1", "C4"), ("C2", "C6")],
                     "DG": [("N1", "C4"), ("C2", "C6")],
                     "DT": [("N3", "C6"), ("C2", "C4")],
                     "DC": [("N3", "C6"), ("C2", "C4")]}
    try:
        base_type = base[:2]
        anch1, anch2 = anchor_library[base_type]
    except :
        raise Exception('No reference frame available for nucleobase type.')

    ref_vec1 = template[anch1[0]] - template[anch1[1]]
    ref_vec2 = template[anch2[0]] - template[anch2[1]]

    binormal = u_vect(ref_vec1)
    tangent = u_vect(np.cross(ref_vec1, ref_vec2))
    normal = u_vect(np.cross(binormal, tangent))

    frame = np.stack((normal, binormal, tangent), axis=1)
    return frame

def orient_template(template, meta_frame, strand, base,
                    is_closed, strand_separation):
    """
    Align the nucleobase reference template with the local refernce frame
    defined on the meta_molecule

    Parameters:
    -----------
    template: dict[collections.abc.Hashable, np.ndarray]
        dict of positions referring to the lower resolution atoms of node
    meta_frame: numpy.ndarray
        local reference on the meta_molecule
    strand: str
        specify if base is component of forward or backward strand
    base: str
       base type
    is_closed: bool
    strand_separation: float
        strand separation in angstroms, measured between the reference
        origins of complementary nucleobases.


    Returns:
    --------
    dict
        The oriented base template
    """

    # Calculate intrinsic frame of the base
    template_frame = _gen_base_frame(base, template)

    # Determine rotation matrices
    inv_rot_template_frame = template_frame.T
    rot_meta_frame = meta_frame

    # Determine origin point of template
    positions = np.array([template["N1"], template["N3"], template["C2"],
                          template["C4"], template["C5"], template["C5"]])
    ref_origin = center_of_geometry(positions)

    # Write the template as array and center at origin
    template_arr = np.zeros((3, len(template)))
    key_to_ndx = {}
    for ndx, key in enumerate(template.keys()):
        template_arr[:, ndx] = template[key] - ref_origin
        key_to_ndx[key] = ndx

    # Rotate base frame into meta_molecule frame
    template_rotated_arr = rot_meta_frame @ (inv_rot_template_frame @ template_arr)

    # Final adjustments to rotated templates
    if strand == "backward":
        template_final_arr = (template_rotated_arr.T -
                              meta_frame[:, 1] * strand_separation).T
    else:
        template_rotated_arr = rotate_from_vect(template_rotated_arr,
                                                np.pi * meta_frame[:, 0])

        template_final_arr = (template_rotated_arr.T +
                             meta_frame[:, 1] * strand_separation).T


    # Write the template back as dictionary
    template_result = {}
    for key, ndx in key_to_ndx.items():
        template_result[key] = template_final_arr[:, ndx]

    return template_result


class Backmap_DNA(Processor):
    """
    This processor takes a a class:`polyply.src.MetaMolecule` and
    places coordinates form a higher resolution createing positions
    for the lower resolution molecule associated with the MetaMolecule.
    """

    def __init__(self, fudge_coords=1, is_closed=False, rotation_per_bp=0.59,
                 strand_separation=0.3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fudge_coords = fudge_coords
        self.rotation_per_bp = rotation_per_bp  # 34° in radian
        self.strand_separation = strand_separation
        self.is_closed = is_closed

    def _place_init_coords(self, meta_molecule):
        """
        For each residue in a class:`polyply.src.MetaMolecule` the
        positions of the atoms associated with that residue stored in
        attr:`polyply.src.MetaMolecule.molecule` are created from a
        template residue located in attr:`polyply.src.MetaMolecule.templates`.
        The orientation of the aforementioned templates is determined by
        reference frames constructed along the DNA curve using the
        double-reflection method by Wang et al. (DOI:10.1145/1330511.1330513)

        Parameters
        ----------
        meta_molecule: :class:`polyply.src.MetaMolecule`
        """

        # Read meta_molecule coordinates
        curve_coords = np.zeros((len(meta_molecule.nodes), 3))
        for node in meta_molecule.nodes:
            curve_coords[node] = meta_molecule.nodes[node]['position']

        # Calculate tangents
        tangents = calc_tangents(curve_coords)
        tangents = np.apply_along_axis(u_vect, axis=1, arr=tangents)

        # Initialize the reference normal vector
        normals = np.zeros_like(curve_coords)
        normals[0] = (tangents[0, 1], -tangents[0, 0], 0.)

        # Construct reference vectors along curve using double reflection
        # Ref: Wang et al. (DOI:10.1145/1330511.1330513)
        for i in range(len(tangents) - 1):
            vec1 = curve_coords[i+1] - curve_coords[i]
            norm1 = vec1 @ vec1
            R_l = normals[i] - (2/norm1) * (vec1 @ normals[i]) * vec1
            T_l = tangents[i] - (2/norm1) * (vec1 @ tangents[i]) * vec1
            vec2 = tangents[i+1] - T_l
            norm2 = vec2 @ vec2
            normals[i+1] = R_l - (2/norm2) * (vec2 @ R_l) * vec2
        normals = np.apply_along_axis(u_vect, axis=1, arr=normals)

        # Calculate the rotated binormals
        binormals = np.cross(tangents, normals)

        # Rotate frame into a darboux frame
        rotation_vectors = [vec * self.rotation_per_bp *
                            i for i, vec in enumerate(tangents, start=1)]
        for ndx, vector in enumerate(rotation_vectors):
            normals[ndx] = rotate_from_vect(normals[ndx], vector)
            binormals[ndx] = rotate_from_vect(binormals[ndx], vector)

        # Comply with boundary conditions if DNA is closed.
        # Uniformly distributing the corrective curviture,
        # minimizes the total squared angular speed.
        if self.is_closed:
            vec1 = curve_coords[0] - curve_coords[-1]
            norm1 = vec1 @ vec1
            R_l = binormals[-1] - (2/norm1) * (vec1 @ binormals[-1]) * vec1
            T_l = tangents[-1] - (2/norm1) * (vec1 @ tangents[-1]) * vec1
            vec2 = tangents[0] - T_l
            norm2 = vec2 @ vec2
            R_calc = R_l - (2/norm2) * (vec2 @ R_l) * vec2
            R_calc = u_vect(R_calc)
            phi = np.arccos(R_calc @ binormals[0])

            correction_per_base = phi / len(tangents)
            for ndx in range(1, len(tangents)):
                rotation_vector = correction_per_base * ndx * tangents[ndx]
                binormals[ndx] = rotate_from_vect(binormals[ndx], rotation_vector)
                binormals[ndx] = u_vect(binormals[ndx])
                normals[ndx] = np.cross(tangents[ndx], binormals[ndx])

        # Construct frame out of generated vectors
        meta_frames = [np.stack((i, j, k), axis=1)
                       for i, j, k in zip(normals, binormals, tangents)]

        for node in meta_molecule.nodes:
            if meta_molecule.nodes[node]["build"]:
                basepair = meta_molecule.nodes[node]["basepair"]
                cg_coord = meta_molecule.nodes[node]["position"]
                forward_base, backward_base = basepair.split(",")

                # Correctly orientate base on forward and backward strands
                forward_template = orient_template(meta_molecule.templates[forward_base],
                                                   meta_frames[node], "forward",
                                                   forward_base, self.is_closed,
                                                   self.strand_separation)
                backward_template = orient_template(meta_molecule.templates[backward_base],
                                                    meta_frames[node], "backward",
                                                    backward_base, self.is_closed,
                                                    self.strand_separation)

                # Place the molecule atoms according to the backmapping
                high_res_atoms = meta_molecule.nodes[node]["graph"].nodes
                for atom_high in high_res_atoms:
                    atomname = meta_molecule.molecule.nodes[atom_high]["atomname"]
                    base = meta_molecule.molecule.nodes[atom_high]["resname"]
                    if base == forward_base:
                        vector = forward_template[atomname]
                    else:
                        vector = backward_template[atomname]
                    new_coords = cg_coord + vector * self.fudge_coords
                    meta_molecule.molecule.nodes[atom_high]["position"] = new_coords

    def run_molecule(self, meta_molecule):
        """
        Apply placing of coordinates to meta_molecule. For more
        detail see `self._place_init_coords`.
        """
        self._place_init_coords(meta_molecule)
        return meta_molecule
