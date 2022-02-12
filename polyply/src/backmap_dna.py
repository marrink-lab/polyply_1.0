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

from .processor import Processor
from .gen_moving_frame import ConstructFrame
from .linalg_functions import u_vect, center_of_geometry, rotate_from_vect

"""
Processor implementing a template based backmapping of DNA strands
meta_molecules to higher resolution (pseudo)atom coordinates.
"""

def gen_template_frame(template, base):
    """
    Given a high resolution 'template' and 'base' type,
    construct the intrinsic base template frame.

    Parameters
    ---------
    base: str
       base type
    template: dict[collections.abc.Hashable, np.ndarray]
        dict of positions referring to the lower resolution atoms of node

    Returns
    ---------
    numpy.ndarray
        base template frame
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

def orient_template(template, target_frame, strand, base, strand_separation):
    """
    Align the base template frame with the local reference frame defined
    on the meta_molecule. The rotational alignment is performed by a
    linear transformation. Later the rotated base is translated to the
    correct position on the meta_molecule.

    For every forcefield, a predefined reference frame needs to be provided
    in order to perform the template orientation. This frame consists out of
    three orthonormal vectors: 1) along the strand tangent, 2) along the
    base-base vector and 3) pointing into the minor grove.

    Parameters:
    -----------
    template: dict[collections.abc.Hashable, np.ndarray]
        dict of positions referring to the high resolution (pseudo)atoms
    target_frame: numpy.ndarray
        local reference frame on meta_molecule
    strand: str
        specify if base on forward or backward strand
    base: str
       base type
    strand_separation: float
        strand separation in angstroms, measured between the reference
        origins of complementary bases.

    Returns:
    --------
    dict
        oriented base template
    """
    # Calculate intrinsic frame of the base
    template_frame = gen_template_frame(template, base)

    # Determine rotation matrices
    inv_rot_template_frame = template_frame.T
    rot_target_frame = target_frame

    # Determine optimal reference point of template
    positions = np.array([template["N1"], template["N3"], template["C2"],
                          template["C4"], template["C5"], template["C5"]])
    ref_origin = center_of_geometry(positions)

    # Write the template as array and centre at origin
    template_arr = np.zeros((3, len(template)))
    key_to_ndx = {}
    for ndx, key in enumerate(template.keys()):
        template_arr[:, ndx] = template[key] - ref_origin
        key_to_ndx[key] = ndx

    # Rotate template frame into meta_molecule frame
    template_rotated_arr = rot_target_frame @ (inv_rot_template_frame @ template_arr)

    # Move rotated template frames to position on helix
    if strand == "backward":
        template_final_arr = (template_rotated_arr.T -
                              target_frame[:, 1] * strand_separation).T
    else:
        template_rotated_arr = rotate_from_vect(template_rotated_arr,
                                                np.pi * target_frame[:, 0])
        template_final_arr = (template_rotated_arr.T +
                             target_frame[:, 1] * strand_separation).T

    # Write the template back as dictionary
    template_result = {}
    for key, ndx in key_to_ndx.items():
        template_result[key] = template_final_arr[:, ndx]

    return template_result

class Backmap_DNA(Processor):
    """
    This processor takes a class:`polyply.src.MetaMolecule` and
    places coordinates from higher resolution templates on positions
    from the lower resolution molecule associated with the MetaMolecule.
    """

    def __init__(self, fudge_coords=1, rotation_per_bp=0.59,
                 strand_separation=0.3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fudge_coords = fudge_coords
        self.rotation_per_bp = rotation_per_bp  # 34° in radian
        self.strand_separation = strand_separation

    def _place_init_coords(self, meta_molecule):
        """
        For each residue in a class:`polyply.src.MetaMolecule` the
        positions of the atoms associated with that residue stored in
        attr:`polyply.src.MetaMolecule.molecule` are created from a
        template residue located in attr:`polyply.src.MetaMolecule.templates`.
        The orientation of the aforementioned templates are determined by
        a moving frame constructed along the DNA curve.

        Parameters
        ----------
        meta_molecule: :class:`polyply.src.MetaMolecule`
        """
        # Generate moving frames on meta_molecule
        construct_frame = ConstructFrame(rotation_per_bp=self.rotation_per_bp)
        moving_frame = construct_frame.run(meta_molecule)

        # Place base templates on reference frames
        for ndx, node in enumerate(meta_molecule.nodes):
            residue = meta_molecule.nodes[node]
            if residue["build"] and residue["restype"] == "DNA":
                basepair = residue["resname"]
                cg_coord = residue["position"]
                forward_base, backward_base = basepair.split(",")

                # Correctly orientate base strand
                forward_template = orient_template(meta_molecule.templates[forward_base],
                                                   moving_frame[ndx], "forward",
                                                   forward_base,
                                                   self.strand_separation)
                backward_template = orient_template(meta_molecule.templates[backward_base],
                                                    moving_frame[ndx], "backward",
                                                    backward_base,
                                                    self.strand_separation)

                # Place the molecule atoms according to the backmapping
                high_res_atoms = residue["graph"].nodes
                for atom_ndx in high_res_atoms:
                    atomname = residue["graph"].nodes[atom_ndx]["atomname"]
                    strand = residue["graph"].nodes[atom_ndx]["strand"]
                    if strand == "forward":
                        vector = forward_template[atomname]
                    else:
                        vector = backward_template[atomname]
                    new_coords = cg_coord + vector * self.fudge_coords
                    meta_molecule.molecule.nodes[atom_ndx]["position"] = new_coords

    def run_molecule(self, meta_molecule):
        """
        Apply placing of coordinates to meta_molecule. For more
        detail see `self._place_init_coords`.
        """
        self._place_init_coords(meta_molecule)
        return meta_molecule
