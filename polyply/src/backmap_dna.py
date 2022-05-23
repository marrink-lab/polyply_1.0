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

def gen_template_frame(template, base, force_field):
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
    anchor_library = {'parmbsc1': {"DA": [[["N1"], ["C4"]], [["C2"], ["C6"]]],
                                   "DG": [[["N1"], ["C4"]], [["C2"], ["C6"]]],
                                   "DT": [[["N3"], ["C6"]], [["C2"], ["C4"]]],
                                   "DC": [[["N3"], ["C6"]], [["C2"], ["C4"]]]
                                   },
                      'martini2': {"DG": [[["SC3", "SC2"], ["SC1", "SC4"]],
                                          [["SC1"],["SC4"]]],
                                   "DA": [[["SC3", "SC2"], ["SC1", "SC4"]],
                                          [["SC1"],["SC4"]]],
                                   "DT": [[["SC3", "SC2"], ["SC1"]],
                                          [["SC2"],["SC3"]]],
                                   "DC": [[["SC3", "SC2"], ["SC1"]],
                                          [["SC2"],["SC3"]]],
                                   }
                    }
    try:
        base_type = base[:2]
        anch1, anch2 = anchor_library[force_field][base_type]
    except :
        raise Exception('No reference frame available for nucleobase type.')

    ref_vec1 = sum(template[key] for key in anch1[0]) / len(anch1[0])\
               - sum(template[key] for key in anch1[1]) / len(anch1[1])
    ref_vec2 = sum(template[key] for key in anch2[0]) / len(anch2[0])\
               - sum(template[key] for key in anch2[1]) / len(anch2[1])

    normal = u_vect(ref_vec2)
    binormal  = u_vect(ref_vec1)
    binormal -= np.dot(binormal, normal) * normal
    binormal  = u_vect(binormal)
    tangent = np.cross(normal, binormal)

    return np.stack((normal, binormal, tangent), axis=1)

def orient_template(template, target_frame, strand_dir,
                    base, force_field, strand_separation):
    """
    Align the base template frame with the local reference frame defined
    on the meta_molecule. The rotational alignment is performed by a
    linear transformation. Later the rotated base is translated to the
    correct position on the meta_molecule.

    For every force field, a predefined reference frame needs to be provided
    in order to perform the template orientation. This frame consists out of
    three orthonormal vectors: 1) along the strand tangent, 2) along the
    base-base vector and 3) pointing into the minor grove.

    Parameters:
    -----------
    template: dict[collections.abc.Hashable, np.ndarray]
        dict of positions referring to the high resolution (pseudo)atoms
    target_frame: numpy.ndarray
        local reference frame on meta_molecule
    strand_dir: str
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
    template_frame = gen_template_frame(template, base, force_field)

    # Determine rotation matrices
    inv_rot_template_frame = template_frame.T
    rot_target_frame = target_frame

    # Determine optimal reference point of template
    ref_origin_library = {'parmbsc1': ["N1", "N3", "C2", "C4", "C5", "C5"],
                          'martini2': ["SC1", "SC2", "SC3"]
                          }
    ref_origin = center_of_geometry([template[key] for key in ref_origin_library[force_field]])

    # Write the template as array and centre at origin
    template_arr = np.zeros((3, len(template)))
    key_to_ndx = {}
    for ndx, key in enumerate(template.keys()):
        template_arr[:, ndx] = template[key] - ref_origin
        key_to_ndx[key] = ndx

    # Rotate template frame into meta_molecule frame
    template_rotated_arr = rot_target_frame @ (inv_rot_template_frame @ template_arr)

    # Move rotated template frames to position on helix
    if strand_dir == "forward":
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

    def __init__(self, force_field="parmbsc1", rotation_per_bp=0.59,
                 strand_separation=0.3, fudge_coords=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fudge_coords = fudge_coords
        self.rotation_per_bp = rotation_per_bp  # 34° in radian
        self.strand_separation = strand_separation
        self.force_field = force_field

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
        strands = {"forward": meta_molecule.nodes,
                  "backward": reversed(list(meta_molecule.nodes))}
        for strand_dir in ["forward", "backward"]:
            for node in strands[strand_dir]:
                residue = meta_molecule.nodes[node]
                basepair, cg_coord = residue["resname"], residue["position"]

                if strand_dir == "forward":
                    nucleobase, resid = basepair.split(",")[0], node + 1
                else:
                    num_bps = len(meta_molecule.nodes)
                    nucleobase, resid = basepair.split(",")[1], 2 * num_bps - node

                # Correctly orientate base strand
                template = orient_template(meta_molecule.templates[nucleobase],
                                           moving_frame[node], strand_dir,
                                           nucleobase, self.force_field,
                                           self.strand_separation)

                # Write corresponding high resolution molecule to meta_molecule
                num_atoms = len(meta_molecule.molecule)
                for atom_ndx, atomname in enumerate(template):
                    vector = template[atomname]
                    new_coords = cg_coord + vector * self.fudge_coords
                    meta_molecule.molecule.add_nodes_from([(num_atoms + atom_ndx,
                                                            {"position": new_coords,
                                                             "atomname": atomname,
                                                             "resname": nucleobase,
                                                             "resid": resid})])

    def run_molecule(self, meta_molecule):
        """
        Apply placing of coordinates to meta_molecule. For more
        detail see `self._place_init_coords`.
        """
        self._place_init_coords(meta_molecule)
        return meta_molecule
