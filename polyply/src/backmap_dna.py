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
from scipy.spatial.transform import Rotation

from polyply.src.processor import Processor
from polyply.src.linalg_functions import u_vect, center_of_geometry
from polyply import jit

"""
Processor implementing a template based back
mapping to lower resolution coordinates for
"""


def _calc_tangents(X):
    dX = np.zeros_like(X)
    # If enough points use fifth order tangent approx. else use second order
    if len(X) > 4:
        # Calculate tangent boundary points
        dX[0] = -25*X[0] + 48*X[1] - 36*X[2] + 16*X[3] - 3*X[4]
        dX[1] = -3*X[0] - 10*X[1] + 18*X[2] - 6*X[3] + X[4]
        dX[-2] = 3*X[-1] + 10*X[-2] - 18*X[-3] + 6*X[-4] - X[-5]
        dX[-1] = 25*X[-1] - 48*X[-2] + 36*X[-3] - 16*X[-4] + 3*X[-5]

        # Calculate tangent interior points
        for i, _ in enumerate(X[2:-2], 2):
            dX[i] = X[i-2] - 8 * X[i-1] + 8 * X[i+1] - X[i+2]
    else:
        # Calculate tangent boundary points
        dX[0] = X[1] - X[0]
        dX[-1] = X[-1] - X[-2]

        # Calculate tangent interior points
        for i, _ in enumerate(X[1:-1], 1):
            dX[i] = X[i-1] - X[i+1]
    return dX

# this is the numba implementation
calc_tangents = jit(_calc_tangents)

def _gen_base_frame(base, template):

    if base[:2] == "DA" or base[:2] == "DG":
        vec1 = template["N1"] - template["C4"]
        vec2 = template["C2"] - template["C6"]

        e2 = u_vect(vec1)
        e3 = u_vect(np.cross(vec1, vec2))
        e1 = u_vect(np.cross(e3, e2))

    else: # base == "DT" or base == "DC":
        vec1 = template["N3"] - template["C6"]
        vec2 = template["C2"] - template["C4"]

        e2 = u_vect(vec1)
        e3 = u_vect(np.cross(vec1, vec2))
        e1 = u_vect(np.cross(e2, e3))

    frame = np.stack((e1, e2, e3), axis=1)
    return frame


def orient_template(strand, base, template, meta_frame):
    """
    Orient DNA bases

    Parameters:
    -----------
    meta_molecule: :class:`polyply.src.meta_molecule`
    current_node:
        node key of the node in meta_molecule to which template referes to
    template: dict[collections.abc.Hashable, np.ndarray]
        dict of positions referring to the lower resolution atoms of node

    Returns:
    --------
    dict
        the oriented template
    """

    # Calculate intrinsic frame of the base
    template_frame = _gen_base_frame(base, template)

    # Detemine rotation matrices
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

    # if base[:2] == "DG" or base[:2] == "DC":
    #     rotation = Rotation.from_rotvec(np.pi * e2)
    #     template_arr = rotation.apply(template_arr.T).T

    # Transform base template into backward strand variant
    if strand == "backward":
        # Rotate 180 around e2 if backward strand
        rotation = Rotation.from_rotvec(3 * np.pi * e2)
        template_arr = rotation.apply(template_arr.T).T

        # Rotate 180 around e3 if backward strand
        rotation = Rotation.from_rotvec(np.pi * e3)
        template_arr = rotation.apply(template_arr.T).T

    # Rotate base frame into meta_molecule frame
    template_rotated_arr = rot_meta_frame @ (inv_rot_template_frame @ template_arr)

    # Translate rotated template along base-base vector
    if strand == "backward":
        template_final_arr = (template_rotated_arr.T + meta_frame[:, 1] * 0.3).T
    else:
        template_final_arr = (template_rotated_arr.T - meta_frame[:, 1] * 0.3).T

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

    def __init__(self, fudge_coords=0.4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.fudge_coords = fudge_coords
        self.fudge_coords = 1
        self.rotation_angle = 0.59  # 34° in radian
        self.closed = True

    def _place_init_coords(self, meta_molecule):
        """
        For each residue in a class:`polyply.src.MetaMolecule` the
        positions of the atoms associated with that residue stored in
        attr:`polyply.src.MetaMolecule.molecule` are created from a
        template residue located in attr:`polyply.src.MetaMolecule.templates`.

        Parameters
        ----------
        meta_molecule: :class:`polyply.src.MetaMolecule`
        """

        # Read meta_molecule coordinates as array
        X = np.zeros((len(meta_molecule.nodes), 3))
        for node in meta_molecule.nodes:
            X[node] = meta_molecule.nodes[node]['position']

        # Calculate tangents
        T = calc_tangents(X)
        T = np.apply_along_axis(u_vect, axis=1, arr=T)

        # Initialize the reference vector which together with
        # the tangent construct the curve's minimal rotating frame
        R = np.zeros_like(X)
        R[0] = (T[0, 1], -T[0, 0], 0)

        # Construct reference vector's along curve using double reflection
        # Ref: Wang et al. (DOI:10.1145/1330511.1330513)
        for i, _ in enumerate(X[:-1]):
            vec1 = X[i+1] - X[i]
            norm1 = vec1 @ vec1
            R_l = R[i] - (2/norm1) * (vec1 @ R[i]) * vec1
            T_l = T[i] - (2/norm1) * (vec1 @ T[i]) * vec1
            vec2 = T[i+1] - T_l
            norm2 = vec2 @ vec2
            R[i+1] = R_l - (2/norm2) * (vec2 @ R_l) * vec2
        R = np.apply_along_axis(u_vect, axis=1, arr=R)

        # Calculate the binormals
        S = np.cross(T, R)

        # Rotate minimal rotating frame into a darboux frame
        rotation_vectors = [vec * self.rotation_angle *
                            i for i, vec in enumerate(T, start=1)]
        rotation = Rotation.from_rotvec(rotation_vectors)

        # e3=tangent vector, e2=base-base vector, e1=minor grove vector
        e1 = rotation.apply(R)
        e2 = rotation.apply(S)
        e3 = T

        # Comply with boundary conditions if DNA is closed.
        # Uniformly distributing the corrective curviture,
        # minimizes the total squared angular speed.
        if self.closed:
            vec1 = X[0] - X[-1]
            norm1 = vec1 @ vec1
            R_l = e2[-1] - (2/norm1) * (vec1 @ e2[-1]) * vec1
            T_l = e3[-1] - (2/norm1) * (vec1 @ e3[-1]) * vec1
            vec2 = e3[0] - T_l
            norm2 = vec2 @ vec2
            R_calc = R_l - (2/norm2) * (vec2 @ R_l) * vec2
            R_calc = u_vect(R_calc)
            phi = np.arccos(R_calc @ e2[0])

            correction_per_base = phi / len(X)
            for ndx, _ in enumerate(e3):
                R = Rotation.from_rotvec(correction_per_base * (ndx) * e3[ndx])

                temp = R.apply(e2[ndx])
                e2[ndx] = u_vect(temp)

                e1[ndx] = np.cross(e3[ndx], e2[ndx])

        # Construct frame out of generated vectors
        meta_frames = [np.stack((i, j, k), axis=1) for i, j, k in zip(e1,e2,e3)]

        for node in meta_molecule.nodes:
            if meta_molecule.nodes[node]["build"]:
                basepair = meta_molecule.nodes[node]["basepair"]
                cg_coord = meta_molecule.nodes[node]["position"]
                forward_base, backward_base = basepair.split(",")

                # Correctly orientate base on forward and backward strands
                forward_template = orient_template("forward", forward_base,
                                                   meta_molecule.templates[forward_base],
                                                   meta_frames[node])
                backward_template = orient_template("backward", backward_base,
                                                    meta_molecule.templates[backward_base],
                                                    meta_frames[node])

                # Place the molecule atoms according to the backmapping
                high_res_atoms = meta_molecule.nodes[node]["graph"].nodes
                for atom_high in high_res_atoms:
                    atomname = meta_molecule.molecule.nodes[atom_high]["atomname"]
                    base = meta_molecule.molecule.nodes[atom_high]["resname"]
                    if base == forward_base:
                        vector = forward_template[atomname]
                        new_coords = cg_coord + vector * self.fudge_coords
                        meta_molecule.molecule.nodes[atom_high]["position"] = new_coords
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
