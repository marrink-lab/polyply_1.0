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
import scipy.optimize
import networkx as nx
from .processor import Processor
from .generate_templates import find_atoms
from .linalg_functions import rotate_xyz
from tqdm import tqdm
"""
Processor implementing a template based back
mapping to lower resolution coordinates for
a meta molecule.
"""

def find_edges(molecule, attr, value):
    """
    Find all edges of a `vermouth.molecule.Molecule` that have the
    attribute `attr` with the corresponding value of `value`. Note
    that this function is not order specific. That means (1, 2) is
    the same as (2, 1).

    Parameters
    ----------
    molecule: :class:vermouth.molecule.Molecule
    attrs: tuple[str, str]
         tuple of the attributes used in matching
    values: tuple
         corresponding values value

    Returns
    ----------
    list
       list of edges found
    """
    edges = []
    for edge in molecule.edges:
        node_a, node_b = edge

        if attr[0] in molecule.nodes[node_a]:
           if molecule.nodes[node_a][attr[0]] == value[0]:
              if  attr[1] in molecule.nodes[node_b]:
                 if molecule.nodes[node_b][attr[1]] == value[1]:
                     edges.append(edge)

        if attr[1] in molecule.nodes[node_a]:
           if molecule.nodes[node_a][attr[1]] == value[1]:
               if attr[0] in molecule.nodes[node_b]:
                  if molecule.nodes[node_b][attr[0]] == value[0]:
                      edges.append(edge)

    return edges


def orient_template(meta_molecule, current_node, template, built_nodes):

    # 1. find neighbours at meta_mol level
    neighbours = nx.all_neighbors(meta_molecule, current_node)

    # 2. find connecting atoms at low-res level
    edges = []
    for resid in neighbours:
        edge = find_edges(meta_molecule.molecule,
                            ("resid", "resid"),
                            (resid+1, current_node+1))
        edges += edge

    # 3. build coordinate system
    ref_coords = np.zeros((3, len(edges)))
    opt_coords = np.zeros((3, len(edges)))
    for ndx, edge in enumerate(edges):
        atom_a, atom_b = edge
        resid_a = meta_molecule.molecule.nodes[atom_a]["resid"] - 1
        resid_b = meta_molecule.molecule.nodes[atom_b]["resid"] - 1
        if resid_a in built_nodes or resid_b in built_nodes:
            if resid_a == current_node:
                 atom_name = meta_molecule.molecule.nodes[atom_a]["atomname"]
                 aa_node = atom_b
            else:
                 atom_name = meta_molecule.molecule.nodes[atom_b]["atomname"]
                 aa_node = atom_a

            opt_coords[:, ndx] = template[atom_name]
            ref_coords[:, ndx] = meta_molecule.molecule.nodes[aa_node]["position"] - meta_molecule.nodes[current_node]["position"]

        else:
            if resid_a == current_node:
                 atom_name = meta_molecule.molecule.nodes[atom_a]["atomname"]
                 cg_node = resid_b
            else:
                 atom_name = meta_molecule.molecule.nodes[atom_b]["atomname"]
                 cg_node = resid_a

            opt_coords[:, ndx] = template[atom_name]
            ref_coords[:, ndx] = meta_molecule.nodes[cg_node]["position"] - meta_molecule.nodes[current_node]["position"]


    # 4. minimize distance to meta_mol neighbours
    def target_function(angles):
        rotated = rotate_xyz(opt_coords, angles[0], angles[1], angles[2])
        diff = rotated - ref_coords
        score = np.sum(np.sqrt(np.sum(diff*diff, axis=0)))
        return score

    angles = np.array([10.0, -10.0, 5.0])
    opt_results = scipy.optimize.minimize(target_function, angles, method='L-BFGS-B',
                                              options={'ftol':0.000001, 'maxiter': 400})
    # 5. rotate template
    template_arr = np.zeros((3,len(template)))
    key_to_ndx = {}
    for ndx, key in enumerate(template.keys()):
        template_arr[:, ndx] = template[key]
        key_to_ndx[key] = ndx

    angles = opt_results['x']
    template_rotated_arr = rotate_xyz(template_arr, angles[0], angles[1], angles[2])
    rotated = rotate_xyz(opt_coords, angles[0], angles[1], angles[2])
    diff = rotated - ref_coords

    template_rotated = {}
    for key, ndx in key_to_ndx.items():
        template_rotated[key] = template_rotated_arr[:,ndx]

    return template_rotated

class Backmap(Processor):
    """
    This processor takes a a class:`polyply.src.MetaMolecule` and
    places coordinates form a higher resolution createing positions
    for the lower resolution molecule associated with the MetaMolecule.
    """
    @staticmethod
    def _place_init_coords(meta_molecule):
        """
        For each residue in a class:`polyply.src.MetaMolecule` the
        positions of the atoms associated with that residue stored in
        attr:`polyply.src.MetaMolecule.molecule` are created from a
        template residue located in attr:`polyply.src.MetaMolecule.templates`.

        Parameters
        ----------
        meta_molecule: :class:`polyply.src.MetaMolecule`
        """
        built_nodes = []
        for node in tqdm(meta_molecule.nodes):
            if  meta_molecule.nodes[node]["build"]:
                resname = meta_molecule.nodes[node]["resname"]
                cg_coord = meta_molecule.nodes[node]["position"]
                resid = node + 1
                low_res_atoms = find_atoms(meta_molecule.molecule, "resid", resid)
                template =  orient_template(meta_molecule, node, meta_molecule.templates[resname], built_nodes)

                for atom_low  in low_res_atoms:
                    atomname = meta_molecule.molecule.nodes[atom_low]["atomname"]
                    vector = template[atomname]
                    new_coords = cg_coord + vector
                    meta_molecule.molecule.nodes[atom_low]["position"] = new_coords
                built_nodes.append(node)

    def run_molecule(self, meta_molecule):
        """
        Apply placing of coordinates to meta_molecule. For more
        detail see `self._place_init_coords`.
        """
        self._place_init_coords(meta_molecule)
        return meta_molecule
