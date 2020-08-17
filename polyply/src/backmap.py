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
    attrs: tuple(str, str)
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


def orient_template(meta_molecule, current_node, template):

    # 1. find neighbours at meta_mol level
    neighbours = nx.all_neighbors(meta_molecule, current_node)

    # 2. find connecting atoms at low-res level
    edges = []
    for resid in neighbours:
        edge = find_edges(meta_molecule.molecule,
                            ("resid", "resid"),
                            (resid+1, current_node+1))
        edges += edge

    nodes = []
    for edge in edges:
        if meta_molecule.molecule.nodes[edge[0]]["resid"] == current_node+1:
           nodes.append((edge[0], meta_molecule.molecule.nodes[edge[1]]["resid"]))
        else:
           nodes.append((edge[1], meta_molecule.molecule.nodes[edge[0]]["resid"]))

    # 3. build coordinate system
    ref_coords = np.zeros((len(nodes),3))
    opt_coords = np.zeros((len(nodes),3))
    print(nodes)
    for ndx, node_resid in enumerate(nodes):
        node, resid = node_resid
        atom_name = meta_molecule.molecule.nodes[node]["atomname"]
        opt_coords[ndx, :] = template[atom_name]
        ref_coords[ndx, :] = meta_molecule.nodes[resid-1]["position"]

    print(ref_coords)
    # 4. minimize distance to meta_mol neighbours
    def target_function(angles):
        rotated = rotate_xyz(opt_coords, angles[0], angles[1], angles[2])
        diff = rotated - ref_coords
        #print(np.matmul(diff, diff.T))
        score = np.sum(np.sqrt(np.matmul(diff, diff.T)))
        return score

    angles = np.array([10.0, -10.0, 5.0])
    opt_results = scipy.optimize.minimize(target_function, angles, method='Powell',
                                              options={'ftol':0.001, 'maxiter': 400})
    print(opt_results)
    # 5. rotate template
    template_arr = np.zeros((len(template),3))
    key_to_ndx = {}
    for ndx, key in enumerate(template.keys()):
        template_arr[ndx,:] = template[key]
        key_to_ndx.update({key:ndx})

    angles = opt_results['x']
    template_rotated_arr = rotate_xyz(template_arr, angles[0], angles[1], angles[2])
    template_rotated = {}
    for key, ndx in key_to_ndx.items():
        template_rotated[key] = template_rotated_arr[ndx]

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
        #count = 0
        for node in meta_molecule.nodes:
            resname = meta_molecule.nodes[node]["resname"]
            cg_coord = meta_molecule.nodes[node]["position"]
            template =  orient_template(meta_molecule, node, meta_molecule.templates[resname])
            resid = node + 1
            low_res_atoms = find_atoms(meta_molecule.molecule, "resid", resid)
            for atom_low  in low_res_atoms:
                atomname = meta_molecule.molecule.nodes[atom_low]["atomname"]
                vector = template[atomname]
                new_coords = cg_coord + vector
                if meta_molecule.molecule.nodes[atom_low]["build"]:
                    meta_molecule.molecule.nodes[atom_low]["position"] = new_coords

    def run_molecule(self, meta_molecule):
        """
        Apply placing of coordinates to meta_molecule. For more
        detail see `self._place_init_coords`.
        """
        self._place_init_coords(meta_molecule)
        return meta_molecule
