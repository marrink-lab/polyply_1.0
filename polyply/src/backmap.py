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
import multiprocessing
import numpy as np
import scipy.optimize
import networkx as nx
from tqdm import tqdm
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

        nodes = [molecule.nodes[node_a], molecule.nodes[node_b]]
        if all(node[at] == val for (node, at, val) in zip(nodes, attr, value)) or\
           all(node[at] == val for (node, at, val) in zip(reversed(nodes), attr, value)):
            edges.append(edge)

    return edges


#@profile
def orient_template(meta_molecule, current_node, template, built_nodes):
    """
    Given a `template` and a `node` of a `meta_molecule` at lower resolution
    find the bonded interactions connecting the higher resolution template
    to its neighbours and orient a template such that the atoms point torwards
    the neighbours. In case some nodes of meta_molecule have already been built
    at the lower resolution they can be provided as `built_nodes`. In case the
    lower resolution is already built the atoms will be oriented torward the lower
    resolution atom participating in the bonded interaction.

    Parameters:
    -----------
    meta_molecule: :class:`polyply.src.meta_molecule`
    current_node:
        node key of the node in meta_molecule to which template referes to
    template: dict[collections.abc.Hashable, np.ndarray]
        dict of positions referring to the lower resolution atoms of node
    built_nodes: list
        list of meta_molecule node keys of residues that are already built

    Returns:
    --------
    dict
        the oriented template
    """
    # 1. find neighbours at meta_mol level
    neighbours = nx.all_neighbors(meta_molecule, current_node)
    current_resid = meta_molecule.nodes[current_node]["resid"]

    # 2. find connecting atoms at low-res level
    edges = []
    for node in neighbours:
        resid = meta_molecule.nodes[node]["resid"]
        edge = find_edges(meta_molecule.molecule,
                          ("resid", "resid"),
                          (resid, current_resid))
        edges += edge

    # 3. build coordinate system
    ref_coords = np.zeros((3, len(edges)))
    opt_coords = np.zeros((3, len(edges)))

    for ndx, edge in enumerate(edges):
        for atom in edge:
            resid = meta_molecule.molecule.nodes[atom]["resid"]
            if resid == current_resid:
                current_atom = atom
            else:
                ref_atom = atom
                ref_resid = resid

        # the reference residue has already been build so we take the lower
        # resolution coordinates as reference
        if ref_resid in built_nodes:
            atom_name = meta_molecule.molecule.nodes[current_atom]["atomname"]

            # record the coordinates of the atom that is rotated
            opt_coords[:, ndx] = template[atom_name]

            # given the reference atom that already exits translate it to the origin
            # of the rotation, this will be the reference point for rotation
            ref_coords[:, ndx] = meta_molecule.molecule.nodes[ref_atom]["position"] -\
                                 meta_molecule.nodes[current_node]["position"]

        # the reference residue has not been build the CG center is taken as
        # reference
        else:
            atom_name = meta_molecule.molecule.nodes[current_atom]["atomname"]
            cg_node = find_atoms(meta_molecule, "resid", ref_resid)[0]

            # record the coordinates of the atom that is rotated
            opt_coords[:, ndx] = template[atom_name]

            # as the reference atom is not built take the cg node as reference point
            # for rotation; translate it to origin
            ref_coords[:, ndx] = meta_molecule.nodes[cg_node]["position"] -\
                                 meta_molecule.nodes[current_node]["position"]


    # 4. optimize the distance between reference nodes and nodes to be placed
    # only using rotation of the complete template
    def target_function(angles):
        rotated = rotate_xyz(opt_coords, angles[0], angles[1], angles[2])
        diff = rotated - ref_coords
        score = np.sum(np.sum(diff*diff, axis=0))
        return score

    # starting angles in degree
    angles = np.array([10.0, -10.0, 5.0])
    opt_results = scipy.optimize.minimize(target_function, angles, method='L-BFGS-B',
                                          options={'ftol':0.000001, 'maxiter': 400})

    # 5. write the template as array and rotate it corrsponding to the result above
    template_arr = np.zeros((3, len(template)))
    key_to_ndx = {}
    for ndx, key in enumerate(template.keys()):
        template_arr[:, ndx] = template[key]
        key_to_ndx[key] = ndx

    angles = opt_results['x']
    template_rotated_arr = rotate_xyz(template_arr, angles[0], angles[1], angles[2])

    # 6. write the template back as dictionary
    template_rotated = {}
    for key, ndx in key_to_ndx.items():
        template_rotated[key] = template_rotated_arr[:, ndx]

    return template_rotated

class Backmap():
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
        for node in meta_molecule.nodes:
            #print(node)
            if  meta_molecule.nodes[node]["build"]:
                resname = meta_molecule.nodes[node]["resname"]
                cg_coord = meta_molecule.nodes[node]["position"]
                resid = meta_molecule.nodes[node]["resid"]
                low_res_atoms = find_atoms(meta_molecule.molecule, "resid", resid)
                template = orient_template(meta_molecule, node,
                                           meta_molecule.templates[resname],
                                           built_nodes)

                for atom_low  in low_res_atoms:
                    atomname = meta_molecule.molecule.nodes[atom_low]["atomname"]
                    vector = template[atomname]
                    new_coords = cg_coord + vector
                    meta_molecule.molecule.nodes[atom_low]["position"] = new_coords
                built_nodes.append(resid)

    def run_molecule(self, meta_molecule):
        """
        Apply placing of coordinates to meta_molecule. For more
        detail see `self._place_init_coords`.
        """
        self._place_init_coords(meta_molecule)
        return meta_molecule

    def run_system(self, system):
        """
        Process `system`.
        Parameters
        ----------
        system: vermouth.system.System
            The system to process. Is modified in-place.
        """
        pool = multiprocessing.Pool(12)
        system.molecules = pool.map(self.run_molecule, tqdm(system.molecules))
