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

import random
import networkx as nx
import numpy as np
import scipy
from numpy.linalg import norm
from .processor import Processor
from .linalg_functions import norm_sphere
"""
Processor implementing a random-walk to generate
coordinates for a meta-molecule.
"""


def _take_step(vectors, step_length, coord):
    """
    Given a list of unit `vectors` choose one randomly,
    multiply it by the `step_length` and add to `coord`.

    Parameters
    ----------
    vectors: list[np.ndarray(n, 3)]
    step_length: float
    coord: np.ndarray(3)

    Returns
    -------
    np.ndarray(3)
        the new coordinate
    int
        the index of the vector choosen
    """
    index = random.randint(0, len(vectors) - 1)
    new_coord = coord + vectors[index] * step_length
    return new_coord, index


def _is_overlap(point, positions, atom_types, vdw_radii, gndx):
    """
    Given a `meta_molecule` and a `point`, check if any of
    the positions of in meta_molecule are closer to the
    point than the tolerance `tol` multiplied by a `fudge`
    factor.

    Parameters
    ----------
    meta_molecules:  list[:class:`polyply.src.meta_molecule.MetaMolecule`]
    point: np.ndarray(3)
    tol: float
    fudge: float

    Returns
    -------
    bool
    """
    red_pos = positions[positions[:, 0] != np.inf]
    traj_tree = scipy.spatial.ckdtree.cKDTree(red_pos)
    ref_tree = scipy.spatial.ckdtree.cKDTree(point.reshape(1,3))
    dist_mat = ref_tree.sparse_distance_matrix(traj_tree, 1.1)
    red_types = atom_types[positions[:, 0] != np.inf ]

    current_atom = atom_types[gndx]
    for pair, dist in dist_mat.items():
        ref = vdw_radii[frozenset([current_atom, red_types[pair[1]]])]
        if dist < ref*1.2:
           return True
    return False

def not_exceeds_max_dimensions(point, maxdim):
    return np.all(point < maxdim) and np.all(point > np.array([0., 0., 0.]))


class RandomWalk(Processor):
    """
    Add coordinates at the meta_molecule level
    through a random walk for all nodes which have
    build defined as true.
    """

    def __init__(self,
                 positions,
                 nodes_to_idx,
                 atom_types,
                 vdw_radii,
                 mol_idx,
                 start=np.array([0, 0, 0]),
                 topology=None,
                 maxiter=50,
                 maxdim=None,
                 vector_sphere=norm_sphere(5000)):

        self.start = start
        self.topology = topology
        self.maxiter = maxiter
        self.success = False
        self.maxdim = maxdim
        self.positions = positions.copy()
        self.nodes_to_gndx = nodes_to_idx
        self.atom_types = np.asarray(atom_types, dtype=str)
        self.vdw_radii = vdw_radii
        self.mol_idx = mol_idx
        self.vector_sphere = vector_sphere

    def update_positions(self, vector_bundle, meta_molecule, current_node, prev_node):
        """
        Take an array of unit vectors `vector_bundle` and generate the coordinates
        for `current_node` by adding a random vector to the position of the previous
        node `prev_node`. The length of that vector is defined as 2 times the vdw-radius
        of the two nodes. The position is updated in place.

        Parameters
        ----------
        vector_bunde: np.ndarray(m,3)
        meta_molecule: :class:polyply.src.meta_molecule.MetaMolecule
        current_node: node_key[int, str]
        prev_node: node_key[int, str]
        topology: :class:polyply.src.topology.Topology
        maxiter: int
           maximum number of iterations
        """
        if "position" in meta_molecule.nodes[current_node]:
            return True
        gndx_prev = self.nodes_to_gndx[(self.mol_idx, prev_node)]
        gndx_current = self.nodes_to_gndx[(self.mol_idx, current_node)]

        last_point = meta_molecule.nodes[prev_node]["position"]
        res_current = self.atom_types[gndx_current]
        res_prev =  self.atom_types[gndx_prev]
        vdw_radius = self.vdw_radii[frozenset([res_prev, res_current])]

        step_length = vdw_radius
        step_count = 0

        while True:
            new_point, index = _take_step(vector_bundle, step_length, last_point)
            overlap = _is_overlap(new_point, self.positions, self.atom_types, self.vdw_radii, gndx_current)
            in_box = not_exceeds_max_dimensions(new_point, self.maxdim)
            if not overlap and in_box:
                #print(step_count)
                meta_molecule.nodes[current_node]["position"] = new_point
                self.positions[gndx_current, :] = new_point
                return True
            elif step_count == self.maxiter:
                return False
            else:
                step_count += 1
                vector_bundle = np.delete(vector_bundle, index, axis=0)


    def _random_walk(self, meta_molecule):
        """
        Perform a random_walk to build positions for a meta_molecule, if
        no position is present for an atom.

        Parameters
        ----------
        meta_molecule:  :class:`polyply.src.meta_molecule.MetaMolecule`
        """
        first_node = list(meta_molecule.nodes)[0]
        if "position" not in meta_molecule.nodes[first_node]:
            gndx_current = self.nodes_to_gndx[(self.mol_idx, first_node)]
            if not _is_overlap(self.start, self.positions, self.atom_types, self.vdw_radii, gndx_current):
                meta_molecule.nodes[first_node]["position"] = self.start
                self.positions[gndx_current ,:] = self.start
                self.success = True
            else:
                self.success = False
                return

        vector_bundle = self.vector_sphere.copy()
        for prev_node, current_node in nx.dfs_edges(meta_molecule, source=0):
            status = self.update_positions(vector_bundle,
                                           meta_molecule,
                                           current_node,
                                           prev_node)
            self.success = status
            if not self.success:
               return

    def run_molecule(self, meta_molecule):
        """
        Perform the random walk for a single molecule.
        """
        self._random_walk(meta_molecule)
        return meta_molecule
