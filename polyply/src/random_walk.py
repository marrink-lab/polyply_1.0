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
from .processor import Processor
from .linalg_functions import norm_sphere
from .topology import lorentz_berthelot_rule
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


def _is_overlap(meta_molecule, point, tol, current_node, fudge=1):
    """
    Given a `meta_molecule` and a `point`, check if any of
    the positions of in meta_molecule are closer to the
    point than the tolerance `tol` multiplied by a `fudge`
    factor.

    Parameters
    ----------
    meta_molecule:  :class:`polyply.src.meta_molecule.MetaMolecule`
    point: np.ndarray(3)
    tol: float
    fudge: float

    Returns
    -------
    bool
    """
    neighbours = nx.neighbors(meta_molecule, current_node)
    for node in meta_molecule:
        try:
            coord = meta_molecule.nodes[node]["position"]
        except KeyError:
            continue

        if np.linalg.norm(coord - point) < tol * fudge:
           if node in neighbours:
              continue
           else:
              return True

    return False

def update_positions(vector_bundle, meta_molecule, current_node, prev_node):
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
    """
    if "position" in meta_molecule.nodes[current_node]:
        return

    current_vectors = vector_bundle.copy()
    last_point = meta_molecule.nodes[prev_node]["position"]

    prev_resname = meta_molecule.nodes[prev_node]["resname"]
    current_resname = meta_molecule.nodes[current_node]["resname"]

    current_vdwr = meta_molecule.volumes[current_resname]
    prev_vdwr = meta_molecule.volumes[prev_resname]
    vdw_radius, _ = lorentz_berthelot_rule(current_vdwr, prev_vdwr, 1, 1)

    # we give 10 percent more than the vdw radius so we don't generate a
    # self ovelap
    step_length = vdw_radius * 0.83
    while True:
        new_point, index = _take_step(vector_bundle, step_length, last_point)
        if not _is_overlap(meta_molecule, new_point, vdw_radius, current_node):
            meta_molecule.nodes[current_node]["position"] = new_point
            break
        else:
            vector_bundle = np.delete(vector_bundle, index, axis=0)


class RandomWalk(Processor):
    """
    Add coordinates at the meta_molecule level
    through a random walk for all nodes which have
    build defined as true.
    """

    @staticmethod
    def _random_walk(meta_molecule):
        """
        Perform a random_walk to build positions for a meta_molecule, if
        no position is present for an atom.

        Parameters
        ----------
        meta_molecule:  :class:`polyply.src.meta_molecule.MetaMolecule`
        """
        first_node = list(meta_molecule.nodes)[0]
        meta_molecule.nodes[first_node]["position"] = np.array([0, 0, 0])
        vector_bundle = norm_sphere(5000)
        for prev_node, current_node in nx.dfs_edges(meta_molecule, source=0):
            update_positions(vector_bundle, meta_molecule,
                             current_node, prev_node)

    def run_molecule(self, meta_molecule):
        """
        Perform the random walk for a single molecule.
        """
        self._random_walk(meta_molecule)
        return meta_molecule
