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
import itertools
import numpy as np
import scipy.spatial
from numba import jit
from .topology import lorentz_berthelot_rule

@jit(nopython=True, cache=True, fastmath=True)
def lennard_jones_force(dist, vect, params):
    """
    Compute the Lennard-Jones force between two particles
    given their interaction parameters, the distance, and
    distance vector.

    Parameters:
    -----------
    dist: float
        the distance between particles
    vect: np.ndarray(3)
        the distance vector
    params: tuple
        tuple of sigma and epsilon parameters

    Returns:
    --------
    numpy.ndarray(3)
        the force vector
    """
    sig, eps = params
    force = 24 * eps / dist * ((2 * (sig/dist)**12.0) - (sig/dist)**6) * vect/dist
    return force


POTENTIAL_FUNC = {"LJ": lennard_jones_force}


def _n_particles(molecules):
    """
    Count the number of meta_molecule nodes
    in the topology.
    """
    n_atoms = 0
    for molecule in molecules:
        n_atoms += len(molecule.nodes)
    return n_atoms


class NonBondMatrix():
    """
    Object for handeling nonbonded interactions
    of the topology. It stores the positions using
    infinity for undefined positions. It also creats
    a mapping of node and molecule index to the global
    index in the position matrix as well as a cdk tree
    of defined positions and a mapping between global
    indices and defined positions. Furthermore, it stores
    the interaction matrix between atomtypes and the atom
    types, where each atomtype corresponds to one of the
    global indices.
    """

    def __init__(self,
                 positions,
                 nodes_to_idx,
                 atom_types,
                 interaction_matrix,
                 cut_off,
                 boxsize):

        self.positions = positions
        self.nodes_to_gndx = nodes_to_idx
        self.atypes = np.asarray(atom_types, dtype=str)
        self.interaction_matrix = interaction_matrix
        self.cut_off = cut_off
        self.boxsize = boxsize

        self.defined_idxs = np.where(self.positions[:, 0].reshape(-1) != np.inf)[0]
        self.position_tree = scipy.spatial.ckdtree.cKDTree(positions[self.defined_idxs],
                                                           boxsize=boxsize)

    def copy(self):
        """
        Return new instance and deep copy of the objects position attribute.
        """
        new_obj = NonBondMatrix(self.positions.copy(),
                                self.nodes_to_gndx,
                                self.atypes,
                                self.interaction_matrix,
                                cut_off=self.cut_off,
                                boxsize=self.boxsize)
        return new_obj

    def update_positions(self, point, mol_idx, node_key):
        """
        Add `point` with global index `gndx` to the nonbonded definitions.
        """
        gndx = self.nodes_to_gndx[(mol_idx, node_key)]
        self.positions[gndx] = point
        self.defined_idxs = np.where(self.positions[:, 0] != np.inf)[0]
        self.position_tree = scipy.spatial.ckdtree.cKDTree(self.positions[self.defined_idxs],
                                                           boxsize=self.boxsize, balanced_tree=False, compact_nodes=False)
    def update_positions_in_molecules(self, molecules):
        """
        Add the positions stored in the object back
        to the molecules.
        """
        for mol_idx, molecule in enumerate(molecules):
            for node in molecule.nodes:
                gndx = self.nodes_to_gndx[(mol_idx, node)]
                molecule.nodes[node]["position"] = self.positions[gndx]

    def get_point(self, mol_idx, node):
        gndx = self.nodes_to_gndx[(mol_idx, node)]
        return self.positions[gndx]

    def get_interaction(self, mol_idx_a, mol_idx_b, node_a, node_b):
        gndx_a = self.nodes_to_gndx[(mol_idx_a, node_a)]
        gndx_b = self.nodes_to_gndx[(mol_idx_b, node_b)]
        atype_a = self.atypes[gndx_a]
        atype_b = self.atypes[gndx_b]
        return self.interaction_matrix[frozenset([atype_a, atype_b])]

    def compute_force_point(self, point, mol_idx, node, exclude=[], potential="LJ"):
        """
        Compute the force on `node` of molecule `mol_idx` with coordinates
        `point` given the potential definition in `potential`.

        Parameters
        ----------
        point: np.ndarray(3)
        mol_idx: int
        node: abc.hashable
            node key
        exclude: list[collections.abc.hashable]
            list of nodes to exclude from computation
        potential: abc.hashable
            definition of potential to use

        Returns
        -------
        np.ndarray(3)
            the force vector
        """
        exclusions = [self.nodes_to_gndx[(mol_idx, node)] for node in exclude]

        ref_tree = scipy.spatial.ckdtree.cKDTree(point.reshape(1, 3),
                                                 boxsize=self.boxsize)
        dist_mat = ref_tree.sparse_distance_matrix(self.position_tree, self.cut_off)

        if any(np.array(list(dist_mat.values())) < 0.1):
           return np.inf

        gndx = self.nodes_to_gndx[(mol_idx, node)]
        current_atype = self.atypes[gndx]

        force = 0
        for pair, dist in dist_mat.items():
            gndx_pair = self.defined_idxs[pair[1]]
            if gndx_pair not in exclusions:
                other_atype = self.atypes[gndx_pair]
                params = self.interaction_matrix[frozenset([current_atype, other_atype])]
                vect = point - self.positions[gndx_pair]
                force += POTENTIAL_FUNC[potential](dist, vect, params)

        return force

    @classmethod
    def from_topology(cls, molecules, topology, box):

        n_atoms = _n_particles(molecules)

        # array of all positions
        positions = np.ones((n_atoms, 3)) * np.inf
        # convert molecule index and node to index
        # in position matrix
        nodes_to_gndx = {}

        atom_types = []
        idx = 0
        mol_count = 0
        for molecule in molecules:
            for node in molecule.nodes:
                if "position" in molecule.nodes[node]:
                    positions[idx, :] = molecule.nodes[node]["position"]

                resname = molecule.nodes[node]["resname"]
                atom_types.append(resname)
                nodes_to_gndx[(mol_count, node)] = idx
                idx += 1

            mol_count += 1

        inter_matrix = {}
        for res_a, res_b in itertools.combinations(set(atom_types), r=2):
            params = lorentz_berthelot_rule(topology.volumes[res_a],
                                            topology.volumes[res_b], 1, 1)
            inter_matrix[frozenset([res_a, res_b])] = params

        for resname in set(atom_types):
            vdw_radius = topology.volumes[resname]
            inter_matrix[frozenset([resname, resname])] = (vdw_radius, 1)

        nonbond_matrix = cls(positions, nodes_to_gndx,
                             atom_types, inter_matrix, cut_off=2.1, boxsize=box)
        return nonbond_matrix
