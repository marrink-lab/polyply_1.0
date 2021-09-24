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
from polyply import jit
from .topology import lorentz_berthelot_rule

def _lennard_jones_force(dist, point, ref, params):
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
    # distances are computed directly from the KD tree, as we compute the force here
    # we also need the distance vector, which does not come from the KDtree. Computing
    # the distance vector here rather than outside the function profits from the numba
    # acceleration. Computing the distance also here would, however, not be faster
    # as the cKDtree is better optimized still. So we get the distance from outside
    # and compute the vector here. All hail the microoptimization.
    vect = point - ref
    force = 24 * eps / dist * ((2 * (sig/dist)**12.0) - (sig/dist)**6) * vect/dist
    return force

# numba implementation if available
lennard_jones_force = jit(_lennard_jones_force)

POTENTIAL_FUNC = {"LJ": lennard_jones_force}


def _n_particles(molecules):
    """
    Count the number of meta_molecule nodes
    in the topology.
    """
    return sum(map(len, molecules))

class NonBondEngine():
    """
    Stores interactions, positions, and pbc
    conditions in an efficent manner such
    that they can be queried, modified and used to
    compute non-bonded forces in a fast manner.
    The class can be created from the `polyply.src.topology`
    class using the class creation method from_topology.
    """

    def __init__(self,
                 positions,
                 nodes_to_idx,
                 atom_types,
                 interaction_matrix,
                 cut_off,
                 boxsize):
        """
        Parameters:
        -----------
        positions: np.ndarray
        nodes_to_idx: dict[(mol_idx, node_key)]
             correspondance dict between indices in position array
             and a unique node in the system identified by molecule
             index and node_key
        atomtypes:  np.ndarray
             array of atom_types corresponding to the atoms in positions
        interaction_matrix:  dict[frozenset(str, str), tuple(float, float)]
             Dict mapping the atom_types to LJ type interaction parameters,
             that is sigma, epsilon or C6, C12 depending on the potential
             used. Currently only the sigma epsilon form is implemented.
        cut_off: float
             cut-off for which to compute the interaction in nm
        boxsize: np.ndarray
             box dimensions in nm
        """

        self.positions = positions
        self.nodes_to_gndx = nodes_to_idx
        self.atypes = np.asarray(atom_types, dtype=str)
        self.interaction_matrix = interaction_matrix
        self.cut_off = cut_off
        self.boxsize = boxsize

        self.defined_idxs = [list(np.where(self.positions[:, 0].reshape(-1) != np.inf)[0])]
        self.position_trees = [scipy.spatial.ckdtree.cKDTree(positions[self.defined_idxs[-1]],
                                                             boxsize=boxsize)]

        # given a global node index, in which tree is the position saved
        self.gndx_to_tree = {idx: 0 for idx in self.defined_idxs[0]}

    def copy(self):
        """
        Return new instance and deep copy of the objects position attribute.
        """
        new_obj = self.__class__(self.positions.copy(),
                                 self.nodes_to_gndx,
                                 self.atypes,
                                 self.interaction_matrix,
                                 cut_off=self.cut_off,
                                 boxsize=self.boxsize)
        return new_obj

    def concatenate_trees(self):
        """
        Rebuild a single tree from all defined coordinates and remove all
        other trees. This function enables you to condense the positions
        saved over multiple trees into a single tree.
        """
        self.defined_idxs = [list(np.where(self.positions[:, 0].reshape(-1) != np.inf)[0])]
        self.position_trees = [scipy.spatial.ckdtree.cKDTree(self.positions[self.defined_idxs[-1]],
                                                             boxsize=self.boxsize)]
        self.gndx_to_tree = {idx: 0 for idx in self.defined_idxs[0]}

    def add_positions(self, point, mol_idx, node_key, start=True):
        """
        Add `point` with global index `gndx` to the position-matrix
        and position tree.
        """
        gndx = self.nodes_to_gndx[(mol_idx, node_key)]
        self.positions[gndx] = point

        # at around 5000 coordinates it is faster to make a new tree than to add the
        # point to the old tree and profit from faster calculation of distances within
        # a single tree
        if start and self.position_trees[-1].n > 5000:
            self.defined_idxs.append([gndx])
            self.gndx_to_tree[gndx] = len(self.position_trees)
            self.position_trees.append(scipy.spatial.ckdtree.cKDTree(point.reshape(1,3),
                                                                     boxsize=self.boxsize,
                                                                     balanced_tree=False,
                                                                     compact_nodes=False))
        else:
            self.defined_idxs[-1].append(gndx)
            self.gndx_to_tree[gndx] = len(self.position_trees) - 1
            new_tree = scipy.spatial.ckdtree.cKDTree(self.positions[self.defined_idxs[-1]],
                                                     boxsize=self.boxsize,
                                                     balanced_tree=False,
                                                     compact_nodes=False)
            self.position_trees[-1] = new_tree

    def remove_positions(self, mol_idx, node_keys):
        """
        Remove `point` with global index `gndx` from the positions
        matrix and position-tree.

        Paramters
        ---------
        mol_idx: int
            index of the molecule for which to remove_positions
        node_keys: abc.iteratable
            which nodes to remove
        """
        tree_idxs = []
        for node_key in node_keys:
            gndx = self.nodes_to_gndx[(mol_idx, node_key)]
            # guard against when a position is not defined
            if gndx not in self.gndx_to_tree:
                continue
            self.positions[gndx] = np.array([np.inf, np.inf, np.inf])
            tree_idx = self.gndx_to_tree[gndx]
            self.defined_idxs[tree_idx].remove(gndx)
            del self.gndx_to_tree[gndx]
            tree_idxs.append(tree_idx)

        for tree_idx in tree_idxs:
            new_tree = scipy.spatial.ckdtree.cKDTree(self.positions[self.defined_idxs[tree_idx]],
                                                     boxsize=self.boxsize,
                                                     balanced_tree=False,
                                                     compact_nodes=False)
            self.position_trees[tree_idx] = new_tree

    def pbc_min_dist(self, pos_a, pos_b):
        """
        Given two nodes and a rectangular box compute
        the distance according to minimum image convention.

        Parameters
        -----------
        pos_a: np.ndarray(3, 1)
        pos_b: np.ndarray(3, 1)
        box:    np.ndarray(3, 1)

        Returns:
        --------
        float
            the minimum distance
        """
        if all(pos_a == np.inf) or all(pos_b == np.inf):
            return np.nan

        box = self.boxsize
        min_dist = np.min(np.vstack(((pos_a - pos_b) % box, (pos_b - pos_a) % box)), axis=0)
        dist = np.linalg.norm(min_dist)
        return dist

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
        """
        Get the position of a point from the molecule index
        and node_key.
        """
        gndx = self.nodes_to_gndx[(mol_idx, node)]
        return self.positions[gndx]

    def get_interaction(self, mol_idx_a, mol_idx_b, node_a, node_b):
        """
        Get the interaction parameters between two nodes identified
        by their molecule indices and node-keys.
        """
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
        force = 0
        for pos_tree, defined_idxs in zip(self.position_trees, self.defined_idxs):

            dist_mat = ref_tree.sparse_distance_matrix(pos_tree, self.cut_off)

            if any(np.array(list(dist_mat.values())) < 0.1):
                return np.inf

            gndx = self.nodes_to_gndx[(mol_idx, node)]
            current_atype = self.atypes[gndx]

            for pair, dist in dist_mat.items():
                gndx_pair = defined_idxs[pair[1]]

                if gndx_pair not in exclusions:
                    other_atype = self.atypes[gndx_pair]
                    params = self.interaction_matrix[frozenset([current_atype, other_atype])]
                    force += POTENTIAL_FUNC[potential](dist, point, self.positions[gndx_pair], params)
        return force

    @classmethod
    def from_topology(cls, molecules, topology, box):
        """
        Create a class instance from a topology object,
        a list of molecules and a box.

        Parameters:
        -----------
        molecules: list
        topology: :class:`polyply.src.topology`
        box: np.nadarray
        """

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
            # as by model definition all epsilon values are set to 1
            params = lorentz_berthelot_rule(topology.volumes[res_a],
                                            topology.volumes[res_b], 1.0, 1.0)
            inter_matrix[frozenset([res_a, res_b])] = params

        for resname in set(atom_types):
            vdw_radius = topology.volumes[resname]
            # as by model definition all epsilon values are set to 1
            inter_matrix[frozenset([resname, resname])] = (vdw_radius, 1.0)

        # dynamically set the cut-off as twice the largest vdw-radius
        cut_off = max(list(inter_matrix.values()))[0] * 2.
        nonbond_matrix = cls(positions, nodes_to_gndx,
                             atom_types, inter_matrix, cut_off=cut_off, boxsize=box)
        return nonbond_matrix
