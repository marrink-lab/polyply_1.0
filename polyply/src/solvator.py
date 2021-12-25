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
from numpy.random import default_rng
from numpy.linalg import norm
import scipy
import scipy.optimize
from tqdm import tqdm
from .processor import Processor
from .nonbond_engine import POTENTIAL_FUNC
from .random_walk import fulfill_geometrical_constraints

def norm_lennard_jones_force(dist, sig, eps):
    """
    Norm of the LJ force between two particles.
    """
    return np.abs(24 * eps / dist * ((2 * (sig/dist)**12.0) - (sig/dist)**6))

class Solvator(Processor):
    """
    Generate coordinates for 1 residue
    molecules that are often solvent
    molecules.
    """
    def __init__(self,
                 nonbond_matrix,
                 max_force,
                 mol_idxs,
                 potential="LJ",
                ):
        """
        Parameters
        ----------
        nonbond_matrix: :class:`polyply.src.nonbond_engine.NonBondMatrix`
            the nonbonded forces and positions of the system
        max_force: float
            maximum allowed force on a particle in the one bead per residue
            description
        mol_idxs:
            the indices of the solvent molecules to build
        potential:
            the form of the nonbonded potential
        """
        self.nonbond_matrix = nonbond_matrix
        # concatenate all trees into one
        self.nonbond_matrix.concatenate_trees()
        # for convenience
        self.box = self.nonbond_matrix.boxsize
        self.max_force = max_force
        self.mol_idxs = np.array(mol_idxs, dtype=int)
        self.potential = potential

    def clever_grid(self):
        """
        Given the largest sigma value of a collection of one bead per residue
        description of solvents generate a grid that has as miniumum grid spacing
        the distance at which the LJ force is at most the maximum force-criterion
        provided. This also works for a mixed system of solvents and sizes, because
        all cross interactions are geometric averages meaning they cannot exceed
        the size of the self-interaction.
        """
        # find the largest self interaction sigma between two of the solvent beads
        max_sigma = 0
        for idx in self.mol_idxs:
            gndx = self.nonbond_matrix.nodes_to_gndx[(idx, 0)]
            atype_key = frozenset([self.nonbond_matrix.atypes[gndx]])
            sigma = self.nonbond_matrix.interaction_matrix[atype_key][0]
            if sigma > max_sigma:
                print(atype_key)
                max_sigma = sigma
        # compute the minimum distance these beads need to have in order
        # to fullfill the max-force criterion
        def _min_ljn(dist):
            return (norm_lennard_jones_force(dist, sig=max_sigma, eps=1.0) - self.max_force)**2.0

        # at distances smaller than 0.1 non-bond forces are considered infinite
        # and larger than 1.1 are beyond cut-off; hence the bounds of the search
        min_dist = scipy.optimize.minimize_scalar(_min_ljn, method='bounded', bounds=(0.1, 1.1)).x
        print(min_dist)
        # build grid
        grid = np.mgrid[0:self.box[0]:min_dist,
                        0:self.box[1]:min_dist,
                        0:self.box[2]:min_dist].reshape(3, -1).T
        print(len(grid))
        return grid

    def run_system(self, molecules):
        """
        Add coordinates for solvent molecules to a system as
        defined by the molecule indices (self.mol_idxs).
        """
        # the solvent molecules to be placed
        not_placed_sols = np.full((self.mol_idxs.shape[0]), True)
        print(np.sum(not_placed_sols))
        sol_idxs = np.arange(0, self.mol_idxs.shape[0])
        sol_coords = np.zeros((self.mol_idxs.shape[0], 3))
        # initialize the grid
        grid = self.clever_grid()
        indices = np.arange(0, len(grid))
        # a boolean mask recording which grid points are used
        avail_indices = np.full((len(grid)), True)
        # the progress bar
        pbar = tqdm(total=not_placed_sols.shape[0])
        while np.sum(not_placed_sols) > 0:
            rng = default_rng()
            selected_idx = rng.choice(indices[avail_indices],
                                      size=not_placed_sols.sum(),
                                      replace=False)

            points = grid[selected_idx]
            for sol_idx, point in zip(sol_idxs[not_placed_sols], points):
                force = self.nonbond_matrix.compute_force_point(point,
                                                                self.mol_idxs[sol_idx],
                                                                node=0)
                node = molecules[self.mol_idxs[sol_idx]].nodes[0]
                if fulfill_geometrical_constraints(point, node) and\
                   norm(force) < self.max_force:

                    not_placed_sols[sol_idx] = False
                    sol_coords[sol_idx][:] = point[:]
                    pbar.update(1)

            avail_indices[selected_idx] = False

        for mol_idx, coord in zip(self.mol_idxs, sol_coords):
            self.nonbond_matrix.add_no_tree_pos(coord,
                                                mol_idx,
                                                node_key=0,
                                                start=True)

        pbar.close()
        return molecules
