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

def norm_lennard_jones_force(dist, sig, eps, force):
    """
    Norm of the LJ force between two particles.
    """
    return 24 * eps / dist * ((2 * (sig/dist)**12.0) - (sig/dist)**6) - force

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
                max_sigma = sigma
        # compute the minimum distance these beads need to have in order
        # to fullfill the max-force criterion
        min_dist = scipy.optimize.fsolve(norm_lennard_jones_force,
                                         x0=0.2,
                                         args=(max_sigma, 1.0, self.max_force))
        # build grid
        grid = np.mgrid[0:self.box[0]:min_dist,
                        0:self.box[1]:min_dist,
                        0:self.box[2]:min_dist].reshape(3, -1).T
        return grid

    def compute_forces(self, points):
        """
        Given an array of `points` compute the forces these points
        have in relation to the coordinates saved in the nonbond
        matrix.

        Parameters
        ----------
        points: np.ndrarry
            the points to compute the forces for

        Returns
        -------
        forces: np.ndarray
            an array of the force vectors corresponding to each point
        """
        #TODO
        # move this implementation to the nonbond matrix
        # probably should refactor the force computation then as well
        # but that will take thorough performance checking
        forces = np.zeros((points.shape[0]))
        ref_tree = scipy.spatial.ckdtree.cKDTree(points,
                                                 boxsize=self.box)

        pos_tree = self.nonbond_matrix.position_trees[0]
        dist_mat = ref_tree.sparse_distance_matrix(pos_tree, self.nonbond_matrix.cut_off)

        for (sol_idx, gndx), dist in dist_mat.items():
            sol_gndx = self.nonbond_matrix.nodes_to_gndx[(sol_idx, 0)]
            atype_sol = self.nonbond_matrix.atypes[sol_gndx]
            atype_ref = self.nonbond_matrix.atypes[gndx]
            params = self.nonbond_matrix.interaction_matrix[frozenset([atype_sol, atype_ref])]
            force_vect = POTENTIAL_FUNC[self.potential](dist,
                                                        points[sol_idx],
                                                        self.nonbond_matrix.positions[gndx],
                                                        params)
            norm_force = norm(force_vect)
            forces[sol_idx] += norm_force
        return forces

    def run_system(self, molecules):
        """
        Add coordinates for solvent molecules to a system as
        defined by the molecule indices (self.mol_idxs).
        """
        # the solvent molecules to be placed
        not_placed_sols = np.full((self.mol_idxs.shape[0]), True)
        sol_idxs = np.arange(0, self.mol_idxs.shape[0])
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
            forces = self.compute_forces(points)
            for sol_idx, point, force in zip(sol_idxs[not_placed_sols], points, forces):
                if force < self.max_force:
                    not_placed_sols[sol_idx] = False
                    self.nonbond_matrix.add_positions(point,
                                                      self.mol_idxs[sol_idx],
                                                      node_key=0,
                                                      start=True)
                    pbar.update(1)

            avail_indices[selected_idx] = False

        pbar.close()
        return molecules
