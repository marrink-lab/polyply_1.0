# Copyright 2022 University of Groningen
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
"""
Class for placing rigid fragments within polyply. Rigid fragements
can be a single residue or collections fo residues.
"""
import numpy as np
from numpy.linalg import norm
import networkx as nx
from .processor import Processor
from .linalg_functions import pbc_complete, norm_sphere, center_of_geometry, rotate_xyz
from .graph_utils import neighborhood
from .random_walk import fulfill_geometrical_constraints

class RigidFragmentPlacer(Processor):
    """
    Places fragments as rigid units.
    """
    def __init__(self,
                 mol_idx,
                 nonbond_matrix,
                 box_grid,
                 maxiter=80,
                 maxrot=100,
                 maxdim=None,
                 max_force=1e3,
                 vector_sphere=norm_sphere(5000)):

        # required input
        self.box_grid = box_grid
        self.mol_idx = mol_idx
        self.nonbond_matrix = nonbond_matrix
        # optional input
        self.maxiter = maxiter
        self.maxrot = maxrot
        self.maxdim = maxdim
        self.max_force = max_force
        self.vector_sphere = vector_sphere
        # attributes for internal use
        self.success = False
        self.molecule = None

    def is_overlap(self, coords, nodes, nrexcl=1):
        for point, node in zip(coords, nodes):
            neighbours = neighborhood(self.molecule, node, nrexcl)
            force_vect = self.nonbond_matrix.compute_force_point(point,
                                                                 self.mol_idx,
                                                                 node,
                                                                 neighbours,
                                                                 potential="LJ")
            if norm(force_vect) > self.max_force:
                return True
        return False

    def fulfill_geometrical_constraints(self, coords, nodes):
        for point, node in zip(coords, nodes):
            node_dict = self.molecule.nodes[node]
            if not fulfill_geometrical_constraints(point, node_dict):
                return False
        return True

    def generate_new_orientation(self, coords, rotate):
        """
        Generate a new orientation for positions.
        """
        angles = np.random.uniform(low=0, high=2*np.pi, size=(3))
        rotated = rotate_xyz(coords, angles[0], angles[1], angles[2])
        rotated = pbc_complete(rotated, self.maxdim)
        return rotated

    def generate_new_location(self, coords):
        """
        Pick a new locatoin for the rigid fragment.
        """
        start_idx = np.random.randint(len(self.box_grid))
        new_coords = coords #a+ (self.box_grid[start_idx] -
                             #  coords[0])
        #new_coords = pbc_complete(new_coords, self.maxdim)
        return new_coords

    def update_coordiantes(self, coords, nodes):
        if self.fulfill_geometrical_constraints(coords, nodes)\
            and not self.is_overlap(coords, nodes):
            for new_point, current_node in zip(coords, nodes):
                self.nonbond_matrix.add_positions(new_point,
                                                  self.mol_idx,
                                                  current_node,
                                                  start=False)
            return True
        return False

    def place_fragment(self, fragment_graph, coords, rotate=(True, True, True)):
        """
        Attempt to place a rigid fragment.
        """
        step_count = 0
        while True:
            # pick a location
            loc_coords = self.generate_new_location(coords)
            # for a given location try several orientations
           # if any(rotate):
           #     for _ in np.arange(0, self.maxrot, 1):
           #         rot_coords = self.generate_new_orientation(loc_coords, rotate)
           #         status = self.update_coordiantes(rot_coords, fragment_graph.nodes)
           ##         if status:
            #            return True
            #else:
            status = self.update_coordiantes(loc_coords, fragment_graph.nodes)
            if status:
                return True

            if step_count == self.maxiter:
                return False

            step_count += 1

        return False

    def place_molecule(self, meta_molecule):
        """
        Place a rigid fragment in a molecule.
        """
        # get which builder to use for one residue
        build_attrs = nx.get_node_attributes(meta_molecule, "builder")

        # build sub-graph of rigid fragment
        rigid_nodes = [node for node, val in build_attrs.items() if val == "rigid"]
        frag_graph = meta_molecule.subgraph(rigid_nodes)
        rotate = list(set(meta_molecule.nodes[node]['rotate'] for node in frag_graph.nodes))

        # check which rotations are allowd; this error shouldn't be raised
        # when calling from CLI or via build file as defaults are properly
        # set
        if len(rotate) > 1:
            raise IOError("Rotations must be the same for all nodes in the fragment")

        # check that there is only one connected component
        if not nx.is_connected(frag_graph):
            raise IOError("At the moment polyply allows only 1 rigid fragment per molecule.")

        # check if all coords are present
        if not all("position" in meta_molecule.nodes[node] for node in frag_graph.nodes):
            raise IOError("At the moment auto generation of rigid fragments is not supported.")

        coords = np.asarray([meta_molecule.nodes[node]["position"] for node in frag_graph.nodes])
        print(rotate[0])
        self.success = self.place_fragment(frag_graph, coords, rotate=rotate)
        return meta_molecule

    def run_molecule(self, meta_molecule):
        """
        Build single lipid on grid.
        """
        self.molecule = meta_molecule
        self.place_molecule(meta_molecule)
        return meta_molecule
