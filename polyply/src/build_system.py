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
# this processor is not a subclass to the regular processor
# class because it cannot run on a single molecule but needs
# the system information
"""
Processor for building systems with more than one molecule
"""
import numpy as np
from tqdm import tqdm
from .random_walk import RandomWalk
from .linalg_functions import norm_sphere
from .nonbond_matrix import NonBondMatrix

def _compute_box_size(topology, density):
    total_mass = 0
    for meta_molecule in topology.molecules:
        molecule = meta_molecule.molecule
        for node in molecule.nodes:
            if 'mass' in molecule.nodes[node]:
                total_mass += molecule.nodes[node]['mass']
            else:
                atype = molecule.nodes[node]["atype"]
                total_mass += topology.atom_types[atype]['mass']
    print(total_mass)
    # amu -> kg and cm3 -> nm3
    #conversion = 1.6605410*10**-27 * 10**27
    box = (total_mass*1.6605410/density)**(1/3.)
    return box

class BuildSystem():
    """
    Compose a system of molecules according
    to the definitions in the topology file.
    """

    def __init__(self, topology,
                 density,
                 n_grid_points=500,
                 maxiter=800,
                 box_size=None):

        self.topology = topology
        self.density = density
        self.n_grid_points = n_grid_points
        self.maxiter = maxiter

        if box_size:
            self.box_size = box_size
        else:
            self.box_size = round(_compute_box_size(topology, self.density), 5)

        self.box_grid = np.arange(0, self.box_size, self.box_size/self.n_grid_points)
        self.maxdim = np.array([self.box_size, self.box_size, self.box_size])
        topology.box = (self.box_size, self.box_size, self.box_size)

        self.nonbond_matrix = NonBondMatrix.from_topology(topology)

    def _handle_random_walk(self, molecule, mol_idx, vector_sphere):
        step_count = 0
        while True:
            start = self.box_grid[np.random.randint(len(self.box_grid), size=3)]
            processor = RandomWalk(mol_idx,
                                   self.nonbond_matrix.copy(),
                                   start=start,
                                   maxiter=50,
                                   maxdim=self.maxdim,
                                   max_force=10**3.0,
                                   vector_sphere=vector_sphere)

            processor.run_molecule(molecule)
            if processor.success:
                return True, processor.nonbond_matrix
            elif step_count == self.maxiter:
                return False, processor.nonbond_matrix
            else:
                step_count += 1

    def _compose_system(self, molecules):
        """
        Place the molecules of the system into a box
        and optimize positions to meet density.

        Parameters
        ----------
        topology:  :class:`vermouth.system`
        density: foat
           density of the system in kg/cm3

        Returns
        --------
        system
        """
        mol_idx = 0
        pbar = tqdm(total=len(self.topology.molecules))
        mol_tot = len(self.topology.molecules)
        vector_sphere = norm_sphere(5000)
        while mol_idx < mol_tot:
            molecule = molecules[mol_idx]
            if all([ "position" in molecule.nodes[node] for node in molecule.nodes]):
                mol_idx += 1
                pbar.update(1)

            success, new_nonbond_matrix = self._handle_random_walk(molecule,
                                                                   mol_idx,
                                                                   vector_sphere)
            if success:
                self.nonbond_matrix = new_nonbond_matrix
                mol_idx += 1
                pbar.update(1)

        pbar.close()
        self.nonbond_matrix.update_positions_in_molecules(molecules)

    def run_system(self, molecules):
        """
        Compose a system according to a the system
        specifications and a density value.
        """
        self._compose_system(molecules)
        return molecules
