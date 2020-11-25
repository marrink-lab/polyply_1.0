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
    #print(total_mass)
    # amu -> kg and cm3 -> nm3
    #conversion = 1.6605410*10**-27 * 10**27
    box = (total_mass*1.6605410/density)**(1/3.)
    return box

def _filter_by_molname(molecules, ignore):
    ellegible_molecules = []
    for molecule in molecules:
        if molecule.mol_name not in ignore:
            ellegible_molecules.append(molecule)
    return ellegible_molecules

class BuildSystem():
    """
    Compose a system of molecules according
    to the definitions in the topology file.
    """

    def __init__(self, topology,
                 density,
                 start_dict,
                 max_force=10**3,
                 grid_spacing=0.2,
                 maxiter=800,
                 maxiter_random=50,
                 box=[],
                 step_fudge=1,
                 push=[],
                 ignore=[],
                 grid=None,
                 nrewind=5):

        self.topology = topology
        self.density = density
        self.grid_spacing = grid_spacing
        self.maxiter = maxiter
        self.push = push
        self.step_fudge = step_fudge
        self.maxiter_random = maxiter_random
        self.max_force = max_force
        self.ignore = ignore
        self.box_grid = grid
        self.start_dict = start_dict
        self.nrewind = nrewind

        # set the box if a box is given
        if len(box) != 0:
            self.box = box
        # if box is not given but density compute it from density
        else:
            box_dim = round(_compute_box_size(topology, self.density), 5)
            self.box = np.array([box_dim, box_dim, box_dim])

        # intialize the grid if there is none given
        if isinstance(self.box_grid, type(None)):
            self.box_grid = np.mgrid[0:self.box[0]:self.grid_spacing,
                                     0:self.box[1]:self.grid_spacing,
                                     0:self.box[2]:self.grid_spacing].reshape(3, -1).T

        # this should be done elsewhere
        topology.box = (self.box[0], self.box[1], self.box[2])

    def _handle_random_walk(self, molecule, mol_idx, vector_sphere):
        step_count = 0
        while True:
            start_idx = np.random.randint(len(self.box_grid))
            start = self.box_grid[start_idx]

            processor = RandomWalk(mol_idx,
                                   self.nonbond_matrix.copy(),
                                   step_fudge=self.step_fudge,
                                   start=start,
                                   maxiter=50,
                                   maxdim=self.box,
                                   max_force=self.max_force,
                                   vector_sphere=vector_sphere,
                                   push=self.push,
                                   nrewind=self.nrewind,
                                   start_node=self.start_dict[mol_idx])

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
        pbar = tqdm(total=len(molecules))
        mol_tot = len(molecules)
        vector_sphere = norm_sphere(5000)
        while mol_idx < mol_tot:

            molecule = molecules[mol_idx]

            if all(["position" in molecule.nodes[node] for node in molecule.nodes]):
                mol_idx += 1
                pbar.update(1)
                continue

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
        # filter all molecules that should be ignored during the building process
        self.molecules = _filter_by_molname(self.topology.molecules, self.ignore)

        # generate the nonbonded matrix wrapping all information about molecular
        # interactions
        self.nonbond_matrix = NonBondMatrix.from_topology(self.molecules, self.topology, self.box)
        self._compose_system(self.molecules)
        return molecules
