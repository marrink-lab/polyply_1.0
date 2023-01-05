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
import inspect
import numpy as np
from tqdm import tqdm
from .random_walk import RandomWalk
from .linalg_functions import norm_sphere
from .nonbond_engine import NonBondEngine
from .persistence import sample_end_to_end_distances
from .restraints import set_restraints

def _compute_box_size(topology, density):
    """
    Lookup the masses in the `topology`
    and compute a cubic box that matches
    the `density` given the number of molecules
    defined in the topology. Units are nm

    Parameters:
    -----------
    topology: :class:`polyply.src.topology`
    density: float
       target density

    Returns:
    --------
    float
        the edge length of cubix box
    """
    total_mass = 0
    for meta_molecule in topology.molecules:
        molecule = meta_molecule.molecule
        for node in molecule.nodes:
            if 'mass' in molecule.nodes[node]:
                total_mass += molecule.nodes[node]['mass']
            else:
                try:
                    atype = molecule.nodes[node]["atype"]
                    total_mass += topology.atom_types[atype]['mass']
                except KeyError as error:
                    msg = ("Trying to compute system density, but cannot "
                           "find mass of atom {} with type {} in topology.")
                    atom = molecule.nodes[node]["atomname"]
                    raise KeyError(msg.format(atom, atype)) from error

    # amu -> kg and cm3 -> nm3
    # conversion = 1.6605410*10**-27 * 10**27
    box = (total_mass*1.6605410/density)**(1/3.)
    return box

def _filter_by_molname(molecules, ignore):
    """
    Selects all molecules from `molecules` which
    do not have a name mentioned in `ignore`.
    """
    for molecule in molecules:
        if molecule.mol_name not in ignore:
            yield molecule

class BuildSystem():
    """
    Compose a system of molecules according
    to the definitions in the topology file.

    This class when run for a system or a molecule
    calls the random-walk processor to generate the
    coordinates for a single moleccule. In contrast
    to the random-walk processor this class handles
    all system related accouting matters such as
    starting points, box-dimensions and existing
    molecules and so forth.

    Parameters:
    -----------
    topology: :class:`polyply.src.topology`
    density: float
        the system density
    start_dict: dict
        a dictionary associating ...
    grid_spacing: flaot
        the distance between grid points
    grid: np.ndarray
        a grid defining grid points this argument
        overwrides generation of the grid using
        the grid-spacing
    maxiter: int
        maximum number of tries to genrate a single
        molecule
    box: np.ndarray[3,1]
        box size, this overwrites generation of the box
        by the density argument
    ignore: list[str]
        list of molecule names to ignore when building
    **kwargs:
        all passed down to random-walk
    """

    def __init__(self,
                 topology,
                 density,
                 start_dict,
                 grid_spacing=0.2,
                 maxiter=800,
                 box=None,
                 ignore=[],
                 grid=None,
                 cycles=[],
                 **kwargs):

        self.topology = topology
        self.density = density
        self.grid_spacing = grid_spacing
        self.maxiter = maxiter
        self.ignore = ignore
        self.box_grid = grid
        self.box = box
        self.start_dict = start_dict
        self.molecules = []
        self.nonbond_matrix = None
        self.cycles = cycles

        # first we check if **kwargs are actually in random-walk
        valid_kwargs = inspect.getfullargspec(RandomWalk).args
        for kwarg in kwargs:
            if kwarg not in valid_kwargs:
                msg = ("Keyword argument {} is not valid for the "
                       "RandomWalk processor class. ")
                raise TypeError(msg.format(kwarg))
        self.rwargs = kwargs

        # set the box if a box is given
        # we use type comparison because box is an array
        if not isinstance(self.box, type(None)):
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
                                   self.nonbond_matrix,
                                   start=start,
                                   maxdim=self.box,
                                   vector_sphere=vector_sphere,
                                   start_node=self.start_dict[mol_idx],
                                   **self.rwargs)

            processor.run_molecule(molecule)

            if processor.success:
                return True, processor.nonbond_matrix
            elif step_count == self.maxiter:
                processor.nonbond_matrix.remove_positions(mol_idx, molecule.nodes)
                return False, processor.nonbond_matrix
            else:
                step_count += 1
                self.nonbond_matrix.remove_positions(mol_idx, molecule.nodes)

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
                self.nonbond_matrix.concatenate_trees()
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
        self.molecules = list(_filter_by_molname(self.topology.molecules, self.ignore))
        # generate the nonbonded matrix wrapping all information about molecular
        # interactions
        self.nonbond_matrix = NonBondEngine.from_topology(self.molecules, self.topology, self.box)
        # apply sampling of persistence length
        sample_end_to_end_distances(self.topology, self.nonbond_matrix)
        # set any other distance and/or position restraints
        set_restraints(self.topology, self.nonbond_matrix)
        self._compose_system(self.topology.molecules)
        return molecules
