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
"""
Test that force field files are properly read.
"""
import math
import pytest
import numpy as np
from numpy.linalg import norm
import vermouth
import polyply
from polyply import TEST_DATA
from polyply.src.meta_molecule import Monomer, MetaMolecule
from polyply.src.topology import Topology
from polyply.src.nonbond_matrix import NonBondMatrix
from polyply.src.build_system import (_compute_box_size,
                                      _filter_by_molname
                                     )

@pytest.fixture
def nonbond_matrix():
    toppath = TEST_DATA + "/struc_build/system.top"
    topology = Topology.from_gmx_topfile(name="test", path=toppath)
    topology.preprocess()
    setattr(topology, "volumes", {"PEO":0.43})
    return NonBondMatrix.from_topology(topology.molecules,
                                       topology,
                                       box=np.array([10., 10., 10.]))

def create_topology(n_molecules):
    toppath = TEST_DATA + "/struc_build/system.top"
    topology = Topology.from_gmx_topfile(name="test", path=toppath)
    molecule = topology.molecules[0]
    print(molecule)
    for n in range(1, n_molecules):
        topology.molecules.append(molecule)
    return topology


def add_positions(nb_matrix, ncoords):
    pos = np.array([[1.0, 1.0, 0.37],
                    [1.0, 1.0, 0.74],
                    [1.0, 1.0, 1.11],
                    [1.0, 1.0, 1.48],
                    [1.0, 1.0, 1.85],
                    [1.0, 1.0, 2.22],
                    [1.0, 1.0, 2.59],
                    [1.0, 1.0, 2.96],
                    [1.0, 1.0, 3.33],
                    [1.0, 1.0, 3.70],
                    [1.0, 1.0, 4.07]])

    nb_matrix.add_positions(pos[0], mol_idx=0, node_key=0, start=True)
    for idx, point in enumerate(pos[1:ncoords]):
        nb_matrix.add_positions(point, mol_idx=0, node_key=idx+1, start=False)
    return nb_matrix

@pytest.mark.parametrize('density, result', (
    (1000.0,
     4.2119903964305125
     ),
    (600.0,
     4.993866813213379
     )))
def test_compute_box_size(density, result):
    top = create_topology(100)
    assert np.isclose(_compute_box_size(top, density), result)


@pytest.fixture()
def test_system():
    """
    Create a dummy test system with three types of molecules AA, BB and
    NA. NA is the molecule to be used a ligand. AA and BB are composed
    of different residues.
    """
    # dummy vermouth force-field
    force_field = vermouth.forcefield.ForceField(name='test_ff')
    # monomers used in the meta-molecule
    ALA = Monomer(resname="ALA", n_blocks=2)
    GLU = Monomer(resname="GLU", n_blocks=1)
    THR = Monomer(resname="THR", n_blocks=1)
    # two meta-molecules
    meta_mol_A = MetaMolecule.from_monomer_seq_linear(force_field,
                                                      [ALA, GLU, THR],
                                                      "AA")
    meta_mol_B = MetaMolecule.from_monomer_seq_linear(force_field,
                                                      [GLU, ALA, THR],
                                                      "BB")
    NA = MetaMolecule()
    NA.add_monomer(current=0, resname="NA", connections=[])
    # molecules for the system
    molecules = [meta_mol_A, meta_mol_A.copy(),
                 meta_mol_B.copy(), NA, NA.copy(),
                 NA.copy(), NA.copy()]
    # create the topology
    top = Topology(force_field=force_field)
    # set topology attributes
    top.molecules = molecules
    top.name = "test"
    #top.atom_types = {}
    #top.types = defaultdict(dict)
    #top.nonbond_params = {}
    setattr(top, "volumes", {"ALA": 0.5, "GLU": 0.8, "THR": 1.2})
    top.mol_idx_by_name = {"AA":[0, 1], "BB": [2], "NA":[3, 4, 5, 6]}
    return top

@pytest.mark.parametrize('ignore', (
    ["NA"],
    ["AA", "NA"],
     ))
def test_filer_by_molname(test_system, ignore):
    molecules = _filter_by_molname(test_system.molecules, ignore)
    for molecule in molecules:
        assert molecule.mol_name not in ignore


#@pytest.mark.parametrize('ignore', (
#      density,
#      start_dict,
#      max_force=10**3,
#      grid_spacing=0.2,
#      maxiter=800,
#      maxiter_random=50,
#      box=[],
#      step_fudge=1,
#      push=[],
#      ignore=[],
#      grid=None):

#def test_run_molecule(test_system, defaults):
#   # iterate over
#   # density vs box
#   # grid_spacind vs grid
#   # start and ignore
#   BuildSystem(test_system,
#               start_dict=start_dict,
#               density=args.density,
#               max_force=args.max_force,
#               grid_spacing=args.grid_spacing,
#               maxiter=args.maxiter,
#               maxiter_random=args.maxiter_random,
#               box=box,
#               step_fudge=args.step_fudge,
#               push=args.push,
#               ignore=args.ignore,
#               grid=grid).run_system(topology.molecules)
