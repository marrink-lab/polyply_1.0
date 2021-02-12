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
Test the nonbond engine computes forces and such accurately
"""
import pytest
import textwrap
import numpy as np
from vermouth.forcefield import ForceField
import polyply
from polyply.src.topology import Topology
from polyply.src.top_parser import read_topology
from polyply.src.nonbond_engine import NonBondEngine

@pytest.fixture
def topology():
    top_lines ="""
    [ defaults ]
    1   1   no   1.0     1.0
    [ atomtypes ]
    N0 72.0 0.000 A 0.0 0.0
    [ nonbond_params ]
    N0   N0   1 4.700000e-01    3.700000e+00
    [ moleculetype ]
    testA 1
    [ atoms ]
    1    N0  1   GLY    BB   1 0.00     45
    2    N0  1   GLY    SC1  1 0.00     45
    3    N0  1   GLY    SC2  1 0.00     45
    4    N0  2   GLU    BB   2 0.00     45
    5    N0  2   GLU    SC1  2 0.00     45
    6    N0  2   GLU    SC2  2 0.00     45
    [ bonds ]
    1    2    1  0.47 2000
    2    3    1  0.47 2000
    1    4    1  0.47 2000
    4    5    1  0.47 2000
    4    6    1  0.47 2000
    [ moleculetype ]
    testB 1
    [ atoms ]
    1    N0  1   ASP    BB   1 0.00     45
    2    N0  1   ASP    SC1  1 0.00     45
    3    N0  1   ASP    SC2  1 0.00     45
    [ bonds ]
    1    2    1  0.47 2000
    2    3    1  0.47 2000
    [ system ]
    test system
    [ molecules ]
    testA 2
    testB 1
    """
    lines = textwrap.dedent(top_lines)
    lines = lines.splitlines()
    force_field = ForceField("test")
    topology = Topology(force_field)
    read_topology(lines=lines, topology=topology, cwdir="./")
    topology.preprocess()
    topology.volumes = {"GLY": 0.53, "GLU": 0.67, "ASP": 0.43}
    return topology

@pytest.mark.parametrize('n_pos', (
      0,
      3,
      5))
def  test_nb_engine_from_top_file(topology, n_pos):

     # we add positions to the topology skipping some
     # positins depending on how many are set in n_pos
     positions = np.random.random(n_pos*3).reshape(-1, 3)
     for mol_idx, mol in enumerate(topology.molecules):
         for node in mol.nodes:
             try:
                mol.nodes[node]["position"] = positions[mol_idx + node]
             except IndexError:
                continue

     # initiate the nb_engine
     nb_engine =  NonBondEngine.from_topology(topology.molecules,
                                              topology,
                                              box=np.array([10., 10., 10.]))

     # these are the expected interactions in the nb_engine
     nb_interactions = {
                        frozenset(["GLY", "GLY"]): 0.53,
                        frozenset(["GLY", "GLU"]): (0.53+0.67)/2.0,
                        frozenset(["GLY", "ASP"]): (0.43+0.53)/2.0,
                        frozenset(["ASP", "ASP"]): 0.43,
                        frozenset(["ASP", "GLU"]): (0.43+0.67)/2.0,
                        frozenset(["GLU", "GLU"]): 0.67,
                       }

     # make sure the cut_off is set dynamically and accurate
     assert nb_engine.cut_off == 2*0.67

     # check if all interactions are created properly
     for res_combo, interaction in nb_interactions.items():
          assert nb_engine.interaction_matrix[res_combo] == (nb_interactions[res_combo], 1.0)

     # make sure that all positions are in the global position matrix
     # and positions which are undefined have a infinity set
     for mol_idx, mol in enumerate(topology.molecules):
         for node in mol.nodes:
            if "position" in mol.nodes[node]:
                assert all(nb_engine.get_point(mol_idx, node) == mol.nodes[node]["position"])
            else:
                assert all(nb_engine.positions[mol_idx, node] == np.array([np.inf, np.inf, np.inf]))

def test_add_positions(topology):
     # we add 2 positions to the nb_matrix
     positions = np.random.random(2*3).reshape(-1, 3)
     for mol_idx, mol in enumerate(topology.molecules):
         for node in mol.nodes:
             try:
                mol.nodes[node]["position"] = positions[mol_idx + node]
             except IndexError:
                continue

     nb_engine =  NonBondEngine.from_topology(topology.molecules,
                                              topology,
                                              box=np.array([10., 10., 10.]))

     # now we check if the 3rd is added correctly
     nb_engine.add_positions(np.array([5.0, 5.0, 5.0]), mol_idx=2, node_key=0, start=False)
     assert all(nb_engine.get_point(2, 0) == np.array([5.0, 5.0, 5.0]))

def test_remove_positions(topology):
     # we add 5 positions to the nb_matrix
     positions = np.random.random(5*3).reshape(-1, 3)
     for mol_idx, mol in enumerate(topology.molecules):
         for node in mol.nodes:
             try:
                mol.nodes[node]["position"] = positions[mol_idx + node]
             except IndexError:
                continue

     nb_engine =  NonBondEngine.from_topology(topology.molecules,
                                              topology,
                                              box=np.array([10., 10., 10.]))

     # now we check if the 3rd is removed correctly
     nb_engine.remove_positions(mol_idx=2, node_keys=[0])
     assert all(nb_engine.get_point(2, 0) == np.array([np.inf, np.inf, np.inf]))

def test_update_positions_in_molecules(topology):
     # we add 15 random positions to all molecules
     positions = np.random.random(5*3).reshape(-1, 3)
     for mol_idx, mol in enumerate(topology.molecules):
         for node in mol.nodes:
             print(mol_idx, node)
             mol.nodes[node]["position"] = positions[mol_idx + node]

     nb_engine =  NonBondEngine.from_topology(topology.molecules,
                                              topology,
                                              box=np.array([10., 10., 10.]))
     # now we overwrite those positoins one
     nb_engine.positions[:, :] = np.ones((5, 3), dtype=float)[:, :]
     # then we update them in the molecules and check if they are actually updated
     nb_engine.update_positions_in_molecules(topology.molecules)
     for mol_idx, mol in enumerate(topology.molecules):
         for node in mol.nodes:
            assert all(mol.nodes[node]["position"] == np.array([1., 1., 1.]))

@pytest.mark.parametrize('mol_idx_a, mol_idx_b, node_a, node_b, expected',
                        ((0, 0, 0, 1, (0.53+0.67)/2.0),
                         (0, 2, 0, 0, (0.43+0.53)/2.0),
                         (0, 2, 1, 0, (0.43+0.67)/2.0)))
def test_get_interaction(topology, mol_idx_a, mol_idx_b, node_a, node_b, expected):
     # initiate the nb_engine
     nb_engine =  NonBondEngine.from_topology(topology.molecules,
                                              topology,
                                              box=np.array([10., 10., 10.]))

     # check get interactions works properly
     value = nb_engine.get_interaction(mol_idx_a=mol_idx_a,
                                       mol_idx_b=mol_idx_b,
                                       node_a=node_a,
                                       node_b=node_b)
     assert value == (expected, 1.0)

@pytest.mark.parametrize('dist, ref, expected',
                        # LJ force is zero at minimum
			((2**(1/6.)*0.35,
                          np.array([0.0, 0.0, 2**1/6.*0.35]),
                          np.array([0.0, 0.0, 0.0])),
                        # LJ force is also zero at infinity
                         (100000.0,
                          np.array([0.0, 0.0, 100000.0]),
                          np.array([0.0, 0.0, 0.0])),
                        # pick one point inbetween
                         (0.42,
                          np.array([0.0, 0.0, 0.42]),
                          np.array([0.0, 0.0, 13.27016])),
                        # distance along different vector
                         (0.42,
                          np.array([0.0, 0.42, 0.0]),
                          np.array([0.0, 13.27016, 0.0])),
                        # truly 3d distance
                         (0.42,
                          np.array([0.2, 0.4, 0.2]),
                          np.array([ 6.31912, 12.63825,  6.31912]))
                        ))
def test_LJ_force(dist, ref, expected):
     point = np.array([0.0, 0.0, 0.0])
     params = (0.35, 2.1)
     value = polyply.src.nonbond_engine._lennard_jones_force(dist, point, ref, params)
     print(value)
     assert np.allclose(value, expected)
