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
Test that solvent molecules are placed correctly.
"""
import textwrap
import pytest
import numpy as np
from vermouth.forcefield import ForceField
import polyply
from polyply.src.top_parser import read_topology
from polyply.src.topology import Topology
from polyply.src.nonbond_engine import NonBondEngine
from polyply.src.solvator import Solvator

@pytest.fixture
def topology():
    top_lines ="""
    [ defaults ]
    1   1   no   1.0     1.0
    [ atomtypes ]
    N0 45.0 0.000 A 0.0 0.0
    [ nonbond_params ]
    N0   N0   1 4.700000e-01    3.700000e+00
    [ moleculetype ]
    testA 1
    [ atoms ]
    1    N0  1   PEO    BB   1 0.00     45
    2    N0  2   PEO    BB   1 0.00     45
    [ bonds ]
    1    2    1  0.47 2000
    [ moleculetype ]
    testB 1
    [ atoms ]
    1    N0  1   GLY    BB   1 0.00     45
    2    N0  1   GLY    SC1  1 0.00     45
    [ bonds ]
    1    2    1  0.47 2000
    [ moleculetype ]
    testC 1
    [ atoms ]
    1    N0  1   ASP    BB   1 0.00
    [ system ]
    test system
    [ molecules ]
    testA 1
    testB 4
    testC 4
    """
    lines = textwrap.dedent(top_lines)
    lines = lines.splitlines()
    force_field = ForceField("test")
    topology = Topology(force_field)
    read_topology(lines=lines, topology=topology, cwdir="./")
    topology.preprocess()
    topology.volumes = {"ASP":0.2, "GLY": 0.4, "PEO": 0.4}
    return topology

@pytest.mark.parametrize('volumes, target_spacing', (
    ({"ASP":0.2, "GLY": 0.4, "PEO": 0.4},
     0.23802992,
    ),
    ({"ASP":0.6, "GLY": 0.4, "PEO": 0.4},
    0.34618275,
    ),))
def test_clever_grid(topology, volumes, target_spacing):
    topology.volumes = volumes
    nb_engine = NonBondEngine.from_topology(topology.molecules,
                                            topology,
                                            box=np.array([4., 4., 4.]))
    solvent_placer = Solvator(nb_engine,
                              max_force=10**5,
                              mol_idxs=[1,2,3,4,5,6,7,8])
    current_grid = solvent_placer.clever_grid()
    ref_grid = np.mgrid[0:4.0:target_spacing,
                        0:4.0:target_spacing,
                        0:4.0:target_spacing].reshape(3, -1).T
    for idx in range(0, ref_grid.shape[0]):
        assert np.allclose(ref_grid[idx], current_grid[idx])

@pytest.mark.parametrize('mol_idxs, pos, ', (
    # simple test; create all coordinates
    ([1, 2, 3, 4, 5, 6, 7, 8],
     None,
    ),
    # here we skip some already build solvents
    ([3, 4, 5, 6, 7, 8],
     np.array([[1.0, 1.0, 0.67],
               [1.0, 1.0, 1.37]]),
     ),
    # here we need to skip one already defined position
    ([2, 4, 5, 6, 7, 8],
     np.array([[1.0, 1.0, 0.67],
               [np.inf, np.inf, np.inf],
               [1.0, 1.0, 1.30]]),
    ),
     ))
def test_run_molecule(topology, mol_idxs, pos):
    # add PEO positions
    topology.molecules[0].nodes[0]["position"] = np.array([0.0, 0.0, 0.0])
    topology.molecules[0].nodes[1]["position"] = np.array([0.0, 0.0, 0.32])

    # add positions and labels of other molecules
    if pos is not None:
        for idx in range(0, pos.shape[0]):
            topology.molecules[idx+1].nodes[0]["position"] = pos[idx, :]

    # collect all top information in nonbond engine
    nb_engine = NonBondEngine.from_topology(topology.molecules,
                                            topology,
                                            box=np.array([4., 4., 4.]))

    solvent_placer = Solvator(nb_engine,
                              max_force=10**5,
                              mol_idxs=mol_idxs)
    solvent_placer.run_system(topology.molecules)

    # make sure all position are defined
    for coord in solvent_placer.nonbond_matrix.positions:
        assert all(coord != np.array([np.inf, np.inf, np.inf]))

    # check that defined positions are presrved
    assert all(solvent_placer.nonbond_matrix.positions[0] == np.array([0.0, 0.0, 0.0]))
    assert all(solvent_placer.nonbond_matrix.positions[1] == np.array([0.0, 0.0, 0.32]))

    solvent_placer.nonbond_matrix.concatenate_trees()

    if pos is not None:
        for idx in range(0, pos.shape[0]):
            if all(pos[idx, :] != np.array([np.inf, np.inf, np.inf])):
                internal_pos = solvent_placer.nonbond_matrix.positions[idx+2]
                assert all(internal_pos == pos[idx, :])
                force = solvent_placer.nonbond_matrix.compute_force_point(point=internal_pos,
                                                                          mol_idx=idx+1,
                                                                          node=0)
                assert np.linalg.norm(force) < 10**5.

@pytest.mark.parametrize('dist, expected',
                        # LJ force is zero at minimum
            			((2**(1/6.)*0.35,
                          np.array([0.0, 0.0, 0.0])),
                        # LJ force is also zero at infinity
                         (100000.0,
                          np.array([0.0, 0.0, 0.0])),
                        # pick one point inbetween
                         (0.42,
                          np.array([0.0, 0.0, 13.27016])),
                        # distance along different vector
                         (0.42,
                          np.array([0.0, 13.27016, 0.0])),
                        # truly 3d distance
                         (0.48989794855663565,
                          np.array([4.09965989, 8.19931978, 4.09965989]))
                        ))
def test_LJ_force(dist, expected):
     ref = np.linalg.norm(expected)
     value = polyply.src.solvator.norm_lennard_jones_force(dist, sig=0.35, eps=2.1)
     assert np.allclose(value, ref)
