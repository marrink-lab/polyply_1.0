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
import polyply
from polyply import TEST_DATA
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


#@pytest.mark.parametrize('box_vect, point, result', (
#    (np.array([5., 5., 10.]),
#     np.array([1., 1., 0.5]),
#     True
#     ),
#    (np.array([5., 5., 10.]),
#     np.array([5., 6., 0.5]),
#     False
#     )))
#def test_not_exceeds_max_dimensions(box_vect, point, result):
#    assert _filter_by_molname(molecules, ignore)



