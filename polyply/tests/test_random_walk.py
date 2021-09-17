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
import networkx as nx
import polyply
from polyply import TEST_DATA
from polyply.src.topology import Topology
from polyply.src.nonbond_engine import NonBondEngine
from polyply.src.random_walk import (fulfill_geometrical_constraints,
                                     pbc_complete,
                                     not_exceeds_max_dimensions,
                                     _take_step,
                                     RandomWalk,
                                     is_restricted)

@pytest.mark.parametrize('restraint_dict, point, result', (
    # test single geometrical constraint
    ({"restraints": [["in", np.array([0.0, 0.0, 0.0]), 1.0, "sphere"]]},
     np.array([0, 0, 0.5]),
     True
     ),
    ({"restraints": [["out", np.array([0.0, 0.0, 0.0]), 1.0, "sphere"]]},
     np.array([0, 0, 1.5]),
     True
     ),
    ({"restraints": [["in", np.array([0.0, 0.0, 0.0]), 1.0, "sphere"]]},
     np.array([0, 0, 1.5]),
     False
     ),
    ({"restraints": [["out", np.array([0.0, 0.0, 0.0]), 1.0, "sphere"]]},
     np.array([0.0, 0.0, 0.5]),
     False
     ),
    ({"restraints": [["in", np.array([0.0, 0.0, 0.0]), 1.0, 2.0, "cylinder"]]},
     np.array([0, 0.5, 0.5]),
     True
     ),
    ({"restraints": [["out", np.array([0.0, 0.0, 0.0]), 1.0, 2.0, "cylinder"]]},
     np.array([0, 1.5, 1.5]),
     True
     ),
    ({"restraints": [["in", np.array([0.0, 0.0, 0.0]), 1.0, 2.0, "cylinder"]]},
     np.array([0, 1.5, 1.5]),
     False
     ),
    ({"restraints": [["out", np.array([0.0, 0.0, 0.0]), 2.0, 2.0, 4.0, "rectangle"]]},
     np.array([0, 0.5, 0.5]),
     False
     ),
    ({"restraints": [["in", np.array([0.0, 0.0, 0.0]), 2.0, 2.0, 4.0, "rectangle"]]},
     np.array([0, 1.0, 0.5]),
     True
     ),
    ({"restraints": [["out", np.array([0.0, 0.0, 0.0]), 2.0, 2.0, 4.0, "rectangle"]]},
     np.array([0, 1.5, 4.5]),
     True
     ),
    ({"restraints": [["in", np.array([0.0, 0.0, 0.0]), 2.0, 2.0, 4.0, "rectangle"]]},
     np.array([0, 1.5, 4.5]),
     False
     ),
    ({"restraints": [["out", np.array([0.0, 0.0, 0.0]), 2.0, 2.0, 4.0, "rectangle"]]},
     np.array([0, 1.5, 3.9]),
     False
     ),
    # test default empty dict
    ({},
     np.array([0, 1.5, 3.9]),
     True
     ),
))
def test_geometric_restrictions(restraint_dict, point, result):
    assert fulfill_geometrical_constraints(point, restraint_dict) == result


@pytest.mark.parametrize('box_vect, point, result', (
    (np.array([5., 5., 10.]),
     np.array([0, 0, 0.5]),
     np.array([0, 0, 0.5])
     ),
    (np.array([5., 5., 10.]),
     np.array([5., 6., 0.5]),
     np.array([0., 1.0, 0.5])
    ),
    (np.array([5., 5., 10.]),
     np.array([5., -3., 0.5]),
     np.array([0., 2.0, 0.5])
     )))
def test_pbc_complete(box_vect, point, result):
    assert all(pbc_complete(point, box_vect) == result)


@pytest.mark.parametrize('box_vect, point, result', (
    (np.array([5., 5., 10.]),
     np.array([1., 1., 0.5]),
     True
     ),
    (np.array([5., 5., 10.]),
     np.array([5., 6., 0.5]),
     False
     )))
def test_not_exceeds_max_dimensions(box_vect, point, result):
    assert not_exceeds_max_dimensions(point, box_vect) == result


def test__take_step():
    coord = np.array([1.0, 1.0, 1.0])
    step_length = 0.5
    vectors = polyply.src.linalg_functions.norm_sphere(50)
    new_coord, _ = _take_step(
        vectors, step_length, coord, np.array([5.0, 5.0, 5.0]))
    assert math.isclose(norm(new_coord - coord), step_length)


@pytest.fixture
def nonbond_matrix():
    toppath = TEST_DATA + "/struc_build/system.top"
    topology = Topology.from_gmx_topfile(name="test", path=toppath)
    topology.preprocess()
    topology.volumes = {"PEO":0.43}
    return NonBondEngine.from_topology(topology.molecules,
                                       topology,
                                       box=np.array([10., 10., 10.]))
@pytest.fixture
def molecule():
    toppath = TEST_DATA + "/struc_build/system.top"
    topology = Topology.from_gmx_topfile(name="test", path=toppath)
    return topology.molecules[0]

def add_positions(nb_matrix, ncoords, pos=None):
    if isinstance(pos, type(None)):
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
        if all(point != np.array([np.inf, np.inf, np.inf])):
           nb_matrix.add_positions(point, mol_idx=0, node_key=idx+1, start=False)
    return nb_matrix

def test_rewind(nonbond_matrix):
    nb_matrix = add_positions(nonbond_matrix, 6)
    processor = RandomWalk(mol_idx=0, nonbond_matrix=nb_matrix, nrewind=3)
    # node 4 is already placed and hence is skipped over
    processor.placed_nodes = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (5, 6)]
    last_idx = processor._rewind(current_step=5)
    assert last_idx == 3
    for idx in [6, 5, 3]:
        assert all(nb_matrix.positions[idx] == np.array([np.inf, np.inf, np.inf]))

@pytest.mark.parametrize('new_point, result', (
    (np.array([1., 1., 2.96]),
     False
     ),
    (np.array([1., 1., 2.3]),
     True
     )))
def test_is_overlap(nonbond_matrix, molecule, new_point, result):
    nb_matrix = add_positions(nonbond_matrix, 6)
    proccessor = RandomWalk(mol_idx=0, nonbond_matrix=nb_matrix)
    proccessor.molecule = molecule
    # node 4 is already placed and hence is skipped over
    assert proccessor._is_overlap(new_point, 7, nrexcl=1) == result

@pytest.mark.parametrize('new_point, restraint, result', (
   # distance restraint true upper_bound
   # ref_node, upper_bound, lower_bound
   (np.array([1., 1., 2.96]),
   [(0, 4.0, 0.0)],
    True
   ),
   #distance restraint false upper_bound
   (np.array([1., 1.0, 2.96]),
   [(0, 1.43, 0.0)],
    False
   ),
   # distance restraint false lower_bound
   (np.array([1., 1.0, 2.96]),
   [(5, 2.00, 1.0)],
    False
   ),
   # distance restraint true lower_bound
   (np.array([1., 1.0, 2.96]),
   [(5, 2.00, 0.47)],
    True
   ),
   # two restraints true
    (np.array([1., 1.0, 2.96]),
    [(5, 2.00, 0.47), (0, 4.0, 0.0)],
    True
    ),
   # two restraints 1 false
    (np.array([1., 1.0, 2.96]),
    [(5, 2.00, 1.0), (0, 4.0, 0.0)],
    False
    ),
   # two restraints 1 false
    (np.array([1., 1.0, 2.96]),
    [(5, 2.00, 0.47), (0, 1.43, 0.0)],
    False
    ),
))
def test_checks_milestone(nonbond_matrix, molecule, new_point, restraint, result):
   nb_matrix = add_positions(nonbond_matrix, 6)
   proccessor = RandomWalk(mol_idx=0, nonbond_matrix=nb_matrix)
   molecule.nodes[7]["distance_restraints"] = restraint
   proccessor.molecule = molecule
   assert proccessor.checks_milestones(7, new_point) == result

@pytest.mark.parametrize('pos, expected', (
    # simple test; should just work
    (np.array([[1.0, 1.0, 0.37],
               [1.0, 1.0, 0.74],
               [1.0, 1.0, 1.11],
               [1.0, 1.0, 1.48],
               [1.0, 1.0, 1.85],
               [1.0, 1.0, 2.22],
               [1.0, 1.0, 2.59]]),
     True),
    # this will fail because all space is blocked
    (np.array([[1.0, 1.0, 0.67],
               [1.0, 1.0, 1.37],
               [1.0, 1.37, 1.0],
               [1.37, 1.0, 1.0],
               [1.0, 0.63, 1.0],
               [0.63, 1.0, 1.0],
               [1.0, 1.0, 1.0]]),
     False
     )))
def test_update_positions(nonbond_matrix, molecule, pos, expected):
    # add positions of the rest of the chain
    nb_matrix = add_positions(nonbond_matrix, 7, pos=pos)
    # create instance of processor
    proccessor = RandomWalk(mol_idx=0,
                            nonbond_matrix=nb_matrix,
                            maxdim=np.array([10., 10., 10.]),
                            max_force=100.0,
                            maxiter=49)
    # set molecule attribute which is normally set by the run_molecule class
    proccessor.molecule = molecule
    vector_bundle = polyply.src.linalg_functions.norm_sphere(50)
    status = proccessor.update_positions(vector_bundle=vector_bundle, current_node=7, prev_node=6)
    assert status == expected
    if status:
       assert all(nb_matrix.positions[7] != np.array([np.inf, np.inf, np.inf]))
    else:
       assert all(nb_matrix.positions[7] == np.array([np.inf, np.inf, np.inf]))

@pytest.mark.parametrize('build_attr, pos, start, npos', (
    # simple test; create all coordinates
    ({0: True, 1: True, 2: True, 3: True,
      4: True, 5: True, 6: True, 7: True, 8: True, 9: True},
     None,
     None,
     0),
    # start in the middle of the chain
    ({0: True, 1: True, 2: True, 3: True,
      4: True, 5: True, 6: True, 7: True, 8: True, 9: True},
     None,
     5,
     0),
    # here we look for a starting point and build the rest
    ({0: False, 1: False, 2: True, 3: True,
      4: True, 5: True, 6: True, 7: True, 8: True, 9: True},
     np.array([[1.0, 1.0, 0.67],
               [1.0, 1.0, 1.37]]),
     None,
     2,
     ),
    # here we need to skip one already defined position
    ({0: False, 1: True, 2: False, 3: True,
      4: True, 5: True, 6: True, 7: True, 8: True, 9: True},
     np.array([[1.0, 1.0, 0.67],
               [np.inf, np.inf, np.inf],
               [1.0, 1.0, 1.30]]),
     None,
     3),
    # here we trigger a rewind
   # ({0: False, 1: False, 2: False, 3: False,
   #   4: False, 5: False, 6: False, 7: True, 8: True, 9: True},
   #  np.array([[1.0, 1.0, 0.67],
   #            [1.0, 1.0, 1.37],
   #            [1.0, 1.37, 1.0],
   #            [1.37, 1.0, 1.0],
   #            [1.0, 0.63, 1.0],
   #            [0.63, 1.0, 1.0],
   #            [1.0, 1.0, 1.0],
   #            ]),
   #  None,
   #  7),
     ))
def test_run_molecule(nonbond_matrix, molecule, build_attr, npos, pos, start):
    # add positions of the rest of the chain
    nb_matrix = add_positions(nonbond_matrix, npos, pos=pos)
    # create instance of processor
    vector_bundle = polyply.src.linalg_functions.norm_sphere(500)
    proccessor = RandomWalk(mol_idx=0,
                            nonbond_matrix=nb_matrix,
                            maxdim=np.array([10., 10., 10.]),
                            max_force=100.0,
                            maxiter=49,
                            vector_sphere=vector_bundle,
                            start_node=start)
    # set molecule attribute which is normally set by the run_molecule class
    nx.set_node_attributes(molecule, build_attr, "build")
    proccessor.run_molecule(molecule)
    for pos in proccessor.nonbond_matrix.positions:
        assert all(pos != np.array([np.inf, np.inf, np.inf]))


@pytest.mark.parametrize('point, old_point, node_dict, expected', (
    # basic example
    (np.array([1., 1., 2.]),
     np.array([1., 1., 1.]),
     {"rw_options": [[np.array([0., 0., 1.0]), 90.0]]},
     True),
    # false because goes back
    (np.array([1., 1., 0.5]),
     np.array([1., 1., 1.]),
     {"rw_options": [[np.array([0., 0., 1.0]), 90.0]]},
     False),
    # false because angle not large enough
    (np.array([1.5, 1.5, 1.5]),
     np.array([1., 1., 1.]),
     {"rw_options": [[np.array([0., 0., 1.0]), 50.0]]},
     False),
))
def test_vector_push(point, old_point, node_dict, expected):
    status = is_restricted(point, old_point, node_dict)
    assert status == expected
