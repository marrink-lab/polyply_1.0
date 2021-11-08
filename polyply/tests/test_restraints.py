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
Test that distance and position restraints are tagged.
"""
import pytest
import numpy as np
import networkx as nx
from hypothesis import given
from hypothesis_networkx import graph_builder
from hypothesis import strategies as st
import polyply
from polyply.tests.test_build_file_parser import test_molecule
from polyply.src.meta_molecule import MetaMolecule

@pytest.mark.parametrize('target_node, ref_node, distance, avg_step_length, tolerance, expected',(
    # test simple revers
   (0, 4, 1.5, 0.47, 0.0,
    {1: [(0, 2.91, 0.375)],
     2: [(0, 2.44, 0.75)],
     3: [(0, 1.97, 1.125)],
     4: [(0, 1.97, 1.5)],}
   ),
    # test simple case
   (4, 0, 1.5, 0.47, 0.0,
    {1: [(0, 2.91, 0.375)],
     2: [(0, 2.44, 0.75)],
     3: [(0, 1.97, 1.125)],
     4: [(0, 1.97, 1.5)],
    }
   ),
   (4, 0, 1.5, 0.47, 0.3,
    {1: [(0, 3.21, 0.075)],
     2: [(0, 2.74, 0.45)],
     3: [(0, 2.27, 0.825)],
     4: [(0, 2.27, 1.2)],
    }
   ),
   ))
def test_set_distance_restraint(test_molecule,
                                target_node,
                                ref_node,
                                distance,
                                avg_step_length,
                                tolerance,
                                expected):

    polyply.src.restraints.set_distance_restraint(test_molecule,
                                                  target_node,
                                                  ref_node,
                                                  distance,
                                                  avg_step_length,
                                                  tolerance)

    attr_list = nx.get_node_attributes(test_molecule, "distance_restraints")
    for node, restr_list in attr_list.items():
        ref_list = expected[node]
        for ref_restr, new_restr in zip(ref_list, restr_list):
            assert ref_restr[0] == new_restr[0]
            assert pytest.approx(ref_restr[1], new_restr[1])
            assert pytest.approx(ref_restr[2], new_restr[2])




builder = graph_builder(graph_type=nx.Graph,
                        node_keys=st.integers(),
                        min_nodes=4, max_nodes=10,
                        min_edges=3, max_edges=None,
                        self_loops=True,
                        connected=True)
@given(graph=builder)
def test_restraints_on_abitr_topology(graph):
    test_molecule = MetaMolecule()
    test_molecule.add_nodes_from(graph.nodes)
    test_molecule.add_edges_from(graph.edges)

    target_node = list(graph.nodes)[0]
    ref_node = list(graph.nodes)[1]
    distance = 4
    avg_step_length = 0.47
    tolerance = 0
    polyply.src.restraints.set_distance_restraint(test_molecule,
                                                  target_node,
                                                  ref_node,
                                                  distance,
                                                  avg_step_length,
                                                  tolerance)
    assert len(nx.get_node_attributes(test_molecule, "distance_restraints")) != 0
