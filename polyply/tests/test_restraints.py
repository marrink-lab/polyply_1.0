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
import polyply
from polyply.tests.test_build_file_parser import test_molecule

@pytest.mark.parametrize('target_node, ref_node, distance, avg_step_length, expected',(
   (0, 4, 1.5, 0.47,
    {0: [(4, 1.97, 1.5)],
     1: [(4, 1.97, 1.125)],
     2: [(4, 2.44, 0.75)],
     3: [(4, 2.91, 0.375)],
     4: [(4, 3.38, 0.0)],}
   ),
   (4, 0, 1.5, 0.47,
    {0: [(0, 3.38, 0.375)],
     1: [(0, 2.91, 0.5)],
     2: [(0, 2.34, 1.5)],
     3: [(0, 1.97, 4.5)],
     4: [(0, 1.97, 6.0)],
    }
   ),
   ))
def test_set_distance_restraint(test_molecule, target_node, ref_node, distance, avg_step_length, expected):

    path = nx.algorithms.shortest_path(test_molecule,
                                       source=target_node,
                                       target=ref_node)

    polyply.src.restraints.set_distance_restraint(test_molecule,
                                                  target_node,
                                                  ref_node,
                                                  distance,
                                                  avg_step_length,
                                                  path)

    attr_list = nx.get_node_attributes(test_molecule, "restraint")
    for node, restr_list in attr_list.items():
        ref_list = expected[node]
        for ref_restr, new_restr in zip(ref_list, restr_list):
            assert pytest.approx(ref_restr, new_restr)


