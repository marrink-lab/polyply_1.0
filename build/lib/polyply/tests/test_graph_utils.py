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
Test graph related functions
"""
import pytest
import networkx as nx
import polyply
from .example_fixtures import example_meta_molecule

@pytest.mark.parametrize('source, max_length, min_length, expected',(
                        (4, 1, 1, [4, 1, 9, 10]),
                        (4, 2, 1, [4, 1, 9, 10, 0, 3]),
                        (4, 3, 3, [0, 3, 7, 8, 2]),
                        (0, 1, 1, [0, 1, 2])
                        ))
def test_neighbourhood(source, max_length, min_length, expected):
    graph = nx.balanced_tree(r=2, h=3)
    neighbours = polyply.src.graph_utils.neighborhood(graph,
                                                      source,
                                                      max_length,
                                                      min_length=min_length)
    assert set(neighbours) == set(expected)

@pytest.mark.parametrize('edges, expected',(
                        # simple linear
                        ([(0, 1), (1, 2), (2, 3)], False),
                        # simple cyclic
                        ([(0, 1), (1, 2), (2, 3), (3, 0)], False),
                        # simple branched
                        ([(0, 1), (1, 2), (1, 3), (3, 4)], True),
                        # cyclic branched
                        ([(0, 1), (1, 2), (2, 3), (3, 0), (0, 5)], True),
                        # no nodes
                        ([], False)
                        ))
def test_is_branched(edges, expected):
    graph = nx.Graph()
    graph.add_edges_from(edges)
    result = polyply.src.graph_utils.is_branched(graph)
    assert result == expected

@pytest.mark.parametrize('nodes, expected',(
                        ((0, 1), [(1, 4)]),
                        # central residue
                        ((1, 2), [(6, 9)]),
                        ))
def test_find_connecting_edges(example_meta_molecule, nodes, expected):
    result =  polyply.src.graph_utils.find_connecting_edges(example_meta_molecule,
                                                            example_meta_molecule.molecule,
                                                            nodes)
    assert result == expected


@pytest.mark.parametrize('del_edge, expected',(
                        ({"u":1, "v":4}, [{"resA": "A", "idxA": 1, "resB": "B", "idxB": 2}]),
                        # central residue
                        ({"u":6, "v":9}, [{"resA": "B", "idxA": 2, "resB": "A", "idxB": 3}]),
                        ))
def test_find_missing_edges(example_meta_molecule, del_edge, expected):
    example_meta_molecule.molecule.remove_edge(**del_edge)
    for idx, missing in enumerate(polyply.src.graph_utils.find_missing_edges(example_meta_molecule,
                                                                             example_meta_molecule.molecule)):
        assert missing == expected[idx]
