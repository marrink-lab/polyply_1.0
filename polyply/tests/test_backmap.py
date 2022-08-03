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
Test backmapping
"""
import numpy as np
from numpy.linalg import norm
import networkx as nx
import vermouth
from vermouth.graph_utils import make_residue_graph
from polyply import MetaMolecule
from polyply.src.backmap import Backmap

def test_backmapping():
    """
    Integral test for the backmapping processor.
    """
    molecule = vermouth.molecule.Molecule()
    molecule.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    nx.set_node_attributes(molecule, {
        1: {"resname": "test", "resid": 1, "atomname": "A"},
        2: {"resname": "test", "resid": 1, "atomname": "B"},
        3: {"resname": "test", "resid": 1, "atomname": "C"},
        4: {"resname": "test", "resid": 2, "atomname": "A"},
        5: {"resname": "test", "resid": 2, "atomname": "B"},
        6: {"resname": "test", "resid": 2, "atomname": "C"},
        7: {"resname": "test", "resid": 3, "atomname": "C",
            "position": np.array([4., 4., 4.])}
        })

    graph = make_residue_graph(molecule)
    meta_molecule = MetaMolecule(graph)
    meta_molecule.molecule = molecule

    nx.set_node_attributes(meta_molecule, {0: {"resname": "test",
                                               "position": np.array([0, 0, 0]),
                                               "resid": 1, "backmap": True},
                                           1: {"resname": "test",
                                               "position": np.array([0, 0, 1.0]),
                                               "resid": 2, "backmap": True},
                                           2: {"resname": "test",
                                               "position": np.array([0, 0, 2.0]),
                                               "resid": 3, "backmap": False}})
    # test if disordered template works
    meta_molecule.templates = {"test": {"B": np.array([0, 0, 0]),
                                        "A": np.array([0, 0, 0.5]),
                                        "C": np.array([0, 0.5, 0])}}

    Backmap().run_molecule(meta_molecule)
    for node in meta_molecule.molecule.nodes:
        assert "position" in meta_molecule.molecule.nodes[node]
    assert norm(meta_molecule.molecule.nodes[7]["position"] - np.array([4., 4., 4.])) == 0
