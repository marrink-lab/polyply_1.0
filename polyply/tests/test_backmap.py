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

import textwrap
import pytest
import numpy as np
from numpy.linalg import norm
import math
import networkx as nx
import vermouth
import polyply
from polyply import MetaMolecule
from polyply.src.backmap import Backmap


class TestBackmap():

    @staticmethod
    def test_backmapping():
        meta_molecule = MetaMolecule()
        meta_molecule.add_edges_from([(0, 1)])
        nx.set_node_attributes(meta_molecule, {0: {"resname": "test", "position": np.array([0, 0, 0])},
                                               1: {"resname": "test", "position": np.array([0, 0, 1.0])}})
        # test if disordered template works
        meta_molecule.templates = {"test": {2: np.array([0, 0, 0]),
                                            1: np.array([0, 0, 0.5]),
                                            3: np.array([0, 0.5, 0])}}
        meta_molecule.molecule = vermouth.molecule.Molecule()
        meta_molecule.molecule.add_edges_from(
            [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
        nx.set_node_attributes(meta_molecule.molecule, {
            1: {"resname": "test", "resid": 1, "build": True},
            2: {"resname": "test", "resid": 1, "build": True},
            3: {"resname": "test", "resid": 1, "build": True},
            4: {"resname": "test", "resid": 2, "build": True},
            5: {"resname": "test", "resid": 2, "build": True},
            6: {"resname": "test", "resid": 2, "build": False,
                                               "position": np.array([4., 4., 4.])}})

        Backmap().run_molecule(meta_molecule)
        for node in meta_molecule.molecule.nodes:
            assert "position" in meta_molecule.molecule.nodes[node]

        assert norm(
            meta_molecule.molecule.nodes[6]["position"]-np.array([4., 4., 4.])) == 0
