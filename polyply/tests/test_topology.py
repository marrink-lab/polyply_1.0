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

import textwrap
import pytest
import numpy as np
import networkx as nx
import vermouth.forcefield
import vermouth.molecule
import polyply.src.meta_molecule
from polyply.src.meta_molecule import (MetaMolecule, Monomer)
from polyply.src.topology import Topology

class TestTopology:

    @staticmethod
    def test_from_gmx_topfile():
        top = Topology.from_gmx_topfile("test_data/topology_test/system.top", "test")
        assert len(top.molecules) == 1

    @staticmethod
    def test_add_positions_from_gro():
        top = Topology.from_gmx_topfile("test_data/topology_test/system.top", "test")
        top.add_positions_from_gro("test_data/topology_test/test.gro")
        for node in top.molecules[0].molecule.nodes:
            if node != 6:
               assert "position" in top.molecules[0].molecule.nodes[node].keys()
               assert top.molecules[0].molecule.nodes[node]["build"] == False
            else:
               assert top.molecules[0].molecule.nodes[node]["build"] == True

    @staticmethod
    def test_convert_to_vermouth_system():
        top = Topology.from_gmx_topfile("test_data/topology_test/system.top", "test")
        system = top.convert_to_vermouth_system()
        assert isinstance(system, vermouth.system.System)
        assert len(system.molecules) == 1
