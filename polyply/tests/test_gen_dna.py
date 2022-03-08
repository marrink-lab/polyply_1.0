# Copyright 2022 University of Groningen
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
Test DNA related functions.
"""
import pytest
import networkx as nx
from polyply.src.meta_molecule import MetaMolecule
from polyply.src.gen_dna import complement_dsDNA

def test_complement_dsDNA():
    test_DNA_mol = MetaMolecule()
    nodes = range(0, 5)
    test_DNA_mol.add_edges_from(zip(nodes[:-1], nodes[1:]))
    resnames = {0: "DC5", 1: "DG", 2: "DT", 3: "DC", 4: "DA", 5: "DG3"}
    nx.set_node_attributes(test_DNA_mol, resnames, "resname")
    nx.set_node_attributes(test_DNA_mol, dict(zip(nodes, range(1, 6))), "resid")
    complement_dsDNA(test_DNA_mol)
    new_resnames = nx.get_node_attributes(test_DNA_mol, "resname")
    resnames.update({6: "DG5", 7: "DC", 2: "DA", 8: "DG", 9: "DT", 10: "DC3"})
    assert new_resnames == resnames
    assert len(list(nx.connected_components(test_DNA_mol))) == 2
