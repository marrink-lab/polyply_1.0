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

@pytest.mark.parametrize('extra_edges, termini, expect_ter', (
    # regular DNA strand
    (
    [],
    {0: "DC5", 5: "DG3"},
    {6: "DG3", 11: "DC5"}
    ),
    # circular DNA strand
    (
    [(5, 0)],
    {0: "DC", 5: "DG"},
    {6: "DG", 11: "DC"}
    )
    ))
def test_complement_dsDNA(extra_edges, termini, expect_ter):
    test_DNA_mol = MetaMolecule()
    nodes = range(0, 6)
    test_DNA_mol.add_edges_from(zip(nodes[:-1], nodes[1:]))
    test_DNA_mol.add_edges_from(extra_edges)
    resnames = {1: "DG", 2: "DT", 3: "DC", 4: "DA"}
    resnames.update(termini)
    nx.set_node_attributes(test_DNA_mol, resnames, "resname")
    nx.set_node_attributes(test_DNA_mol, dict(zip(nodes, range(1, 7))), "resid")
    complement_dsDNA(test_DNA_mol)
    new_resnames = nx.get_node_attributes(test_DNA_mol, "resname")
    resnames.update({7: "DC", 8: "DA", 9: "DG", 10: "DT"})
    resnames.update(expect_ter)
    assert new_resnames == resnames
    assert len(list(nx.connected_components(test_DNA_mol))) == 2
