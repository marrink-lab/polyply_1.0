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
    {6: "DC5", 11: "DG3"}
    ),
    # circular DNA strand
    (
    [(5, 0)],
    {0: "DC", 5: "DG"},
    {6: "DC", 11: "DG"}
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
    edge_attr = dict(zip(zip(nodes[:-1], nodes[1:]), ["A", "B", "C", "D"]))
    nx.set_edge_attributes(test_DNA_mol, edge_attr, "test")

    complement_dsDNA(test_DNA_mol)

    resnames.update({7: "DT", 8: "DG", 9: "DA", 10: "DC"})
    resnames.update(expect_ter)
    new_edge_attrs = {(7, 8): "D", (8, 9): "C", (9, 10): "B", (10, 11): "A"}
    edge_attr.update(new_edge_attrs)

    new_resnames = nx.get_node_attributes(test_DNA_mol, "resname")
    assert new_resnames == resnames

    new_edge_attrs = nx.get_edge_attributes(test_DNA_mol, "test")
    for idx, jdx in new_edge_attrs:
        if (idx, jdx) in edge_attr:
            assert new_edge_attrs[(idx, jdx)] == edge_attr[(idx, jdx)]
        else:
            assert new_edge_attrs[(idx, jdx)] == edge_attr[(jdx, idx)]
    assert len(list(nx.connected_components(test_DNA_mol))) == 2


def test_complement_dsDNA_error():
    """
    Test that an unkown residue generates an error when
    passed to gen_dna processor.
    """
    test_DNA_mol = MetaMolecule()
    nodes = range(0, 6)
    test_DNA_mol.add_edges_from(zip(nodes[:-1], nodes[1:]))
    resnames = {0: "DA", 1: "DG", 2: "DT", 3: "DC", 4: "XX", 5: "DT"}
    nx.set_node_attributes(test_DNA_mol, resnames, "resname")
    nx.set_node_attributes(test_DNA_mol, dict(zip(nodes, range(1, 7))), "resid")
    edge_attr = dict(zip(zip(nodes[:-1], nodes[1:]), ["A", "B", "C", "D"]))
    nx.set_edge_attributes(test_DNA_mol, edge_attr, "test")

    with pytest.raises(IOError) as context:
        complement_dsDNA(test_DNA_mol)

    msg = ("Trying to complete a dsDNA strand. However, resname XX with resid 5 "
           "does not match any of the known base-pair resnames. Note that polyply "
           "at the moment only allows base-pair completion for molecules that only "
           "consist of dsDNA. Please conact the developers if you whish to create a "
           "more complicated molecule.")
    assert str(context.value) == msg
