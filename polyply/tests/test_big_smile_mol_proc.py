import pytest
import networkx as nx
from polyply.src.big_smile_mol_processor import (DefBigSmileParser,
                                                 generate_edge)

@pytest.mark.parametrize('bonds_source, bonds_target, edge, btypes',(
                        # single bond source each
                        ({0: "$"},
                         {3: "$"},
                         (0, 3),
                         ('$', '$')),
                        # multiple sources one match
                        ({0: '$1', 2: '$2'},
                         {1: '$2', 3: '$'},
                         (2, 1),
                         ('$2', '$2')),
                        # left right selective bonding
                        ({0: '$', 1: '>', 3: '<'},
                         {0: '>', 1: '$5'},
                         (3, 0),
                         ('<', '>')),
                        # left right selective bonding
                        # with identifier
                        ({0: '$', 1: '>', 3: '<1'},
                         {0: '>', 1: '$5', 2: '>1'},
                         (3, 2),
                         ('<1', '>1')),

))
def test_generate_edge(bonds_source, bonds_target, edge, btypes):
    source = nx.path_graph(5)
    target = nx.path_graph(4)
    nx.set_node_attributes(source, bonds_source, "bonding")
    nx.set_node_attributes(target, bonds_target, "bonding")
    new_edge, new_btypes = generate_edge(source, target, bond_type="bonding")
    assert new_edge == edge
    assert new_btypes == btypes
