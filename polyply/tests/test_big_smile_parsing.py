import pytest
import networkx as nx
from polyply.src.big_smile_parsing import (res_pattern_to_meta_mol,
                                           tokenize_big_smile,
                                           fragment_iter)

@pytest.mark.parametrize('smile, nodes, edges',(
                        # smiple linear seqeunce
                        ("{[#PMA][#PEO][#PMA]}",
                        ["PMA", "PEO", "PMA"],
                        [(0, 1), (1, 2)]),
                        # simple branched sequence
                        ("{[#PMA][#PMA]([#PEO][#PEO])[#PMA]}",
                        ["PMA", "PMA", "PEO", "PEO", "PMA"],
                        [(0, 1), (1, 2), (2, 3), (1, 4)]),
                        # simple sequence two branches
                        ("{[#PMA][#PMA][#PMA]([#PEO][#PEO])([#CH3])[#PMA]}",
                        ["PMA", "PMA", "PMA", "PEO", "PEO", "CH3", "PMA"],
                        [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (2, 6)]),
                        # simple linear sequence with expansion
                        ("{[#PMA]|3}",
                        ["PMA", "PMA", "PMA"],
                        [(0, 1), (1, 2)]),
                        # simple branch expension
                        ("{[#PMA]([#PEO][#PEO][#OHter])|2}",
                        ["PMA", "PEO", "PEO", "OHter",
                         "PMA", "PEO", "PEO", "OHter"],
                        [(0, 1), (1, 2), (2, 3),
                         (0, 4), (4, 5), (5, 6), (6, 7)]
                         ),
                        # nested branched with expansion
                        ("{[#PMA]([#PEO]|3)|2}",
                        ["PMA", "PEO", "PEO", "PEO",
                         "PMA", "PEO", "PEO", "PEO"],
                        [(0, 1), (1, 2), (2, 3),
                         (0, 4), (4, 5), (5, 6), (6, 7)]
                         )
))
def test_res_pattern_to_meta_mol(smile, nodes, edges):
    """
    Test that the meta-molecule is correctly reproduced
    from the simplified smile string syntax.
    """
    meta_mol = res_pattern_to_meta_mol(smile)
    assert len(meta_mol.edges) == len(edges)
    for edge in edges:
        assert meta_mol.has_edge(*edge)
    resnames = nx.get_node_attributes(meta_mol, 'resname')
    assert nodes == list(resnames.values())

@pytest.mark.parametrize('big_smile, smile, bonding',(
                        # smiple symmetric bonding
                        ("[$]COC[$]",
                         "COC",
                        {0: '$', 2: '$'}),
                        # named different bonding descriptors
                        ("[$1]CCCC[$2]",
                         "CCCC",
                        {0: "$1", 3: "$2"}),
                        # ring and bonding descriptors
                        ("[$1]CC[$2]C1CCCCC1",
                         "CCC1CCCCC1",
                        {0: "$1", 1: "$2"}),
                        # bonding descript. after branch
                        ("C(COC[$1])[$2]CCC[$3]",
                         "C(COC)CCC",
                        {0: '$2', 3: '$1', 6: '$3'}),
                        # left rigth bonding desciptors
                        ("[>]COC[<]",
                        "COC",
                        {0: '>', 2: '<'})
))
def test_tokenize_big_smile(big_smile, smile, bonding):
    new_smile, new_bonding = tokenize_big_smile(big_smile)
    assert new_smile == smile
    assert new_bonding == bonding

@pytest.mark.parametrize('fragment_str, nodes, edges',(
                        # single fragment
                        ("{#PEO=[$]COC[$]}",
                        {"PEO": ((0, {"atomname": "C0", "resname": "PEO", "bonding": "$", "element": "C"}),
                                 (1, {"atomname": "O1", "resname": "PEO", "element": "O"}),
                                 (2, {"atomname": "C2", "resname": "PEO", "bonding": "$", "element": "C"}),
                                 (3, {"atomname": "H3", "resname": "PEO", "element": "H"}),
                                 (4, {"atomname": "H4", "resname": "PEO", "element": "H"}),
                                 (5, {"atomname": "H5", "resname": "PEO", "element": "H"}),
                                 (6, {"atomname": "H6", "resname": "PEO", "element": "H"}),
                                )},
                        {"PEO": [(0, 1), (1, 2), (0, 3), (0, 4), (2, 5), (2, 6)]}),
                        # test NH3 terminal
                        ("{#AMM=N[$]}",
                        {"AMM": ((0, {"atomname": "N0", "resname": "AMM", "bonding": "$", "element": "N"}),
                                 (1, {"atomname": "H1", "resname": "AMM", "element": "H"}),
                                 (2, {"atomname": "H2", "resname": "AMM", "element": "H"}),
                                )},
                        {"AMM": [(0, 1), (0, 2)]}),
                        # single fragment + 1 terminal (i.e. only 1 bonding descrpt
                        ("{#PEO=[$]COC[$],#OHter=[$][OH]}",
                        {"PEO": ((0, {"atomname": "C0", "resname": "PEO", "bonding": "$", "element": "C"}),
                                 (1, {"atomname": "O1", "resname": "PEO", "element": "O"}),
                                 (2, {"atomname": "C2", "resname": "PEO", "bonding": "$", "element": "C"}),
                                 (3, {"atomname": "H3", "resname": "PEO", "element": "H"}),
                                 (4, {"atomname": "H4", "resname": "PEO", "element": "H"}),
                                 (5, {"atomname": "H5", "resname": "PEO", "element": "H"}),
                                 (6, {"atomname": "H6", "resname": "PEO", "element": "H"}),
                                 ),
                         "OHter": ((0, {"atomname": "O0", "resname": "OHter", "bonding": "$", "element": "O"}),
                                   (1, {"atomname": "H1", "resname": "OHter", "element": "H"}))},
                        {"PEO": [(0, 1), (1, 2), (0, 3), (0, 4), (2, 5), (2, 6)],
                         "OHter": [(0, 1)]}),
))
def test_fragment_iter(fragment_str, nodes, edges):
    for resname, mol_graph in fragment_iter(fragment_str):
        assert len(mol_graph.nodes) == len(nodes[resname])
        for node, ref_node in zip(mol_graph.nodes(data=True), nodes[resname]):
           assert node[0] == ref_node[0]
           for key in ref_node[1]:
                assert ref_node[1][key] == node[1][key]
        assert sorted(mol_graph.edges) == sorted(edges[resname])
