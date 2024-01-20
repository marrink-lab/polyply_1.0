import pytest
import networkx as nx
from polyply.src.big_smile_parsing import (res_pattern_to_meta_mol,
                                           tokenize_big_smile)

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
                       ## simple branched with expansion
                       #("{[#PMA]([#PEO]|3)|2}",
                       #["PMA", "PEO", "PEO", "PEO",
                       # "PMA", "PEO", "PEO", "PEO"],
                       #[(0, 1), (1, 2), (2, 3),
                       # (0, 4), (4, 5), (5, 6), (6, 7)]
                       # )
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
