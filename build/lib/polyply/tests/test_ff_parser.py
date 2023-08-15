import textwrap
import pytest
import numpy as np
import vermouth.forcefield
import vermouth.molecule
import polyply.src.ff_parser_sub

@pytest.mark.parametrize(
    'interaction_lines, edges, edge_attr', (
        (  # the edge section creates edges
            """
            [ edges ]
            SC1 SC2
            BB SC3
            """,
            (('SC1', 'SC2'), ('BB', 'SC3')),
            ({}, {}, {}),
        ),
        (  # the edge section creates edges and
           # can set edge attributes
            """
            [ edges ]
            SC1 SC2
            BB SC3 {"attr": "edge"}
            """,
            (('SC1', 'SC2'), ('BB', 'SC3')),
            ({}, {"attr": "edge"}),
        ),
    )
)
def test_interaction_edges(interaction_lines, edges, edge_attr):
    """
    Edges are created where expected.
    """
    lines = """
    [ moleculetype ]
    GLY 1
    [ atoms ]
    1 P4 1 ALA BB 1
    2 P3 1 ALA SC1 2
    3 P2 1 ALA SC2 3
    4 P2 1 ALA SC3 3
    """
    lines = textwrap.dedent(lines) + textwrap.dedent(interaction_lines)
    lines = lines.splitlines()

    ff = vermouth.forcefield.ForceField(name='test_ff')
    polyply.src.ff_parser_sub.read_ff(lines, ff)
    block = ff.blocks['GLY']
    for edge, attr in zip(edges, edge_attr):
        assert block.has_edge(edge[0], edge[1]) or block.has_edge(edge[1], edge[0])
        assert attr == block.edges[edge]

def test_interaction_edges_error():
    """
    Make sure an error is raised when someone
    tries to set node attributes in edges.
    """
    lines = """
    [ moleculetype ]
    GLY 1
    [ atoms ]
    1 P4 1 ALA BB 1
    2 P3 1 ALA SC1 2
    3 P2 1 ALA SC2 3
    4 P2 1 ALA SC3 3
    [ edges ]
    BB SC1
    SC1 {"some_attr": "some_value"} SC2
    """
    lines = textwrap.dedent(lines)
    lines = lines.splitlines()

    ff = vermouth.forcefield.ForceField(name='test_ff')
    with pytest.raises(IOError):
        polyply.src.ff_parser_sub.read_ff(lines, ff)
