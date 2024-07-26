import textwrap
import pytest
import networkx as nx
from vermouth.forcefield import ForceField
import polyply
from polyply.src.topology import Topology
from polyply.src.top_parser import read_topology
from polyply.src.check_residue_equivalence import group_residues_by_hash
from .example_fixtures import example_meta_molecule

@pytest.mark.parametrize('resnames, gen_template_graphs', (
                        # two different residues no template_graphs
                        (['A', 'B', 'A'], []),
                        # two different residues one template_graphs
                        (['A', 'B', 'A'], [1]),
                        # all residues with same name but not equivalent
                        (['A', 'A', 'A'], []),
                        # all different residues two template_graphs
                        (['A', 'B', 'A'], [0, 1]),
))
def test_group_by_hash(example_meta_molecule, resnames, gen_template_graphs):
    # set the residue names
    for resname, node in zip(resnames, example_meta_molecule.nodes):
        example_meta_molecule.nodes[node]['resname'] = resname
        nx.set_node_attributes(example_meta_molecule.nodes[node]['graph'], resname, 'resname')

    # extract template graphs if needed
    template_graphs = {}
    for node in gen_template_graphs:
        graph = example_meta_molecule.nodes[node]['graph']
        graph_hash = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(graph, node_attr='atomname')
        nx.set_node_attributes(graph, True, 'template')
        template_graphs[graph_hash] = graph

    # perfrom the grouping
    unique_graphs = group_residues_by_hash(example_meta_molecule, template_graphs)

    # check the outcome
    assert len(unique_graphs) == 2

    for graph_hash in template_graphs:
        templated = list(nx.get_node_attributes(unique_graphs[graph_hash], 'template').values())
        assert all(templated)

@pytest.mark.parametrize('top_lines', (
    # residue in two different molecules not equivalent by
    # number of atoms
    ("""
    [ defaults ]
    1   1   no   1.0     1.0
    [ atomtypes ]
    N0 72.0 0.000 A 0.0 0.0
    N1 72.0 0.000 A 0.0 0.0
    [ nonbond_params ]
    N0   N0   1 4.700000e-01    3.700000e+00
    N1   N1   1 4.700000e-01    3.700000e+00
    [ moleculetype ]
    testA 1
    [ atoms ]
    1    N0   1   GLY    BB   1 0.00     45
    2    N0   1   GLY    SC1  1 0.00     45
    3    N0   1   GLY    SC2  1 0.00     45
    4    N1   2   GLU    BB   2 0.00     45
    5    N1   2   GLU    SC1  2 0.00     45
    6    N1   2   GLU    SC2  2 0.00     45
    [ bonds ]
    1     2    1  0.47 2000
    2     3    1  0.47 2000
    3     4    1  0.47 2000
    4     5    1  0.47 2000
    5     6    1  0.47 2000
    [ moleculetype ]
    testB 1
    [ atoms ]
    1    N0   1   ASP    BB   1 0.00     45
    2    N0   1   ASP    SC3  1 0.00     45
    3    N0   1   ASP    SC4  1 0.00     45
    4    N1   2   GLU    BB   2 0.00     45
    5    N1   2   GLU    SC1  2 0.00     45
    [ bonds ]
    1     2    1  0.47 2000
    2     3    1  0.47 2000
    3     4    1  0.47 2000
    4     5    1  0.47 2000
    [ system ]
    test system
    [ molecules ]
    testA 1
    testB 1
    """),
    # residue in two different molecules not equivalent by
    # connectivity
    ("""
    [ defaults ]
    1   1   no   1.0     1.0
    [ atomtypes ]
    N0 72.0 0.000 A 0.0 0.0
    N1 72.0 0.000 A 0.0 0.0
    [ nonbond_params ]
    N0   N0   1 4.700000e-01    3.700000e+00
    N1   N1   1 4.700000e-01    3.700000e+00
    [ moleculetype ]
    testA 1
    [ atoms ]
    1    N0   1   GLY    BB   1 0.00     45
    2    N0   1   GLY    SC1  1 0.00     45
    3    N0   1   GLY    SC2  1 0.00     45
    4    N1   2   GLU    BB   2 0.00     45
    5    N1   2   GLU    SC1  2 0.00     45
    6    N1   2   GLU    SC2  2 0.00     45
    7    N1   2   GLU    SC3  2 0.00     45
    [ bonds ]
    1     2    1  0.47 2000
    2     3    1  0.47 2000
    3     4    1  0.47 2000
    4     5    1  0.47 2000
    5     6    1  0.47 2000
    6     7    1  0.47 2000
    [ moleculetype ]
    testB 1
    [ atoms ]
    1    N0   1   ASP    BB   1 0.00     45
    2    N0   1   ASP    SC3  1 0.00     45
    3    N0   1   ASP    SC4  1 0.00     45
    4    N1   2   GLU    BB   2 0.00     45
    5    N1   2   GLU    SC1  2 0.00     45
    6    N1   2   GLU    SC2  2 0.00     45
    7    N1   2   GLU    SC3  2 0.00     45
    [ bonds ]
    1     2    1  0.47 2000
    2     3    1  0.47 2000
    3     4    1  0.47 2000
    4     5    1  0.47 2000
    5     6    1  0.47 2000
    5     7    1  0.47 2000
    [ system ]
    test system
    [ molecules ]
    testA 1
    testB 1
    """)))
def test_raise_residue_error(top_lines):
    lines = textwrap.dedent(top_lines)
    lines = lines.splitlines()
    force_field = ForceField("test")
    topology = Topology(force_field)
    read_topology(lines=lines, topology=topology, cwdir="./")
    topology.preprocess()
    topology.volumes = {"GLY": 0.53, "GLU": 0.67, "ASP": 0.43}
    with pytest.raises(IOError):
        polyply.src.check_residue_equivalence.check_residue_equivalence(topology)

def test_raise_residue_no_error():
    """
    This test makes sure that we actually skip molecules that are defined
    in the top file but not used in the actual system.
    """
    top_lines="""
    [ defaults ]
    1   1   no   1.0     1.0
    [ atomtypes ]
    N0 72.0 0.000 A 0.0 0.0
    N1 72.0 0.000 A 0.0 0.0
    [ nonbond_params ]
    N0   N0   1 4.700000e-01    3.700000e+00
    N1   N1   1 4.700000e-01    3.700000e+00
    [ moleculetype ]
    testA 1
    [ atoms ]
    1    N0   1   GLY    BB   1 0.00     45
    2    N0   1   GLY    SC1  1 0.00     45
    3    N0   1   GLY    SC2  1 0.00     45
    4    N1   2   GLU    BB   2 0.00     45
    5    N1   2   GLU    SC1  2 0.00     45
    6    N1   2   GLU    SC2  2 0.00     45
    7    N1   2   GLU    SC3  2 0.00     45
    [ bonds ]
    1     2    1  0.47 2000
    2     3    1  0.47 2000
    3     4    1  0.47 2000
    4     5    1  0.47 2000
    5     6    1  0.47 2000
    6     7    1  0.47 2000
    [ moleculetype ]
    testB 1
    [ atoms ]
    1    N0   1   ASP    BB   1 0.00     45
    2    N0   1   ASP    SC3  1 0.00     45
    3    N0   1   ASP    SC4  1 0.00     45
    4    N1   2   GLU    BB   2 0.00     45
    5    N1   2   GLU    SC1  2 0.00     45
    6    N1   2   GLU    SC2  2 0.00     45
    7    N1   2   GLU    SC3  2 0.00     45
    [ bonds ]
    1     2    1  0.47 2000
    2     3    1  0.47 2000
    3     4    1  0.47 2000
    4     5    1  0.47 2000
    5     6    1  0.47 2000
    5     7    1  0.47 2000
    [ system ]
    test system
    [ molecules ]
    testA 1
    """
    lines = textwrap.dedent(top_lines)
    lines = lines.splitlines()
    force_field = ForceField("test")
    topology = Topology(force_field)
    read_topology(lines=lines, topology=topology, cwdir="./")
    topology.preprocess()
    topology.volumes = {"GLY": 0.53, "GLU": 0.67, "ASP": 0.43}
    polyply.src.check_residue_equivalence.check_residue_equivalence(topology)
