import pytest
import networkx as nx
from polyply.src.check_residue_equivalence import group_residues_by_isomorphism
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
def test_group_by_isomorphism(example_meta_molecule, resnames, gen_template_graphs):
    # set the residue names
    for resname, node in zip(resnames, example_meta_molecule.nodes):
        example_meta_molecule.nodes[node]['resname'] = resname
        nx.set_node_attributes(example_meta_molecule.nodes[node]['graph'], resname, 'resname')

    # extract template graphs if needed
    template_graphs = {}
    for node in gen_template_graphs:
        graph = example_meta_molecule.nodes[node]['graph']
        nx.set_node_attributes(graph, True, 'template')
        template_graphs[example_meta_molecule.nodes[node]['resname']] = graph

    # perfrom the grouping
    unique_graphs = group_residues_by_isomorphism(example_meta_molecule, template_graphs)

    # check the outcome
    assert len(unique_graphs) == 2

    for graph in template_graphs.values():
        graph_hash = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(graph, node_attr='atomname')
        templated = list(nx.get_node_attributes(unique_graphs[graph_hash], 'template').values())
        assert all(templated)
