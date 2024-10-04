import networkx as nx
from vermouth.molecule import attributes_match
from collections import defaultdict

def check_residue_equivalence(topology):
    """
    Check that each residue in all moleculetypes
    is unique.

    Parameters
    ----------
    topology: :class:`polyply.src.topology.Topology`

    Raises
    ------
    IOError
        raise a single error the first time two non-equivalent
        residues have been found
    """
    visited_residues = {}
    resids = {}
    molnames = {}
    for mol_name, mol_idxs in topology.mol_idx_by_name.items():
        # molecules that are included but not used in the topology
        # simply don't get mol_idxs but we need to skip them here
        if not mol_idxs:
            continue
        molecule = topology.molecules[mol_idxs[0]]
        for node in molecule.nodes:
            resname = molecule.nodes[node]["resname"]
            graph = molecule.nodes[node]["graph"]
            if resname in visited_residues:
                if not nx.is_isomorphic(graph, visited_residues[resname]):
                    msg = ("Residue {resname} with resid {residA} in moleculetype {molnameA} is\n"
                           "different from residue {resname} with resid {residB} in moleculetype\n"
                           "{molnameB}. All residues have to be unique in polyply.\n")
                    raise IOError(msg.format(**{"resname": resname,
                                                "molnameB": mol_name,
                                                "molnameA": molnames[resname],
                                                "residA": molecule.nodes[node]["resid"],
                                                "residB": resids[resname]}))
            else:
                visited_residues[resname] = graph
                molnames[resname] = mol_name
                resids[resname] = molecule.nodes[node]["resid"]

def group_residues_by_hash(meta_molecule, template_graphs={}):
    """
    Collect all unique residue graphs using the Weisfeiler-Lehman has.
    A dict of unique graphs with the hash as key is returned. The
    `meta_molecule` nodes are annotated with the hash using the template
    keyword. If required template_graphs can be given that are used for
    matching rather than the first founds residue.

    Parameters
    ----------
    meta_molecule: `:class:polyply.meta_molecule.MetaMolecule`
    template_graphs: dict[`:class:nx.Graph`]

    Returns
    -------
    dict[`:class:nx.Graph`]
        keys are the hash of the graph
    """
    unique_graphs = template_graphs
    for node in meta_molecule.nodes:
        graph = meta_molecule.nodes[node]["graph"]
        graph_hash = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(graph, node_attr='atomname')
        if graph_hash not in unique_graphs:
            unique_graphs[graph_hash] = graph
        meta_molecule.nodes[node]["template"] = graph_hash

    return unique_graphs
