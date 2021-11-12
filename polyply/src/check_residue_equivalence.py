import networkx as nx
from vermouth.molecule import attributes_match

def _match(node1, node2):
    """
    Helper function. Return true when atomtypes in
    node attribute dicts are the same.
    """
    ignore = [key for key in node2.keys() if key != "atype"]
    return attributes_match(node1, node2, ignore_keys=ignore)

def _are_subgraph_isomorphic(graph1, graph2):
    """
    Wrapper for isomoprhism check.
    """
    GM = nx.isomorphism.GraphMatcher(graph1,
                                     graph2,
                                     node_match=_match,)
    return GM.is_isomorphic()

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
        molecule = topology.molecules[mol_idxs[0]]
        for node in molecule.nodes:
            resname = molecule.nodes[node]["resname"]
            graph = molecule.nodes[node]["graph"]
            if resname in visited_residues:
                if not _are_subgraph_isomorphic(graph, visited_residues[resname]):
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
