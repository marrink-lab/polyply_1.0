import networkx as nx
from vermouth.molecule import attributes_match

def _atoms_match(node1, node2):
    return node1["atomname"] == node2["atomname"]

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

def group_residues_by_isomorphism(meta_molecule, template_graphs={}):
    """
    Collect all unique residue graphs. If the same resname matches
    multiple graphs the resname is appended by a number. If required
    template_graphs can be given that are used for matching rather
    than the first founds residue.
    """
    unique_graphs = template_graphs
    for node in meta_molecule.nodes:
        resname = meta_molecule.nodes[node]["resname"]
        graph = meta_molecule.nodes[node]["graph"]
        if resname in unique_graphs and not nx.is_isomorphic(graph,
                                                             template_graphs[resname],
                                                             node_match=_atoms_match,):
            template_name = resname + str(len(template_graphs))
            meta_molecule.nodes[node]["template"] = template_name
            unique_graphs[template_name] = graph
        else:
            meta_molecule.nodes[node]["template"] = resname
            unique_graphs[resname] = graph
    return template_graphs
