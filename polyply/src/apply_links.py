import networkx as nx
from networkx.algorithms import isomorphism
import vermouth.molecule
from polyply.src.processor import Processor

class MatchError(Exception):
    """Raised we find no match between links and molecule"""

def find_atoms(molecule, **attrs):
    """
    Yields all indices of atoms that match `attrs`

    Parameters
    ----------
    **attrs: collections.abc.Mapping
        The attributes and their desired values.

    Yields
    ------
    collections.abc.Hashable
        All atom indices that match the specified `attrs`
    """
    try:
        ignore = attrs['ignore']
        del attrs['ignore']
    except KeyError:
        ignore = []

    for node_idx in molecule:
        node = molecule.nodes[node_idx]
        if vermouth.molecule.attributes_match(node, attrs, ignore_keys=ignore):
            yield node_idx

def _build_link_interaction_from(molecule, interaction, match):
    atoms = tuple(match[idx] for idx in interaction.atoms)
    parameters = [
        param(molecule, match) if callable(param) else param
        for param in interaction.parameters
    ]
    new_interaction = interaction._replace(
        atoms=atoms,
        parameters=parameters
    )
    return new_interaction

def apply_link_between_residues(molecule, link, resids):
    """
    Applies a link between specific residues, if and only if
    the link atoms incl. all attributes match at most one atom
    in a respective link.
    Parameters
    ----------
    link: :class:`vermouth.molecule.Link`
        A vermouth link definition
    resids: dictionary
        a list of node attributes used for link matching aside from
        the residue ordering
    """
    # we have to go on resid or at least one criterion otherwise
    # the matching will be super slow, if we need to iterate
    # over all combintions of a possible links.
    #print("resids", resids)
    nx.set_node_attributes(link, dict(zip(link.nodes, resids)), 'resid')
    #print("resids", resids)
    link_to_mol = {}
    for node in link.nodes:
        attrs = link.nodes[node]
        attrs.update({'ignore':['order', 'charge_group']})
        #print(link.interactions)
        matchs = [atom for atom in find_atoms(molecule, **attrs)]
        #print("match", matchs)
        if len(matchs) == 1:
            link_to_mol[node] = matchs[0]
        else:
            msg = "Found no matchs for atom {} in resiue {}. Cannot apply link."
            raise MatchError(msg.format(attrs["atomname"], attrs["resid"]))

    for inter_type in link.interactions:
        for interaction in link.interactions[inter_type]:
            new_interaction = _build_link_interaction_from(molecule, interaction, link_to_mol)
            molecule.add_or_replace_interaction(inter_type, *new_interaction)
            atoms = interaction.atoms
            new_edges = [(link_to_mol[atoms[i]], link_to_mol[atoms[i+1]]) for i in
                         range(0, len(atoms)-1)]
            molecule.add_edges_from(new_edges)

def neighborhood(graph, node, degree):
    # Adobted from: https://stackoverflow.com/questions/
    #22742754/finding-the-n-degree-neighborhood-of-a-node

    path_lengths = nx.single_source_dijkstra_path_length(graph, node)
    neighbours = [node for node, length in path_lengths.items() if length <= degree]
    return neighbours

def get_subgraphs(meta_molecule, orders, edge):
    sub_graph_idxs = [[]]
    zero_idx = list(orders).index(0)

    for idx, order in enumerate(orders):
        if isinstance(order, int):
            resid = edge[0] + order
            for idx_set in sub_graph_idxs:
                idx_set.append(resid)
        else:
            res_ids = neighborhood(meta_molecule, edge[0], idx-zero_idx)
            new_sub_graph_idxs = []
            for idx_set in sub_graph_idxs:
                for ids in res_ids:
                    new = idx_set.append(ids)
                    new_sub_graph_idxs.append(new)

            sub_graph_idxs = new_sub_graph_idxs

    graphs = []
    for idx_set in sub_graph_idxs:
        graph = nx.Graph()
        for idx, node in enumerate(idx_set[:-1]):
            if node != idx_set[idx+1]:
               graph.add_edge(node, idx_set[idx+1])
        graphs.append(graph)

    return graphs, sub_graph_idxs

def is_subgraph(graph1, graph2):
    graph_matcher = isomorphism.GraphMatcher(graph1, graph2)
    return graph_matcher.subgraph_is_isomorphic()

def _get_links(meta_molecule, edge):
    links = []
    res_names = meta_molecule.get_edge_resname(edge)
    link_resids = []
    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    for link in meta_molecule.force_field.links:
        #print("link" ,link)
        #1. first check a link applies to the residues by name
        link_resnames = set(nx.get_node_attributes(link, "resname").values())

        if res_names[0] in link_resnames and res_names[1] in link_resnames:
            #2. check if order attributes match and extract resids
            orders = list(nx.get_node_attributes(link, "order").values())
            sub_graphs, resids = get_subgraphs(meta_molecule, orders, edge)
            for idx, graph in enumerate(sub_graphs):
                if is_subgraph(meta_molecule, graph):
                    link_resids.append([ i+1 for i in resids[idx]])
                    links.append(link)

    return links, link_resids

class ApplyLinks(Processor):
    """
    This processor takes a a class:`polyply.src.MetaMolecule` and
    creates edges for the higher resolution molecule stored with
    the MetaMolecule.
    """

    def run_molecule(self, meta_molecule):
        """
        Process a single molecule. Must be implemented by subclasses.
        Parameters
        ----------
        molecule: polyply.src.meta_molecule.MetaMolecule
             The meta molecule to process.
        Returns
        -------
        vermouth.molecule.Molecule
            Either the provided molecule, or a brand new one.
        """

        molecule = meta_molecule.molecule
        force_field = meta_molecule.force_field

        for edge in meta_molecule.edges:
            links, resids = _get_links(meta_molecule, edge)
            print("resids", resids, edge)
            for link, idxs in zip(links, resids):
                try:
                    #print(idxs)
                    apply_link_between_residues(molecule, link, idxs)
                except MatchError:
                    continue

        meta_molecule.molecule = molecule
        return meta_molecule
