import networkx as nx
from networkx.algorithms import isomorphism
import vermouth.molecule
from polyply.src.processor import Processor  # Should be a relative import: from .processor import Processor


class MatchError(Exception):
    """Raised we find no match between links and molecule"""


def find_atoms(molecule, **attrs):
    """
    Yields all indices of atoms that match `attrs`

    Parameters
    ----------
    molecule: :class:`vermouth.molecule.Molecule`
    **attrs: collections.abc.Mapping
        The attributes and their desired values.

    Yields
    ------
    collections.abc.Hashable
        All atom indices that match the specified `attrs`
    """
    try:
        ignore = attrs['ignore']  # We discussed this elsewhere.
        del attrs['ignore']
    except KeyError:
        ignore = []

    for node_idx in molecule:
        node = molecule.nodes[node_idx]
        if vermouth.molecule.attributes_match(node, attrs, ignore_keys=ignore):
            yield node_idx


def _build_link_interaction_from(molecule, interaction, match):
    """
    Creates a new interaction from a link type interaction
    and a match specification for a molecule. Note that
    it can overwrite equivalent interactions.

    Parameters
    ----------
    molecule: :class:`vermouth.molecule.Molecule`
    interaction: :type:`vermoth.molecule.Interaction`
    match: :type:`dict`
        The mapping between the interaction atom names
        and the molecule atom names.

    Returns
    ------
    `vermouth.molecule.Interaction`
    """
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
    in a respective link. It updates the molecule in place.

    Parameters
    ----------
    molecule: :class:`vermouth.molecule.Molecule`
        A vermouth molecule definition
    link: :class:`vermouth.molecule.Link`
        A vermouth link definition
    resids: dictionary
        a list of node attributes used for link matching aside from
        the residue ordering
    """
    # we have to go on resid or at least one criterion otherwise
    # the matching will be super slow, if we need to iterate
    # over all combinations of a possible links.
    link = link.copy()
    nx.set_node_attributes(link, dict(zip(link.nodes, resids)), 'resid')
    link_to_mol = {}
    for node in link.nodes:
        attrs = link.nodes[node]
        attrs.update({'ignore': ['order', 'charge_group']})
        matchs = [atom for atom in find_atoms(molecule, **attrs)]
        if len(matchs) == 1:
            link_to_mol[node] = matchs[0]
        elif len(matchs) == 0:
            msg = "Found no matchs for atom {} in resiue {}. Cannot apply link."
            raise MatchError(msg.format(attrs["atomname"], attrs["resid"]))
        else:
            msg = "Found {} matches for atom {} in resiue {}. Cannot apply link."
            raise MatchError(msg.format(len(matchs), attrs["atomname"], attrs["resid"]))

    for inter_type in link.interactions:
        for interaction in link.interactions[inter_type]:
            new_interaction = _build_link_interaction_from(molecule, interaction, link_to_mol)
            molecule.add_or_replace_interaction(inter_type, *new_interaction)
            atoms = interaction.atoms
            new_edges = [(link_to_mol[at1], link_to_mol[at2]) for at1, at2 in zip(atoms[:-1], atoms[1:])]
            molecule.add_edges_from(new_edges)


def apply_explicit_link(molecule, link):
    """
    Applies interactions from a link regardless of any
    checks. This requires atoms in the link to be of
    int type. Within polyply the explicit flag can  # Atoms are never int. Their keys could be though
    be set to these links for the program to know
    to apply them using this method.

    Parameters
    ----------
    molecule: :class:`vermouth.molecule.Molecule`
        A vermouth molecule definintion
    link: :class:`vermouth.molecule.Link`
        A vermouth link definintion
    """
    for inter_type, value in link.interactions.items():  # value bad variable name
        new_interactions = []
        for interaction in value:
            try:
                interaction.atoms[:] = [int(atom) - 1 for atom in interaction.atoms]
            except ValueError:
                msg = """Trying to apply an explicit link but interaction
                      {} but cannot convert the atoms to integers. Note
                      explicit links need to be defined by atom numbers."""
                raise ValueError(msg.format(interaction))
            if set(interaction.atoms).issubset(set(molecule.nodes)):
                new_interactions.append(interaction)
            else:
                raise IOError("Atoms of link interaction {} are not "
                              "part of the molecule.".format(interaction))
        molecule.interactions[inter_type] += new_interactions  # Use molecule.add_or_replace_interaction?


def neighborhood(graph, node, degree):  # Degree wrong name. Distance is more appropriate.
    """
    Returns all neighbours of `node` that are less or equal
    to `degree` nodes away within a graph excluding the node
    itself.

    Adapted from: https://stackoverflow.com/questions/
    22742754/finding-the-n-degree-neighborhood-of-a-node

    Parameters
    ----------
    graph: :class:`networkx.Graph`
        A networkx graph definintion
    node:
        A node key matching one in the graph
    degree: :type:`int`
        The maxdistance between the node and its
        neighbours.

    Returns
    --------
    int
    """

    path_lengths = nx.single_source_dijkstra_path_length(graph, node)  # networkx documentation is down, so I can't chcek. Can't you give a max_distance to traverse to this method? Is there  asimilar method wher eyou can?
    # See also: nx.ego_graph
    neighbours = [node for node, length in path_lengths.items() if 0 < length <= degree]
    return neighbours


def get_subgraphs(meta_molecule, orders, edge):  # incomprehensible docstring, bad function name
    """
    Creates a residue based graph from the orders
    attribute and an edge of a meta_molecule. For each order
    index in orders a matching resid in meta_molecule
    is found essentially by adding the first edge atom
    and the order index unless the order index is a
    a signed order (e.g. '<'). In that case all neighbours
    that are the numbers of orders - 1 away are found. At residue
    level all interactions must be connected and are at most
    the term length (i.e. orders - 1) away. Currently this way
    of interpreting the signed orders ignores larger versus
    smaller. For each set of neighbours a new graph is created.

    Parameters
    ----------
    meta_molecule: :class:`polyply.MetaMolecule`
        A polyply meta_molecule definintion
    orders: :type:`list`
        A list of all order parameters
    edge:
        Single edge matching one in the meta_molecule

    Returns
    ----------
    list  # List containing what? Also incorrect, actually returns tuple.
    """
    # This functions needs drastic refactoring, better variable names, more helper functions, and a better docstring.
    sub_graph_idxs = [[]]

    for idx, order in enumerate(orders):
        if isinstance(order, int):
            resid = edge[0] + order
            for idx_set in sub_graph_idxs:
                idx_set.append(resid)
        else:
            res_ids = neighborhood(meta_molecule, edge[0], len(orders)-1)
            new_sub_graph_idxs = []
            for idx_set in sub_graph_idxs:
                for _id in res_ids:
                    new = idx_set + [_id]
                    new_sub_graph_idxs.append(new)

            sub_graph_idxs = new_sub_graph_idxs

    graphs = []
    for idx_set in sub_graph_idxs:
        graph = nx.Graph()
        for idx, node in enumerate(idx_set[:-1]):  # Why not the last one? Also, you're not treating idx_set as a set. So BAD var name.
            if node != idx_set[idx+1]:
                graph.add_edge(node, idx_set[idx+1])  # So, if I understand correctly, idx_set actuall contains a bunch of edges (in a condensed format)?
        graphs.append(graph)

    return graphs, sub_graph_idxs


def is_subgraph(graph1, graph2):
    """
    Checks if graph1 is subgraph isomorphic to graph1.

    Parameters
    ----------
    graph1: :class:`networkx.Graph`
    graph2: :class:`networkx.Graph`

    Returns
    ----------
    bool
    """
    # I think there should be a nx.is_subgraph_isomorphic function.
    graph_matcher = isomorphism.GraphMatcher(graph1, graph2)
    return graph_matcher.subgraph_is_isomorphic()


def _get_link_resnames(link):
    # Missing docstring
    res_names = list(nx.get_node_attributes(link, "resname").values())
    out_resnames = []
    for name in res_names:
        try:
            out_resnames += name.value  # ??
        except AttributeError:
            out_resnames.append(name)

    return set(out_resnames)  # Make out_resnames a set to start with


def _get_links(meta_molecule, edge):
    """
    Collects all links and matching resids within a `meta_molecule`
    that include the atoms defined in `edge`. It ignores all
    links, which have the meta_molecule explicit flag set to True.
    A link is applicable to an edge if the resnames of the edge
    match the resnames specified with the link and if the graph
    of resids is a subgraph to the meta_molecule resid graph.

    Parameters
    -----------
    meta_molecule: :class:`polyply.MetaMolecule`
        A polyply meta_molecule definition
    edge:
        Single edge matching one in the meta_molecule
    Returns
    ---------
    list of links and list of lists of resids
    """
    links = []
    res_names = meta_molecule.get_edge_resname(edge)
    link_resids = []
    for link in meta_molecule.force_field.links:
        link_resnames = _get_link_resnames(link)
        if link.molecule_meta.get('explicit'):
            continue
        elif res_names[0] in link_resnames and res_names[1] in link_resnames:
            #2. check if order attributes match and extract resids  # Where's #1.?
            orders = list(nx.get_node_attributes(link, "order").values())
            sub_graphs, resids = get_subgraphs(meta_molecule, orders, edge)
            for idx, graph in enumerate(sub_graphs):
                if is_subgraph(meta_molecule, graph):
                    link_resids.append([i+1 for i in resids[idx]])
                    links.append(link)

    return links, link_resids


class ApplyLinks(Processor):
    """
    This processor takes a a class:`polyply.src.MetaMolecule` and
    creates edges for the higher resolution molecule stored with
    the MetaMolecule.
    """
    def run_molecule(self, meta_molecule):
        """  # This docstring comes from the parent class. Either remove it, or update it to what this class does
        Process a single molecule. Must be implemented by subclasses.

        Parameters
        ----------
        molecule: :class:`polyply.src.meta_molecule.MetaMolecule`
             The meta molecule to process.

        Returns
        ---------
        :class: `polyply.src.meta_molecule.MetaMolecule`
        """
        molecule = meta_molecule.molecule
        force_field = meta_molecule.force_field

        for edge in meta_molecule.edges:
            links, resids = _get_links(meta_molecule, edge)
            for link, idxs in zip(links, resids):
                try:
                    apply_link_between_residues(molecule, link, idxs)
                except MatchError:
                    continue

        for link in force_field.links:
            if link.molecule_meta.get('explicit'):
                apply_explicit_link(molecule, link)

        meta_molecule.molecule = molecule  # This should not be nescessary
        return meta_molecule
