# Copyright 2020 University of Groningen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import defaultdict
from itertools import combinations
import networkx as nx
from tqdm import tqdm
import vermouth.molecule
from vermouth.molecule import Interaction
from vermouth.processors.do_links import match_order
from .processor import Processor

class MatchError(Exception):
    """Raised we find no match between links and molecule"""

def expand_excl(molecule):
    """
    Given a `molecule` add exclusions for nodes that
    have the exclude attribute.

    Parameters:
    -----------
    molecule: `:class:vermouth.molecule`

    Returns:
    --------
    `:class:vermouth.molecule`
    """
    exclude = nx.get_node_attributes(molecule, "exclude")
    nrexcl = molecule.nrexcl
    had_excl=[]
    for node, excl in exclude.items():
        if excl > nrexcl:
            excluded_nodes = neighborhood(molecule, node, max_length=excl, min_length=nrexcl)
            for ndx in excluded_nodes:
                excl = Interaction(atoms=[node, ndx],
                                   parameters=[],
                                   meta={})
                if frozenset([node, ndx]) not in had_excl and node != ndx:
                    had_excl.append(frozenset([node, ndx]))
                    molecule.interactions["exclusions"].append(excl)
    return molecule

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


def apply_explicit_link(molecule, link):
    """
    Applies interactions from a link regardless of any
    checks. This requires atom keys in the link to be of
    int type. Within polyply the explicit flag can
    be set to these links for the program to know
    to apply them using this method.

    Parameters
    ----------
    molecule: :class:`vermouth.molecule.Molecule`
        A vermouth molecule definintion
    link: :class:`vermouth.molecule.Link`
        A vermouth link definintion
    """
    for inter_type, inter_list in link.interactions.items():
        for interaction in inter_list:
            try:
                interaction.atoms[:] = [int(atom) - 1 for atom in interaction.atoms]
            except ValueError as err:
                msg = """Trying to apply an explicit link but interaction
                      {} but cannot convert the atoms to integers. Note
                      explicit links need to be defined by atom numbers."""
                raise ValueError(msg.format(interaction)) from err
            if set(interaction.atoms).issubset(set(molecule.nodes)):
                molecule.add_or_replace_interaction(inter_type, interaction.atoms,
                                                    interaction.parameters,
                                                    meta=interaction.meta)
                atoms = interaction.atoms
                new_edges = [(at1, at2)
                            for at1, at2 in zip(atoms[:-1], atoms[1:])]
                molecule.add_edges_from(new_edges)
            else:
                raise IOError("Atoms of link interaction {} are not "
                              "part of the molecule.".format(interaction))

def neighborhood(graph, source, max_length, min_length=1):
    """
    Returns all neighbours of `source` that are less or equal
    to `cutoff` nodes away and more or equal to `start` away
    within a graph excluding the node itself.

    Parameters
    ----------
    graph: :class:`networkx.Graph`
        A networkx graph definintion
    source:
        A node key matching one in the graph
    max_length: :type:`int`
        The maxdistance between the node and its
        neighbours.
    min_length: :type:`int`
        The minimum length of a path. Default
        is zero

    Returns
    --------
    list
       list of all nodes distance away from reference
    """
    paths = nx.single_source_shortest_path(G=graph, source=source, cutoff=max_length)
    neighbours = [ node for node, path in paths.items() if min_length <= len(path)]
    return neighbours

def _check_relative_order(resids, orders):
    """
    This function checks if the relative order of
    list of lists of residues adheres to the order
    specifications of vermouth orders.
    """

    order_match = {}
    for order, resid in zip(orders, resids):
        if order not in order_match:
            order_match[order] = resid
        # Assert all orders correspond to the same resid
        elif order_match[order] != resid:
            return False

    for ((order1, resid1), (order2, resid2)) in combinations(order_match.items(), 2):
        # Assert the differences between resids correspond to what
        # the orders require.
        if not match_order(order1, resid1, order2, resid2):
            return False

    return True

def _orders_to_paths(meta_molecule, orders, node):
    """
    Takes the `order` attributes of a `vermouth.molecule`
    and a `MetaMolecule` and returns a list of all paths
    that match the order centered at a given `node`. In
    this context a path is of length orders - 1 and starts
    at node. Note that at the residue level a path cannot
    be between nodes that are not connected by an edge. Also
    note that because of tokens of the form '<' a single
    order token can generate multiple paths.

    Parameters
    ------------
    meta_molecule: :class:`polyply.MetaMolecule`
        A polyply meta_molecule definintion
    orders: :type:`list`
        A list of order tokens
    node:
        Single node matching one in the meta_molecule
    """
    paths = [[]]
    for token in orders:
        #1. deal with order tokens that are resids
        if isinstance(token, int):
            resid = node + token
            for path in paths:
                path.append(resid)

        #2. deal with larger and smaller tokens
        elif '>' in token or '<' in token:
            neighbours = neighborhood(meta_molecule, node,
                                      len(orders)-1)
            if '>' in token:
                res_ids = [_id for _id in neighbours if
                           _id > node]
            else:
                res_ids = [_id  for _id in neighbours if
                           _id < node]

            # each resid in resids spwans a new path in
            # that needs to be added to paths discarding
            # the old one. so we first generate them and
            # then overwrite the old paths list.
            new_paths = []
            for path in paths:
                for _id in res_ids:
                    new = path + [_id]
                    new_paths.append(new)

            paths = new_paths

        #3. raise error if we do not know a token
        else:
            msg = "Cannot interpret order token {}."
            raise IOError(msg.format(token))

    clean_paths = []
    for path in paths:
        if _check_relative_order(path, orders):
            clean_paths.append(path)

    return clean_paths

def gen_link_fragments(meta_molecule, orders, node):
    """
    Generate all fragments of meta_molecule that match
    the order specification at a given node. The function
    returns a list of graphs, which are each a single
    path in residue space matching the order attribute
    at a given node. It also returns the raw paths,
    which are the residues in meta_molecule corresponding
    to the nodes of a given link.

    Parameters
    ----------
    meta_molecule: :class:`polyply.MetaMolecule`
        A polyply meta_molecule definintion
    orders: :type:`list`
        A list of all order parameters
    node:
        Single node matching one in the meta_molecule

    Returns
    ----------
    tuple
         list of nx.Graph
             fragments of link in residue space
         list of int
             indices of link nodes
    """
    paths = _orders_to_paths(meta_molecule, orders, node)
    graphs = []
    for path in paths:
        graph = nx.Graph()
        for idx, _node in enumerate(path[:-1]):
            if _node != path[idx+1]:
                graph.add_edge(_node, path[idx+1])
        graphs.append(graph)

    return graphs, paths


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
    for node in graph2.nodes:
        if not node in graph1.nodes:
            return False

    for edge in graph2.edges:
        if not graph1.has_edge(edge[0], edge[1]):
            return False

    return True


def _get_link_resnames(link):
    """
    Get the resnames of a `link` directly from the node attributes
    of the link atoms. It is not safe enough to just take the
    link name, because it is often undefined or empty. Note that
    a link name can also be a `vermouth.molecule.Choice` object.

    Parameters
    ----------
    link: :class:`vermouth.molecule.Link`

    Returns
    ----------
    set
      all unique resnames of a molecule
    """
    res_names = list(nx.get_node_attributes(link, "resname").values())
    out_resnames = set()

    for name in res_names:
        if isinstance(name, vermouth.molecule.Choice):
            out_resnames.update(name.value)
        else:
            out_resnames.add(name)

    return out_resnames


def _get_links(meta_molecule, edge):
    """
    Collects all links and matching resids within a `meta_molecule`
    that include the edge defined in `edge`. It ignores all
    links, which have the meta_molecule explicit flag set to True.
    A link is applicable to an edge, if the resnames of the edge
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
    tuple
        list of links
        list of lists of resids
    """

    links = []
    res_names = meta_molecule.get_edge_resname(edge)
    link_resids = []

    for link in meta_molecule.force_field.links:
        link_resnames = _get_link_resnames(link)

        if link.molecule_meta.get('by_atom_id'):
            continue

        if res_names[0] in link_resnames and res_names[1] in link_resnames:
            orders = list(nx.get_node_attributes(link, "order").values())
            for idx in edge:
                sub_graphs, resids = gen_link_fragments(meta_molecule, orders, idx)
                for idx, graph in enumerate(sub_graphs):
                    if is_subgraph(meta_molecule, graph):
                        if len(resids[idx]) == len(orders):
                            link_resid = [i+1 for i in resids[idx]]
                            link_resids.append(link_resid)
                            links.append(link)

    return links, link_resids


class ApplyLinks(Processor):
    """
    This processor takes a a class:`polyply.src.MetaMolecule` and
    creates edges for the higher resolution molecule stored with
    the MetaMolecule.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.applied_links = defaultdict(dict)

    def apply_link_between_residues(self, meta_molecule, link, resids):
        """
        Applies a link between specific residues, if and only if
        the link atoms (incl. all attributes) match at most one atom
        in a respective link. It adds the link to the applied_links
        instance variable, which from which later the links are added to
        the molecule. Note that replace statements are already update
        the molecule, as they potentially apply to conscutive links.
        Edges are also updated in place.

        Parameters
        ----------
        meta_molecule: :class:`polyply.src.MetaMolecule`
        link: :class:`vermouth.molecule.Link`
            A vermouth link definition
        resids: dictionary
            a list of node attributes used for link matching aside from
            the residue ordering
        """
        # handy variable for later referencing
        molecule = meta_molecule.molecule
        # we have to go on resid or at least one criterion otherwise
        # the matching will be super slow, if we need to iterate
        # over all combinations of a possible links.
        link = link.copy()
        nx.set_node_attributes(link, dict(zip(link.nodes, resids)), 'resid')
        link_to_mol = {}
        for node in link.nodes:
            block = meta_molecule.nodes[link.nodes[node]["resid"]-1]["block"]
            attrs = link.nodes[node]
            attrs.update({'ignore': ['order', 'charge_group', 'replace']})
            matchs = [atom for atom in find_atoms(block, **attrs)]

            if len(matchs) == 1:
                link_to_mol[node] = matchs[0]
            elif len(matchs) == 0:
                msg = "Found no matchs for atom {} in resiue {}. Cannot apply link."
                raise MatchError(msg.format(attrs["atomname"], attrs["resid"]))
            else:
                msg = "Found {} matches for atom {} in resiue {}. Cannot apply link."
                raise MatchError(msg.format(len(matchs), attrs["atomname"], attrs["resid"]))

        for node in link.nodes:
            if "replace" in link.nodes[node]:
                # if we don't find a key a MatchError is directly detected and the link
                # not applied
                for key, item in link.nodes[node]["replace"].items():
                    molecule.nodes[link_to_mol[node]][key] = item

        for inter_type in link.interactions:
            for interaction in link.interactions[inter_type]:
                new_interaction = _build_link_interaction_from(molecule, interaction, link_to_mol)
                # it is not guaranteed that interaction.atoms is a tuple
                # the key is the atoms involved in the interaction and the version type so
                # that multiple versions are kept and not overwritten
                interaction_key = tuple(new_interaction.atoms) +\
                                  tuple([new_interaction.meta.get("version",1)])
                self.applied_links[inter_type][interaction_key] = new_interaction
                # now we already add the edges of this link
                atoms = tuple(interaction.atoms)
                new_edges = [(link_to_mol[at1], link_to_mol[at2])
                             for at1, at2 in zip(atoms[:-1], atoms[1:])]
                molecule.add_edges_from(new_edges)

    def run_molecule(self, meta_molecule):
        """
        Given a meta_molecule the function iterates over all edges
        (i.e. pairs of residues) and finds all links that fit based
        on residue-name and order attribute. Subsequently it tries
        to apply these links. If a MatchError is encountered the
        link is not applied. The meta_molecule is updated in place
        but also returned.

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

        for edge in tqdm(meta_molecule.edges):
            links, resids = _get_links(meta_molecule, edge)
            for link, idxs in zip(links, resids):
                try:
                    self.apply_link_between_residues(meta_molecule, molecule, link, idxs)
                except MatchError:
                    continue

        for inter_type in self.applied_links:
            for interaction in self.applied_links[inter_type].values():
                meta_molecule.molecule.interactions[inter_type].append(interaction)

        for link in force_field.links:
            if link.molecule_meta.get('by_atom_id'):
                apply_explicit_link(molecule, link)

        expand_excl(molecule)
        return meta_molecule
