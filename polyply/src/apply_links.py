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
#
from collections import defaultdict
from itertools import combinations
import networkx as nx
from tqdm import tqdm
import vermouth.molecule
from vermouth.log_helpers import StyleAdapter, get_logger
from vermouth.molecule import Interaction, attributes_match
from vermouth.graph_utils import make_residue_graph
from vermouth.processors.do_links import match_order, _is_valid_non_edges, _any_pattern_match
from .processor import Processor
from .graph_utils import neighborhood

LOGGER = StyleAdapter(get_logger(__name__))

class MatchError(Exception):
    """Raised we find no match between links and molecule"""

def expand_excl(molecule):
    """
    Given a `molecule` add exclusions for nodes that
    have the exclude attribute.

    Parameters:
    -----------
    molecule: :class:`vermouth.molecule`

    Returns:
    --------
    :class:`vermouth.molecule`
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

def find_atoms(molecule, ignore=[], **attrs):
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
                new_edges = list(zip(atoms[:-1], atoms[1:]))
                molecule.add_edges_from(new_edges)
            else:
                raise IOError("Atoms of link interaction {} are not "
                              "part of the molecule.".format(interaction))

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

def _assign_link_resids(res_link, match):
    """
    Given a link at residue level (`res_link`) and
    a dict (`match`) specifying to which resids
    the res_link nodes map, create a correspondence
    dict that maps the higher resolution nodes to
    the resids specified in` match`. Each `res_link`
    node by definition can only map to one resid provided
    in `match`. The lower resolution nodes associated
    to a particular res_link node are stored in the 'graph'
    attribute of res_link.
    Note that the nodes in that graph are consecutive
    and uniquely specify to a single node in the graph
    specifying all residues at higher resolution.

    Parameters:
    -----------
    res_link: :class:`nx.Graph`
        must have a 'graph' attribute for each node
        that itself is a graph of the atoms represented
    match: dict
        dict matching a resid to the node_key of res_link

    Returns:
    --------
    :type:dict
        correspondence of high resolution nodes to resid
    """
    link_node_to_resid = {}
    for resid, link_node in match.items():
        for node in res_link.nodes[link_node]['graph']:
            link_node_to_resid[node] = resid
    return link_node_to_resid

def match_link_and_residue_atoms(meta_molecule, link, link_to_resid):
    """
    Given a meta_molecule a link and a correspondence of the link
    nodes and those in the meta_molecule, establish a correspondence
    between the link nodes and those in the higher resolution
    molecule of the meta_molecule. Note that the meta_molecule needs
    to have the "block" attribute describing which atoms a single
    meta_molecule node described.

    Parameters:
    -----------
    meta_molecule: :class:`polyply.src.meta_molecule.MetaMolecule`
    link:          :class:`vermouth.molecule.Link`
    link_to_resid:  :type:dict
        correspondence dict of link nodes to meta_molecule nodes

    Returns:
    --------
    :type:dict
        correspondence dict of link nodes to atoms in the
        the meta_molecule.molecule attribute
    """
    link_to_mol = {}
    for node in link.nodes:
        meta_mol_key = link_to_resid[node]
        block = meta_molecule.nodes[meta_mol_key]["graph"]
        resid = block.nodes[list(block.nodes)[0]]["resid"]
        attrs = link.nodes[node]
        # relative resid has been asserted before so we can
        # exclude it here
        ignore = ['order', 'charge_group', 'replace', 'resid']
        matchs = list(find_atoms(block, ignore=ignore, **attrs))

        if len(matchs) == 1:
            link_to_mol[node] = matchs[0]
        elif len(matchs) == 0:
            msg = "Found no matchs for node {} in resiue {}. Cannot apply link."
            raise MatchError(msg.format(node, resid))
        else:
            msg = "Found {} matches for node {} in resiue {}. Cannot apply link."
            raise MatchError(msg.format(len(matchs), node, resid))

    return link_to_mol

def _res_match(node1, node2):
    """
    Helper function which returns true if the resname
    attribute of two nodes matches and false otherwise.
    This function correctly handles choice objects
    for the resname.

    Parameters:
    -----------
    node1:  :type:dict
        attribute dict of node
    node2:  :type:dict
        attribute dict of node

    Returns:
    --------
    bool
    """
    # this is not equivalent to comparing just the resname
    # attributes because resname can be a choice object
    # which at the moment does not compare properly
    ignore = [key for key in node2.keys() if key != "resname"]
    return attributes_match(node1, node2, ignore_keys=ignore)

def _linktype_match(edge_attrs1, edge_attrs2):
    type1 = edge_attrs1.get("linktype", None)
    type2 = edge_attrs2.get("linktype", None)
    return type1 == type2

def _resnames_match(resnames, allowed_resnames):
    """
    Return true if one element in resnames matches
    one element in allowed_resnames.

    Parameters
    ----------
    resnames: `abc.iterable`
    allowed_resnames: `abc.iterable`
    """
    for resname in resnames:
        if resname in allowed_resnames:
           return True
    return False

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
        if isinstance(name, vermouth.molecule.LinkPredicate) and not isinstance(name.value, str):
            out_resnames.update(name.value)
        else:
            out_resnames.add(name)

    return out_resnames

class ApplyLinks(Processor):
    """
    This processor takes a a class:`polyply.src.MetaMolecule` and
    based on links defined in the `force-field` attribute of the
    MetaMolecule applies them when appropiate.

    Links are defined as connections between two residues at the
    high resolution level whereas the MetaMolecule is the molecule
    at residue level or higher degree of CG. The algorithm proceeds
    by first coarse-graining a link to residue level. Subsequently
    it is checked based on the connectivity of the link at residue
    level and the residue name to which residues in the MetaMolecule
    this link fits. Subsequently, based on the high resolution
    definition the algorithm looks up the atoms in the matching residues
    corresponding to the link. If all atoms are present the link is
    applied otherwise the link is not applied.

    Note that because links can overwrite each other they are first
    stored in the instance variable `applied_links` and only written
    to the MetaMolecule.molecule graph after all links have been
    checked and overwritten if needed. This means the apply_links_between_residues
    method only populates this instance variable, while write to
    molecule applies those links.
    """
    def __init__(self, *args, debug=False, **kwargs):
        self.debug = debug
        super().__init__(*args, **kwargs)
        self.applied_links = defaultdict(dict)
        self.nodes_to_remove = []

    def _update_interactions_dict(self, interactions_dict, molecule, citations, mapping=None):
        """
        Helper function for adding links to the applied links dictionary.
        If mapping is given the interaction atoms are mapped to the molecule
        atoms using mapping. Otherwise interactions are assumed to be written
        in terms of molecule nodes.
        """
        for inter_type, interactions in interactions_dict.items():
            for interaction in interactions:
                # it is not guaranteed that interaction.atoms is a tuple
                # the key is the atoms involved in the interaction and the version type so
                # that multiple versions are kept and not overwritten
                if mapping:
                    new_interaction = _build_link_interaction_from(molecule,
                                                                   interaction,
                                                                   mapping)
                else:
                    new_interaction = interaction

                interaction_key = (*new_interaction.atoms, new_interaction.meta.get("version", 1))
                self.applied_links[inter_type][interaction_key] = (new_interaction, citations)

    def apply_link_between_residues(self, meta_molecule, link, link_to_resid):
        """
        Applies a link between specific residues, if and only if
        the link atoms (incl. all attributes) match at most one atom
        in a respective link. It adds the link to the applied_links
        instance variable, from which later the links are added to
        the molecule. Note that replace statements are already update
        the molecule, as they potentially apply to consecutive links.
        Edges are also updated in place.

        Parameters
        ----------
        meta_molecule: :class:`polyply.src.MetaMolecule`
        link: :class:`vermouth.molecule.Link`
            A vermouth link definition
        link_to_resids: :type:dict
            a dict matching link nodes to a resid in meta_molecule
        """
        # handy variable for later referencing
        molecule = meta_molecule.molecule

        # look-up which atoms in the link correspond
        # to which atoms in the molecule this function
        # raises an MatchError if no atoms are found

        link_to_mol = match_link_and_residue_atoms(meta_molecule,
                                                   link,
                                                   link_to_resid)

        # check if the non-edge criteria are satisfied
        if not _is_valid_non_edges(molecule, link, link_to_mol):
            msg = "Found edge, which should not be there. Cannot apply link."
            raise MatchError(msg)

        # check if any of the patterns matches
        any_pattern_match = _any_pattern_match(molecule, link.patterns, link_to_mol)
        if link.patterns and (not any_pattern_match):
            msg = "No pattern matches! Cannot apply link."
            raise MatchError(msg)

        # if all atoms have a match the link applies and we first
        # replace any attributes from the link node section
        for node in link.nodes:
            # if the atomname is set to null we schedule the node to be removed
            if link.nodes[node].get('replace', {}).get('atomname', False) is None:
                self.nodes_to_remove.append(link_to_mol[node])
            else:
                molecule.nodes[link_to_mol[node]].update(link.nodes[node].get('replace', {}))

        # based on the match we build the link interaction
        self._update_interactions_dict(link.interactions,
                                       molecule,
                                       link.citations,
                                       mapping=link_to_mol)

        # now we already add the edges of this link
        # links can overwrite each other but the edges must be the same
        # this is safer than using the make_edge method because it accounts
        # for edges written in the edges directive
        for edge in link.edges:
            molecule.add_edge(link_to_mol[edge[0]], link_to_mol[edge[1]])

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
        :class:`polyply.src.meta_molecule.MetaMolecule`
        """
        molecule = meta_molecule.molecule
        force_field = meta_molecule.force_field
        # we need to update the temporary interactions dict
        self._update_interactions_dict(molecule.interactions, molecule, molecule.citations, mapping=None)
        # now we can clear the molecule dict
        molecule.interactions = defaultdict(list)

        resnames = set(nx.get_node_attributes(molecule, "resname").values())
        for link in tqdm(force_field.links):
            link_resnames = _get_link_resnames(link)
            if not _resnames_match(resnames, link_resnames) or not attributes_match(molecule.meta, link.molecule_meta):
                continue

            # we only use the order because each order needs to be
            # matching exactly 1 residue, which means their resname
            # needs to match as well. However, resname can be a
            # choice object which is not hashable so we don't
            # use resname in constructing the res-graph.
            res_link = make_residue_graph(link, attrs=('order',))
            # however when finding the LCIS we do match against the residue
            # name and topology of the link
            GM = nx.isomorphism.GraphMatcher(meta_molecule,
                                             res_link,
                                             node_match=_res_match,
                                             edge_match=_linktype_match)
            raw_matchs = GM.subgraph_isomorphisms_iter()
            for match in raw_matchs:
                nodes = match.keys()
                resids =[meta_molecule.nodes[node]["resid"] for node in nodes]
                orders = [ res_link.nodes[match[node]]["order"] for node in nodes]
                if _check_relative_order(resids, orders):
                    link_node_to_resid = _assign_link_resids(res_link, match)
                    try:
                        self.apply_link_between_residues(meta_molecule, link, link_node_to_resid)
                    except MatchError as error:
                        LOGGER.debug(str(error), type='step')

        # take care to remove nodes if there are any scheduled for removal
        # we do this here becuase that's more efficent
        molecule.remove_nodes_from(self.nodes_to_remove)
        # now we add all interactions but not the ones that contain the removed
        # nodes
        for inter_type in self.applied_links:
            for atoms, (interaction, citation) in self.applied_links[inter_type].items():
                if not any(atom in self.nodes_to_remove for atom in atoms):
                    meta_molecule.molecule.interactions[inter_type].append(interaction)
                    meta_molecule.molecule.citations.update(citation)

        for link in force_field.links:
            if link.molecule_meta.get('by_atom_id'):
                apply_explicit_link(molecule, link)

        expand_excl(molecule)
        return meta_molecule
