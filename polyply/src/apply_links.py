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
from vermouth.molecule import Interaction, attributes_match
from vermouth.graph_utils import make_residue_graph
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

def  _assing_link_resids(res_link, match):
     link_node_to_resid = {}
     for resid, link_node in match.items():
         for node in res_link.nodes[link_node]['graph']:
             link_node_to_resid[node] = resid
     return link_node_to_resid

class ApplyLinks(Processor):
    """
    This processor takes a a class:`polyply.src.MetaMolecule` and
    creates edges for the higher resolution molecule stored with
    the MetaMolecule.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.applied_links = defaultdict(dict)

    def apply_link_between_residues(self, meta_molecule, link, link_to_resid):
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
        link = link.copy()
        # we have to go on resid or at least one criterion otherwise
        # the matching will be super slow, if we need to iterate
        # over all combinations of a possible links.
        #nx.set_node_attributes(link, dict(zip(link.nodes, resids)), 'resid')

        link_to_mol = {}
        for node in link.nodes:
            meta_mol_key = link_to_resid[node]
            block = meta_molecule.nodes[meta_mol_key]["block"]
            attrs = link.nodes[node]
            attrs.update({'ignore': ['order', 'charge_group', 'replace']})
            matchs = [atom for atom in find_atoms(block, **attrs)]

            if len(matchs) == 1:
                link_to_mol[node] = matchs[0]
            elif len(matchs) == 0:
                msg = "Found no matchs for atom {} in resiue {}. Cannot apply link."
                raise MatchError #(msg.format(attrs["atomname"])) #, attrs["resid"]))
            else:
                msg = "Found {} matches for atom {} in resiue {}. Cannot apply link."
                raise MatchError #(msg.format(len(matchs), attrs["atomname"], attrs["resid"]))

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
        def _res_match(node1, node2):
            ignore = [key for key in node2.keys() if key != "resname"]
            return attributes_match(node1, node2, ignore_keys=ignore)

        molecule = meta_molecule.molecule
        force_field = meta_molecule.force_field

        for link in tqdm(force_field.links):
            res_link = make_residue_graph(link, attrs=('order',))
            print(res_link.nodes(data=True))
            GM = nx.isomorphism.GraphMatcher(meta_molecule, res_link, node_match=_res_match)
            raw_matchs = GM.subgraph_isomorphisms_iter()
            for match in raw_matchs:
                resids = match.keys()
                orders = [ res_link.nodes[match[resid]]["order"] for resid in resids]
                if _check_relative_order(resids, orders):
                    print("go here")
                    link_node_to_resid = _assing_link_resids(res_link, match)
                    try:
                        self.apply_link_between_residues(meta_molecule, link, link_node_to_resid)
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
