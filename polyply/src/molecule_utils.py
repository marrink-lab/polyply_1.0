# Copyright 2022 University of Groningen
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
import itertools
from collections import defaultdict
import numpy as np
import networkx as nx
import vermouth
from vermouth.molecule import Interaction
from polyply.tests.test_lib_files import _interaction_equal
from .topology import replace_defined_interaction
from .graph_utils import find_connecting_edges

def diffs_to_prefix(atoms, resid_diffs):
    """
    Given a list of atoms and corresponding differences
    between their resids, generate the offset prefix for
    the atomnames according to the vermouth sepcific offset
    language.

    The reference atom must have resid_diff value of 0.
    Other atoms either get - or + signs
    depending on their resid offset.

    Parameters
    ----------
    atoms: abc.itertable[str]
    resid_diff: abc.itertable[int]
        the differences in resid with respeect to
        the smallest/largest resid which is 0

    Returns
    -------
    abc.itertable
        list with prefixed atom names
    """
    prefixed_atoms = []
    for atom, diff in zip(atoms, resid_diffs):
        if diff > 0:
            prefix = "".join(["+" for i in range(0, diff)])
        else:
            prefix = "".join(["-" for i in range(diff, 0)])
        prefixed_atoms.append(prefix + atom)
    return prefixed_atoms

def _extract_edges_from_shortest_path(atoms, block, min_resid):
    """
    Given a list atoms generate a list of edges correspoding to
    all edges required to connect all atoms by at least one
    shortest path. Edges are retunred on atomname basis with
    prefix relative to the `min_resid`. See diffs_to_prefix.

    Paramters:
    ----------
    atoms: abc.itertable
        the atoms to collect edges for
    block: :class:`vermouth.molecule.Block`
        the molecule which to servey for edges
    min_resid: int
        the resid to which the prefix indicate relative resid
        distance

    Returns
    -------
    list[tuple]
        the edge list by atomname with prefix indicating relative
        residue distance to min_resid
    """
    edges = []
    had_edges = []
    final_atoms = {}
    resnames = {}
    for origin, target in itertools.combinations(atoms, r=2):
        path = list(nx.shortest_simple_paths(block, source=origin, target=target))[0]
        for edge in zip(path[:-1], path[1:]):
            if edge not in had_edges:
                resid_diffs = np.array([block.nodes[node]['resid'] for node in edge]) - min_resid
                atom_names = [block.nodes[node]["atomname"] for node in edge]
                link_names = diffs_to_prefix(atom_names, resid_diffs)
                final_atoms.update(dict(zip(edge, link_names)))
                edges.append(link_names)
                had_edges.append(edge)
                resnames.update(zip(link_names, [ block.nodes[node]["resname"] for node in edge]))
    return final_atoms, edges, resnames


def extract_links(molecule):
    """
    Given a molecule that has the resid and resname attributes
    correctly set, extract the interactions which span more than
    a single residue and generate a link.

    Parameters
    ----------
    molecule: :class:`vermouth.molecule.Molecule`
        the molecule from which to extract interactions

    Returns
    -------
    list[:class:`vermouth.molecule.Links`]
        a list with a links found
    """
    links = []
    # patterns are a sqeuence of atoms that define an interaction
    # sometimes multiple interactions are defined for one pattern
    # in that case they are all collected in this dictionary
    patterns = defaultdict(dict)
    # for each found pattern the resnames are collected; this is important
    # because the same pattern may apply to residues with different name
    resnames_for_patterns = defaultdict(dict)
    link_atoms_for_patterns = defaultdict(list)
    # as additional safe-gaurd against false links we also collect the edges
    # that span the interaction by finding the shortest simple path between
    # all atoms in patterns. Note that the atoms in patterns not always have
    # to be directly bonded. For example, pairs are not directly bonded and
    # can span multiple residues
    for inter_type in molecule.interactions:
        for kdx, interaction in enumerate(molecule.interactions[inter_type]):
            # extract resids and resname corresponding to interaction atoms
            resids = np.array([molecule.nodes[atom]["resid"] for atom in interaction.atoms])
            resnames = [molecule.nodes[atom]["resname"] for atom in interaction.atoms]
            # compute the resid offset to be used for the atom prefixes
            min_resid = min(resids)
            diff = resids - min_resid
            pattern = tuple(set(list(zip(diff, resnames))))

            # in this case all interactions are in a block and we skip
            if np.sum(diff) == 0:
                continue

            # we collect the edges corresponding to the simple paths between pairs of atoms
            # in the interaction
            mol_atoms_to_link_atoms, edges, resnames = _extract_edges_from_shortest_path(interaction.atoms, molecule, min_resid)
            #link_to_mol_atoms = {value:key for key, value in mol_atoms_to_link_atoms.items()}
            link_atoms =  [mol_atoms_to_link_atoms[atom] for atom in interaction.atoms]
            link_inter = Interaction(atoms=link_atoms,
                                     parameters=interaction.parameters,
                                     meta={})

            # here we deal with filtering redundancy
            if pattern in patterns and inter_type in patterns[pattern]:
                for other_inter in patterns[pattern].get(inter_type, []):
                    if _interaction_equal(other_inter, link_inter, inter_type):
                        break
                else:
                    patterns[pattern][inter_type].append(link_inter)
                    resnames_for_patterns[pattern].update(resnames)
                    link_atoms_for_patterns[pattern] += link_atoms
            else:
                patterns[pattern][inter_type] = [link_inter]
                resnames_for_patterns[pattern].update(resnames)
                link_atoms_for_patterns[pattern] += link_atoms

    # we make new links for each unique interaction per type
    for pattern in patterns:
        link = vermouth.molecule.Link()
        link.add_nodes_from(set(link_atoms_for_patterns[pattern]))
        resnames = resnames_for_patterns[pattern]
        nx.set_node_attributes(link, resnames, "resname")

        had_parameters = []
        for inter_type, inters in patterns[pattern].items():
            for idx, interaction in enumerate(inters):
                #new_parameters = interaction.parameters
                new_meta = interaction.meta
                #new_atoms = interaction.atoms
                # to account for the fact when multiple interactions with the same
                # atom patterns need to be written to ff
                new_meta.update({"version": idx})
                new_meta.update({"comment": "link"})
                had_parameters.append(interaction.parameters)
                # map atoms to proper atomnames ..
                link.interactions[inter_type].append(interaction)
        links.append(link)
    return links


def _relabel_interaction_atoms(interaction, mapping):
    """
    Relables the atoms in interaction according to the
    rules defined in mapping.

    Parameters
    ----------
    interaction: `vermouth.molecule.Interaction`
    mapping: `:class:dict`

    Returns
    -------
    interaction: `vermouth.molecule.Interaction`
        the new interaction with updated atoms
    """
    new_atoms = [mapping[atom] for atom in interaction.atoms]
    new_interaction = interaction._replace(atoms=new_atoms)
    return new_interaction


def extract_block(molecule, template_graph, defines):
    """
    Given a `vermouth.molecule` and a `resname`
    extract the information of a block from the
    molecule definition and replace all defines
    if any are found.

    Parameters
    ----------
    molecule:  :class:vermouth.molecule.Molecule
    template_graph: :class:`nx.Graph`
        the graph of the template reisdue
    defines:   dict
      dict of type define: value

    Returns
    -------
    :class:vermouth.molecule.Block
    """
    block = vermouth.molecule.Block()

    # select all nodes with the same first resid and
    # make sure the block node labels are atomnames
    # also build a correspondance dict between node
    # label in the molecule and in the block for
    # relabeling the interactions
    mapping = {}
    for node in template_graph.nodes:
        attr_dict = molecule.nodes[node]
        block.add_node(attr_dict["atomname"], **attr_dict)
        mapping[node] = attr_dict["atomname"]

    for inter_type in molecule.interactions:
        had_interactions = []
        versions = {}
        for interaction in molecule.interactions[inter_type]:
            if all(atom in mapping for atom in interaction.atoms):
                interaction = replace_defined_interaction(interaction, defines)
                interaction = _relabel_interaction_atoms(interaction, mapping)
                if tuple(interaction.atoms) in had_interactions:
                    n = versions.get(tuple(interaction.atoms), 1) + 1
                    meta = {"version": n}
                    versions[tuple(interaction.atoms)] = n
                    interaction.meta.update(meta)
                block.interactions[inter_type].append(interaction)
                had_interactions.append(tuple(interaction.atoms))

    for inter_type in ["bonds", "constraints", "virtual_sitesn",
                       "virtual_sites2", "virtual_sites3", "virtual_sites4"]:
        block.make_edges_from_interaction_type(inter_type)

    return block

def find_termini_mods(meta_molecule, molecule, force_field):
    """
    Terminii are a bit special in the sense that they are often
    different from a repeat unit of the polymer in the polymer.
    """
    terminal_nodes = [ node for node in meta_molecule.nodes if meta_molecule.degree(node) == 1 ]
    for meta_node in terminal_nodes:
        # get the node that is next to the terminal; by definition
        # it can only be one neighbor
        neigh_node = next(nx.neighbors(meta_molecule, meta_node))

        # some useful info
        neigh_resname = meta_molecule.nodes[neigh_node]['resname']
        resids = [meta_molecule.nodes[neigh_node]['resid'],
                  meta_molecule.nodes[meta_node]['resid']]
        ref_block = force_field.blocks[neigh_resname]
        target_block = meta_molecule.nodes[neigh_node]['graph']

        # find different properties
        replace_dict = defaultdict(dict)
        for node in target_block.nodes:
            target_attrs = target_block.nodes[node]
            ref_attrs = ref_block.nodes[target_attrs['atomname']]
            for attr in ['atype', 'mass']:
                if target_attrs[attr] != ref_attrs[attr]:
                    replace_dict[node][attr] = target_attrs[attr]
        # a little dangerous but mostly ok; if there are no changes to
        # the atoms we can continue
        if len(replace_dict) == 0:
            continue

        # bonded interactions could be different too so we need to check them
        overwrite_inters = defaultdict(list)
        for inter_type in ref_block.interactions:
            for ref_inter in ref_block.interactions[inter_type]:
                for target_inter in target_block.interactions[inter_type]:
                    target_atoms = [target_block.nodes[atom]['atomname'] for atom in target_inter.atoms]
                    if target_atoms == ref_inter.atoms and\
                    target_inter.parameters != ref_inter.parameters:
                         mol_atoms_to_link_atoms, edges, resnames = _extract_edges_from_shortest_path(target_inter.atoms,
                                                                                                      molecule,
                                                                                                      min(resids))
                         #link_to_mol_atoms = {value:key for key, value in mol_atoms_to_link_atoms.items()}
                         link_atoms =  [mol_atoms_to_link_atoms[atom] for atom in target_inter.atoms]
                         link_inter = Interaction(atoms=link_atoms,
                                                  parameters=target_inter.parameters,
                                                   meta={})
                         overwrite_inters[inter_type].append(link_inter)

        # we make a link
        mol_atoms = list(replace_dict.keys()) + list(meta_molecule.nodes[meta_node]['graph'].nodes)
        link = vermouth.molecule.Link()
        mol_to_link, edges, resnames = _extract_edges_from_shortest_path(mol_atoms,
                                                                         molecule,
                                                                         min(resids))
        link_atoms = mol_to_link.values()
        link = vermouth.molecule.Link()
        link.add_nodes_from(link_atoms)
        for node in mol_atoms:
            link.nodes[mol_to_link[node]]['resname'] = molecule.nodes[node]['resname']
            link.nodes[mol_to_link[node]]['replace'] = replace_dict[node]

        force_field.links.append(link)
        for inter_type in overwrite_inters:
            link.interactions[inter_type].append(overwrite_inters)

        edges = find_connecting_edges(meta_molecule, molecule, [meta_node, neigh_node])
        for ndx, jdx in edges:
            link.add_edge(mol_to_link[ndx], mol_to_link[jdx])

    return force_field
