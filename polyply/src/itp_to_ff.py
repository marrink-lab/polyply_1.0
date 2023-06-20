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

import itertools
from collections import defaultdict
import numpy as np
import networkx as nx
import pysmiles
import vermouth
from vermouth.forcefield import ForceField
from vermouth.molecule import Interaction
from polyply.src.topology import Topology
from polyply.src.generate_templates import extract_block
from polyply.src.fragment_finder import FragmentFinder
from polyply.src.ffoutput import ForceFieldDirectiveWriter

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
    #edges_for_patterns = defaultdict(list)
    for inter_type in molecule.interactions:
        #print("TYPE", inter_type)
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
            #print(kdx, resnames)
            link_to_mol_atoms = {value:key for key, value in mol_atoms_to_link_atoms.items()}
            link_atoms =  [mol_atoms_to_link_atoms[atom] for atom in interaction.atoms]
            link_inter = Interaction(atoms=link_atoms,
                                     parameters=interaction.parameters,
                                     meta={})
            #print("inter number", kdx)
            # here we deal with filtering redundancy
            if pattern in patterns and inter_type in patterns[pattern]:
                #print(pattern)
           #     if pattern == ((0, 'PEO'), (1, 'PEO')):
           #         print(kdx, link_inter.atoms, patterns[pattern].get(inter_type, []), "\n")

                for other_inter in patterns[pattern].get(inter_type, []):
                    if other_inter.atoms == link_inter.atoms:
                        if  other_inter.parameters == link_inter.parameters:
                            break
                else:
                    patterns[pattern][inter_type].append(link_inter)
                    resnames_for_patterns[pattern].update(resnames)
                    link_atoms_for_patterns[pattern] += link_atoms
            else:
                patterns[pattern][inter_type] = [link_inter]
                resnames_for_patterns[pattern].update(resnames)
                #edges_for_patterns[pattern] += edges
                link_atoms_for_patterns[pattern] += link_atoms
            #print('resnames', resnames_for_patterns[pattern], '\n')
#    for inter in patterns[list(patterns.keys())[0]]['angles']:
#        print(inter)
    # we make new links for each unique interaction per type
    for pattern in patterns:
        link = vermouth.molecule.Link()
        link.add_nodes_from(set(link_atoms_for_patterns[pattern]))
        #link.add_edges_from(edges_for_patterns[pattern])
        resnames = resnames_for_patterns[pattern]
     #   print(resnames)
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
    print(links)
    return links

def equalize_charges(molecule, target_charge=0):
    """
    Make sure that the total charge of molecule is equal to
    the target charge by substracting the differences split
    over all atoms.

    Parameters
    ----------
    molecule: :class:`vermouth.molecule.Molecule`
    target_charge: float
        the charge of the molecule

    Returns
    -------
    molecule
        the molecule with updated charge attribute
    """
    total = nx.get_node_attributes(molecule, "charge")
    diff = (sum(list(total.values())) - target_charge)/len(molecule.nodes)
    for node in molecule.nodes:
        charge = float(molecule.nodes[node]['charge']) - diff
        molecule.nodes[node]['charge'] = charge
    total = nx.get_node_attributes(molecule, "charge")
    return molecule

def handle_chirality(molecule, chiral_centers):
    pass

def hcount(molecule, node):
    hcounter = 0
    for node in molecule.neighbors(node):
        if molecule.nodes[node]["element"] == "H":
            hcounter+= 1
    return hcounter

def itp_to_ff(itppath, fragment_smiles, resnames, term_prefix, outpath, charge=0):
    """
    Main executable for itp to ff tool.
    """
    # read the target itp-file
    top = Topology.from_gmx_topfile(itppath, name="test")
    mol = top.molecules[0].molecule
    mol = equalize_charges(mol, target_charge=charge)

    # read the target fragments and convert to graph
    fragment_graphs = []
    for resname, smile in zip(resnames, fragment_smiles):
        fragment_graph = pysmiles.read_smiles(smile, explicit_hydrogen=True)
        nx.set_node_attributes(fragment_graph, resname, "resname")
        fragment_graphs.append(fragment_graph)

    # identify and extract all unique fragments
    unique_fragments = FragmentFinder(mol, prefix=term_prefix).extract_unique_fragments(fragment_graphs)
    force_field = ForceField("new")
    for name, fragment in unique_fragments.items():
        new_block = extract_block(mol, list(fragment.nodes), defines={})
        nx.set_node_attributes(new_block, 1, "resid")
        new_block.nrexcl = mol.nrexcl
        force_field.blocks[name] = new_block

#    for node in mol.nodes:
#        print(mol.nodes[node])

    force_field.links = extract_links(mol)

    with open(outpath, "w") as filehandle:
        ForceFieldDirectiveWriter(forcefield=force_field, stream=filehandle).write()
