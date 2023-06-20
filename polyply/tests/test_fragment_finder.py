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
"""
Test the fragment finder for itp_to_ff.
"""

import textwrap
import pytest
from pathlib import Path
import numpy as np
import networkx as nx
import vermouth.forcefield
import vermouth.molecule
from vermouth.gmx.itp_read import read_itp
from polyply import TEST_DATA
import polyply.src.meta_molecule
from polyply.src.meta_molecule import (MetaMolecule, Monomer)
import polyply
from collections import defaultdict
import pysmiles

@pytest.mark.parametrize(
    "node1, node2, expected",
    [
        ({"element": "C"}, {"element": "C"}, True),
        ({"element": "H"}, {"element": "O"}, False),
        ({"element": "N"}, {"element": "N"}, True),
        ({"element": "O"}, {"element": "S"}, False),
    ],
)
def test_element_match(node1, node2, expected):
    assert polyply.src.fragment_finder._element_match(node1, node2) == expected

@pytest.mark.parametrize(
    "match_keys, node1, node2, expected",
    [
        (["element"], {"element": "C"}, {"element": "C"}, True),
        (["element"], {"element": "H"}, {"element": "O"}, False),
        (["element", "charge"], {"element": "N", "charge": 0}, {"element": "N", "charge": 1}, False),
        (["element", "charge"], {"element": "O", "charge": -1}, {"element": "O", "charge": -1}, True),
    ],
)
def test_node_match(match_keys, node1, node2, expected):
    # molecule and terminal label don't matter
    frag_finder = polyply.src.fragment_finder.FragmentFinder(None, "ter")
    frag_finder.match_keys = match_keys
    assert frag_finder._node_match(node1, node2) == expected

def find_studs(mol):
    """
    By element find all undersatisfied connections
    at the all-atom level.
    """
    atom_degrees = {"H":1,
                    "C":4,
                    "O":2,
                    "N":3}
    for node in mol.nodes:
        ele = mol.nodes[node]['element']
        if mol.degree(node) != atom_degrees[ele]:
            yield node

def set_mass(mol):
    masses = {"O": 16, "N":14,"C":12,
              "S":32, "H":1}

    for atom in mol.nodes:
        mol.nodes[atom]['mass'] = masses[mol.nodes[atom]['element']]
    return mol

def polymer_from_fragments(fragments, resnames, remove_resid=True):
    """
    Given molecule fragments as smiles
    combine them into different polymer
    molecules.
    """
    fragments_to_mol = []
    frag_mols = []
    frag_graph = pysmiles.read_smiles(fragments[0], explicit_hydrogen=True)
    nx.set_node_attributes(frag_graph, 1, "resid")
    nx.set_node_attributes(frag_graph, resnames[0], "resname")
    frag_mols.append(frag_graph)
    mol = vermouth.Molecule(frag_graph)
    # terminals should have one stud anyways
    prev_stud = next(find_studs(frag_graph))
    fragments_to_mol.append({node: node for node in mol.nodes})
    for resname, smile in zip(resnames[1:], fragments[1:]):
        frag_graph = pysmiles.read_smiles(smile, explicit_hydrogen=True)
        nx.set_node_attributes(frag_graph, resname, "resname")
        frag_mols.append(frag_graph)
        next_mol = vermouth.Molecule(frag_graph)
        correspondance = mol.merge_molecule(next_mol)
        fragments_to_mol.append(correspondance)
        stud_iter = find_studs(frag_graph)
        mol.add_edge(prev_stud, correspondance[next(stud_iter)])

        try:
            prev_stud = correspondance[next(stud_iter)]
        except StopIteration:
            # we're done molecule is complete
            continue
    mol = set_mass(mol)
    if remove_resid:
        nx.set_node_attributes(mol, {node: None for node in mol.nodes} ,"resid")
        nx.set_node_attributes(mol, {node: None for node in mol.nodes} ,"resname")
    return mol, frag_mols, fragments_to_mol

@pytest.mark.parametrize(
    "smiles, resnames",
    [
     # completely defined molecule with two termini
     (["[CH3]", "[CH2]O[CH2]", "[CH3]"], ["CH3", "PEO", "CH3"]),
     # two different termini
     (["[OH][CH2]", "[CH2]O[CH2]", "[CH3]"], ["OH", "PEO", "CH3"]),
     # two different termini with the same repeat unit
     (["[OH][CH2]", "[CH2]O[CH2]","[CH2]O[CH2]", "[CH3]"], ["OH", "PEO", "PEO", "CH3"]),
     # sequence with two monomers and multiple "wrong" matchs
     (["[CH3]", "[CH2][CH][CH][CH2]", "[CH2]O[CH2]", "[CH2][OH]"], ["CH3", "PBD", "PEO", "OH"]),
     # sequence with two monomers, four repeats and multiple "wrong" matchs
     (["[CH3]", "[CH2][CH][CH][CH2]", "[CH2][CH][CH][CH2]", "[CH2][CH][CH][CH2]",
      "[CH2][CH][CH][CH2]", "[CH2]O[CH2]", "[CH2]O[CH2]", "[CH2]O[CH2]", "[CH2]O[CH2]",
      "[CH2][OH]"], ["CH3", "PBE", "PBE", "PBE", "PBE", "PEO", "PEO", "PEO", "PEO", "OH"]),
     # super symmtry - worst case scenario
     (["[CH3]", "[CH2][CH2]", "[CH2][CH2]", "[CH2][CH2]","[CH2][CH2]", "[CH2][CH2]","[CH3]"],
      ["CH3", "PE", "PE", "PE", "PE", "PE", "CH3"]),
    ])
def test_label_fragments(smiles, resnames):
    molecule, frag_mols, fragments_in_mol = polymer_from_fragments(smiles, resnames)
    frag_finder = polyply.src.fragment_finder.FragmentFinder(molecule, "ter")
    unique_fragments = frag_finder.label_fragments_from_graph(frag_mols)
    for resid, (resname, frag_to_mol) in enumerate(zip(resnames, fragments_in_mol), start=1):
        for frag_node, mol_node in frag_to_mol.items():
            assert frag_finder.molecule.nodes[mol_node]['resname'] == resname
            assert frag_finder.molecule.nodes[mol_node]['resid'] == resid

@pytest.mark.parametrize(
    "smiles, resnames, remove, new_name",
    [
     # do not match termini
     (["[CH3]", "[CH2]O[CH2]", "[CH2]O[CH2]", "[CH2]O[CH2]", "[CH3]"],
      ["CH3", "PEO", "PEO", "PEO", "CH3"],
      {1:2, 6:3},
      {1: "PEO", "4": "PEO"},
     ),
     # have dangling atom in center
     (["[CH3]", "[CH2][CH2]", "[CH2][CH2]", "[CH2]O[CH2]", "[CH2][CH2]","[CH2][CH2]", "[CH2][CH2]","[CH3]"],
      ["CH3", "PE", "PE", "PEO", "PE", "PE", "PE", "CH3"],
      {4:5},
      {4:"PE"},
     ),
    ])
def test_label_unmatched_atoms(smiles, resnames, remove, new_name):
    molecule, frag_mols, fragments_in_mol = polymer_from_fragments(smiles, resnames, remove_resid=False)
    nodes_to_label = {}
    max_by_resid = {}

    for node in molecule.nodes:
        resid = molecule.nodes[node]['resid']
        if resid in remove:
            del molecule.nodes[node]['resid']
            del molecule.nodes[node]['resname']
            nodes_to_label[node] = resid
        else:
            if resid in max_by_resid:
                known_atom = node
                max_by_resid[resid] += 1
            else:
                max_by_resid[resid] = 1

    resids = nx.get_node_attributes(molecule, "resid")
    # the frag finder removes resid attributes so we have to later reset them
    frag_finder = polyply.src.fragment_finder.FragmentFinder(molecule, "ter")
    nx.set_node_attributes(frag_finder.molecule, resids, "resid")
    frag_finder.max_by_resid = max_by_resid
    frag_finder.known_atom = known_atom
    frag_finder.label_unmatched_atoms()
    for node, old_id in nodes_to_label.items():
        assert frag_finder.molecule.nodes[node]['resid'] == remove[old_id]
        assert frag_finder.molecule.nodes[node]['resname'] == new_name[old_id]

@pytest.mark.parametrize(
    "smiles, resnames, remove, uni_frags",
    [
     # completely defined molecule with two termini
     (["[CH3]", "[CH2]O[CH2]", "[CH3]"],
      ["CH3", "PEO", "CH3"],
      {},
      {"CH3ter": 0, "PEO": 1}
     ),
     # two different termini
     (["[OH][CH2]", "[CH2]O[CH2]", "[CH3]"],
      ["OH", "PEO", "CH3"],
      {},
      {"OHter": 0, "PEO": 1, "CH3ter": 2}
     ),
     # sequence with two monomers, four repeats and multiple "wrong" matchs
     (["[CH3]", "[CH2][CH][CH][CH2]", "[CH2][CH][CH][CH2]", "[CH2][CH][CH][CH2]",
      "[CH2][CH][CH][CH2]", "[CH2]O[CH2]", "[CH2]O[CH2]", "[CH2]O[CH2]", "[CH2]O[CH2]",
      "[CH2][OH]"],
      ["CH3", "PBE", "PBE", "PBE", "PBE", "PEO", "PEO", "PEO", "PEO", "OH"],
      {},
      {"CH3ter": 0, "PBE": 1, "PEO": 5, "OHter": 9}
     ),
     # super symmtry - worst case scenario
     (["[CH3]", "[CH2][CH2]", "[CH2][CH2]", "[CH2][CH2]","[CH2][CH2]", "[CH2][CH2]","[CH3]"],
      ["CH3", "PE", "PE", "PE", "PE", "PE", "CH3"],
      {},
      {"CH3ter":0, "PE": 1}
     ),
     # different fragments with same resname
     (["[CH3]O[CH2]", "[CH2]O[CH2]", "[CH3]"],
      ["PEO", "PEO", "CH3"],
      {3:2},
      {"PEOter": 0, "PEOter_1": (1,2)}
     ),
     # do not match termini
     (["[CH3]", "[CH2]O[CH2]", "[CH2]O[CH2]", "[CH2]O[CH2]", "[CH3]"],
      ["CH3", "PEO", "PEO", "PEO", "CH3"],
      {5: 4},
      {"CH3ter":0, "PEO": 1, "PEOter": (3, 4)},
     ),
     # have dangling atom in center; this is a bit akward but essentially serves
     # as a guard of having really shitty input
     (["[CH3]", "[CH2][CH2]", "[CH2][CH2]", "[CH2]O[CH2]", "[CH2][CH2]","[CH2][CH2]", "[CH2][CH2]","[CH3]"],
      ["CH3", "PE", "PE", "PEO", "PE", "PE", "PE", "CH3"],
      {4: 3},
      {"CH3ter": 0, "PE": 1, "PEter": (2, 3, 4, 5, 6, 7)},
     ),
    ])
def test_extract_fragments(smiles, resnames, remove, uni_frags):
    molecule, frag_mols, fragments_in_mol = polymer_from_fragments(smiles, resnames, remove_resid=True)
    for node in molecule.nodes:
        resid = molecule.nodes[node]['resid']
        if resid in remove:
            del molecule.nodes[node]['resid']
            del molecule.nodes[node]['resname']

    match_mols = []
    for idx, frag in enumerate(frag_mols):
        if idx not in remove.values():
            match_mols.append(frag)

    frag_finder = polyply.src.fragment_finder.FragmentFinder(molecule, "ter")
    fragments = frag_finder.extract_unique_fragments(match_mols)
    assert len(fragments) == len(uni_frags)
    for resname, graph in fragments.items():
        frag_finder.match_keys = ['element', 'mass', 'resname']
        if type(uni_frags[resname]) == tuple:
           new_smiles = [smiles[idx] for idx in uni_frags[resname]]
           new_resnames = [resnames[idx] for idx in uni_frags[resname]]
           ref, _, _ = polymer_from_fragments(new_smiles, new_resnames)
           nx.set_node_attributes(ref, resname, "resname")
        else:
            ref = frag_mols[uni_frags[resname]]
        # because the terminii are not labelled yet in the fragment
        # graphs used to make the match
        nx.set_node_attributes(ref, resname, "resname")
        assert nx.is_isomorphic(ref, graph, node_match=frag_finder._node_match)
        # make sure all molecule nodes are named correctly
        frag_finder.match_keys = ['atomname', 'resname']
        for node in frag_finder.res_graph:
           resname_mol = frag_finder.res_graph.nodes[node]["resname"]
           if resname == resname_mol:
               target = frag_finder.res_graph.nodes[node]["graph"]
               assert nx.is_isomorphic(target, graph, node_match=frag_finder._node_match)
