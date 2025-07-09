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
Test that force field files are properly read.
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
from .example_fixtures import example_meta_molecule
from polyply.src.topology import Topology

class TestPolyply:
    @staticmethod
    def test_add_monomer():
        ff = vermouth.forcefield.ForceField(name='test_ff')
        meta_mol = MetaMolecule(name="test", force_field=ff)
        meta_mol.add_monomer(0,"PEO",[])
        meta_mol.add_monomer(1,"PEO",[(1, 0)])

        assert nx.get_node_attributes(meta_mol, "resname") == {0: 'PEO', 1: 'PEO'}
        assert list(meta_mol.nodes) == [0, 1]
        assert list(meta_mol.edges) == [(0, 1)]
        assert meta_mol.nodes[0]["build"]
        assert meta_mol.nodes[1]["build"]
        assert meta_mol.nodes[0]["backmap"]
        assert meta_mol.nodes[1]["backmap"]

    @staticmethod
    def test_add_monomer_fail():
        ff = vermouth.forcefield.ForceField(name='test_ff')
        meta_mol = MetaMolecule(name="test", force_field=ff)
        meta_mol.add_monomer(0,"PEO",[])
        with pytest.raises(IOError):
            meta_mol.add_monomer(1,"PEO",[(1, 8)])

    @staticmethod
    def test_get_edge_resname():
        ff = vermouth.forcefield.ForceField(name='test_ff')
        meta_mol = MetaMolecule(name="test", force_field=ff)
        meta_mol.add_monomer(0,"PEO",[])
        meta_mol.add_monomer(1,"PEO",[(1, 0)])
        name = meta_mol.get_edge_resname((1, 0))
        assert name == ["PEO", "PEO"]

    @staticmethod
    @pytest.mark.parametrize('monomers, edges, nodes, attrs', (
        # multiple blocks from single monomer
          ([Monomer(resname="PEO", n_blocks=2)],
           [(0,1)],
           [0,1],
           {0: 'PEO', 1: 'PEO'}
          ),
        # two blocks from two monomers
          ([Monomer(resname="PEO", n_blocks=1), Monomer(resname="PS",n_blocks=1)],
           [(0,1)],
           [0,1],
           {0: 'PEO', 1: 'PS'}
           ),
        # multiple blocks from two monomers
          ([Monomer(resname="PEO", n_blocks=2), Monomer(resname="PS",n_blocks=2)],
           [(0,1),(1,2),(2,3)],
           [0,1,2,3],
           {0: 'PEO', 1: 'PEO', 2: 'PS', 3: 'PS'}
         )))
    def test_from_monomer_seq_linear(monomers, edges, nodes, attrs):
        ff = vermouth.forcefield.ForceField(name='test_ff')
        name = "test"
        meta_mol = MetaMolecule.from_monomer_seq_linear(ff, monomers, name)

        assert nx.get_node_attributes(meta_mol, "resname") == attrs
        assert list(meta_mol.nodes) == nodes
        assert list(meta_mol.edges) == edges

    @staticmethod
    @pytest.mark.parametrize('cgsmiles_str, edges, nodes, attrs', (
        # multiple blocks from single monomer
          ("{[#PEO][#PEO]}",
           [(0,1)],
           [0,1],
           {0: 'PEO', 1: 'PEO'}
          ),
        # two blocks from two monomers
          ("{[#PEO][#PS]}",
           [(0,1)],
           [0,1],
           {0: 'PEO', 1: 'PS'}
           ),
        # use multilevel descript
          ("{[#PU]|2}.{#PU=[>][#PEO][#MDI][<]}",
           [(0,1), (0, 3), (2, 3)],
           [0, 1, 2, 3],
           {0: 'PEO', 1: 'MDI', 2: 'PEO', 3: 'MDI'}
           ),
        # branched monomers
          ("{[#PMA][#PMA]([#PEO])[#PMA]}",
           [(0,1),(1,2),(1,3)],
           [0,1,2,3],
           {0: 'PMA', 1: 'PMA', 2: 'PEO', 3: 'PMA'}
         )))
    def test_from_cgsmiles_str(cgsmiles_str, edges, nodes, attrs):
        ff = vermouth.forcefield.ForceField(name='test_ff')
        name = "test"
        meta_mol = MetaMolecule.from_cgsmiles_str(ff, cgsmiles_str, name)

        assert nx.get_node_attributes(meta_mol, "resname") == attrs
        assert list(meta_mol.nodes) == nodes
        assert list(meta_mol.edges) == edges

    @staticmethod
    def test_from_cgsmiles_seq_ony():
        ff = vermouth.forcefield.ForceField(name='test_ff')
        name = "test"
        cgsmiles_str = "{[#A][#B][#A]}.{#A=[>][#BB][#BB1][<]([#SC1][#SC2]),#B=[>][#BB][#BB1]([#SC1][#SC2])[#BB2][<]}"
        meta_mol = MetaMolecule.from_cgsmiles_str(ff,
                                                  cgsmiles_str,
                                                  name,
                                                  seq_only=False,
                                                  all_atom=False)
        attrs = {0: "A", 1: "B", 2: "A"}
        assert nx.get_node_attributes(meta_mol, "resname") == attrs
        assert list(meta_mol.nodes) == [0, 1, 2]
        assert list(meta_mol.edges) == [(0, 1), (1, 2)]
        molecule = meta_mol.molecule
        attrs = {0: "A", 1: "A", 2: "A", 3: "A",
                4: "B", 5: "B", 6: "B", 7: "B", 8: "B",
                9: "A", 10: "A", 11: "A", 12: "A"}
        assert nx.get_node_attributes(molecule, "resname") == attrs
        assert list(molecule.edges) == [(0, 1), (0, 8), (1, 2), (2, 3), (4, 5),
                                        (4, 10), (5, 6), (5, 8), (6, 7), (9, 10),
                                        (10, 11), (11, 12)]
    @staticmethod
    @pytest.mark.parametrize('file_name, edges, nodes, attrs', (
        # multiple blocks from single monomer
          (TEST_DATA / "json" / "linear.json",
           [(0,1), (1,2)],
           [0,1,2],
           {0: 'PEO', 1: 'PEO', 2: 'PEO'}
          ),
       # two blocks from two monomers
         (TEST_DATA / "json" / "single_branch.json",
          [(1,2),(2,3),(3,4),(2,5)],
          [1,2,3,4,5],
          {1: 'PEO', 2: 'PEO', 3: 'PS', 4: 'PS', 5:'PEO'}
          ),
       # two blocks from two monomers
         (TEST_DATA / "json" / "double_branch.json",
          [(1,2),(2,3),(2,4),(4,5),(5,6),(5,8),(6,7)],
          [1,2,3,4,5,6,7,8],
          {1: 'PEO', 2: 'PEO', 3: 'PS', 4: 'PEO', 5:'PEO', 6: 'PS',
           7: 'PS', 8: 'PEO'}
          ),
       # Hyperbranched
         (TEST_DATA / "json" / "hyperbranched.json",
          [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 6), (2, 7), (3, 8), (3, 9),
           (4, 10), (4, 11), (5, 12), (5, 13)],
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
          {0: 'N1', 1: 'N2', 2: 'N3', 3: 'N3', 4: 'N2', 5: 'N3', 6: 'N3',
           7: 'N1', 8: 'N2', 9: 'N3', 10: 'N3', 11: 'N2', 12: 'N3', 13: 'N3'}
          ),
        # check that ordering is restored
          (TEST_DATA / "json" / "linear_rev.json",
           [(0, 1), (1, 2)],
           [0, 1, 2],
           {0: 'PEO', 1: 'PEO', 2: 'PEO'}
          ),
           ))
    def test_from_seq_file(file_name, edges, nodes, attrs):
        ff = vermouth.forcefield.ForceField(name='test_ff')
        name = "test"
        meta_mol = MetaMolecule.from_sequence_file(ff, file_name, name)

        assert len(nx.get_node_attributes(meta_mol, "resid")) == len(nodes)
        assert set(meta_mol.nodes) == set(nodes)
        assert set(meta_mol.edges) == set(edges)

    @staticmethod
    def test_resid_assignment_error():
        ff = vermouth.forcefield.ForceField(name='test_ff')
        plain_graph = nx.Graph()
        plain_graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
        with pytest.raises(IOError):
            MetaMolecule(plain_graph, force_field=ff, mol_name="test")

    @staticmethod
    def test_from_itp():
       file_name = TEST_DATA / "itp" / "PEO.itp"
       edges = [(0,1), (1,2)]
       nodes = [0, 1, 2]
       attrs = {0: 'PEO', 1: 'PEO', 2: 'PEO'}

       ff = vermouth.forcefield.ForceField(name='test_ff')
       name = "PEO"
       meta_mol = MetaMolecule.from_itp(ff, file_name, name)

       assert set(meta_mol.nodes) == set(nodes)
       assert set(meta_mol.edges) == set(edges)

    @staticmethod
    def test_from_block():
       file_name = TEST_DATA / "itp" / "PEO.itp"
       edges = [(0,1), (1,2)]
       nodes = [0, 1, 2]
       attrs = {0: 'PEO', 1: 'PEO', 2: 'PEO'}

       ff = vermouth.forcefield.ForceField(name='test_ff')
       name = "PEO"
       with open(file_name, "r") as _file:
            lines = _file.readlines()
       read_itp(lines, ff)
       meta_mol = MetaMolecule.from_block(ff, name)

       assert set(meta_mol.nodes) == set(nodes)
       assert set(meta_mol.edges) == set(edges)

@pytest.mark.parametrize('split_pattern, expct_high_res, expct_low_res, edges, resid_low, resid_high', (
    # split single residues of one type into two
    (
    ["B:B1-BB,BB1,BB2:B2-SC1,SC2"],
    {0: "A", 1: "A", 2: "A", 3: "A", 4: "B1", 5: "B1", 6: "B1", 7: "B2", 8: "B2",
     9: "A", 10: "A", 11: "A", 12: "A"},
    {0: "A", 1: "B1", 2: "B2", 3: "A"},
    [(0, 1), (1, 3), (1, 2)],
    {0: 0, 1: 1, 2: 2, 3: 3},
    {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2,
     9: 3, 10: 3, 11: 3, 12: 3},
    ),
    # split two residues of one type into two
    (
    ["A:A1-BB,BB1:A2-SC1,SC2"],
    {0: "A1", 1: "A1", 2: "A2", 3: "A2", 4: "B", 5: "B", 6: "B", 7: "B", 8: "B",
     9: "A1", 10: "A1", 11: "A2", 12: "A2"},
    {0: "A1", 1: "A2", 2: "B", 3: "A1", 4: "A2"},
    [(0, 1), (0, 2), (2, 3), (3, 4)],
    {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
    {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2,
     9: 3, 10: 3, 11: 4, 12: 4},
    ),
    # split all residues
    (
    ["A:A1-BB,BB1:A2-SC1,SC2", "B:B1-BB,BB1,BB2:B2-SC1,SC2"],
    {0: "A1", 1: "A1", 2: "A2", 3: "A2", 4: "B1", 5: "B1", 6: "B1", 7: "B2", 8: "B2",
     9: "A1", 10: "A1", 11: "A2", 12: "A2"},
    {0: "A1", 1: "A2", 2: "B1", 3: "B2", 4: "A1", 5: "A2"},
    [(0, 1), (0, 2), (2, 3), (2, 4), (4, 5)],
    {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3,
     9: 4, 10: 4, 11: 5, 12: 5},
    ),
    ))
def test_split_residue(example_meta_molecule,
                       split_pattern,
                       expct_high_res,
                       expct_low_res,
                       edges,
                       resid_low,
                       resid_high):

    example_meta_molecule.split_residue(split_pattern)
    new_resnames_high = nx.get_node_attributes(example_meta_molecule.molecule, "resname")
    assert new_resnames_high == expct_high_res
    new_resnames_low = nx.get_node_attributes(example_meta_molecule, "resname")
    assert new_resnames_low == expct_low_res
    new_resid_low = nx.get_node_attributes(example_meta_molecule, "resid")
    assert new_resid_low == resid_low
    new_resid_high = nx.get_node_attributes(example_meta_molecule.molecule, "resid")
    assert new_resid_high == resid_high

    for node_a, node_b in edges:
        assert example_meta_molecule.has_edge(node_a, node_b)

def test_split_residue_err(example_meta_molecule):
    split_pattern = ["A:A1-BB,BB1:A2-BB1,SC1,SC2"]
    with pytest.raises(IOError):
         example_meta_molecule.split_residue(split_pattern)

def test_unkown_fromat_error():
    with pytest.raises(IOError):
        ff = vermouth.forcefield.ForceField(name='test_ff')
        test_path = Path("random_file.extension")
        MetaMolecule.from_sequence_file(force_field=ff,
                                        file_path=test_path,
                                        mol_name="test")


@pytest.mark.parametrize('cgs, new_cgs, expct_high_res, expct_low_res, edges, resid_low, resid_high, all_atom, weights', (
    # split single residues of one type into three
    ("{[#A]}.{#A=[#BB]([#SC1][#SC1])[#BB]([#SC1][#SC1])[#BB]([#SC1][#SC1])}",
     "{[#Q]|3}.{#Q=[$][#BB][$]([#SC1][#SC1])}}",
     {0: "Q", 1: "Q", 2: "Q", 3: "Q", 4:"Q", 5:"Q", 6:"Q", 7:"Q", 8:"Q"},
     {0: "Q", 1: "Q", 2: "Q"},
    [(0, 1), (1, 2)],
     {0: 0, 1: 1, 2: 2},
     {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2},
     False,
     None),
    # group three residues into one
    ("{[#Q]|3}.{#Q=[$][#BB][$][#SC1][#SC1]}}",
     "{[#A]}.{#A=[#BB]([#SC1][#SC1])[#BB]([#SC1][#SC1])[#BB]([#SC1][#SC1])}",
     {0: "A", 1: "A", 2: "A", 3: "A", 4: "A", 5: "A", 6: "A", 7: "A", 8: "A"},
     {0: "A"},
    [],
     {0: 0},
     {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6:0, 7:0, 8:0},
     False,
     None),
    # an all atom molecule split
    ("{[#DXE]}.{#DXE=COCCOC}",
     "{[#EO][#EO]}.{#EO=[$]COC[$]}",
     {0:"EO", 1: "EO", 2: "EO", 3: "EO", 4: "EO", 5: "EO", 6: "EO", 7: "EO",
      8: "EO", 9: "EO", 10: "EO" , 11: "EO", 12: "EO", 13: "EO", 14: "EO", 15: "EO"},
     {0: "EO", 1: "EO"},
     [(0, 1)],
     {0: 0, 1: 1},
     {0:0, 1: 0, 2: 0, 3: 1, 4:1, 5:1, 6:0, 7:0,
      8:0, 9:0, 10:0 , 11:1, 12: 1, 13:1, 14: 1, 15: 1},
     True,
     {"H": {"mass": 1}, "O": {"mass": 16}, "C": {"mass": 12}})
))
def test_relabel_from_cgsmiles_str(cgs,
                                   new_cgs,
                                   expct_high_res,
                                   expct_low_res,
                                   edges,
                                   resid_low,
                                   resid_high,
                                   all_atom,
                                   weights):
    mol = MetaMolecule.from_cgsmiles_str([],
                                         cgs,
                                         "test",
                                         seq_only=False,
                                         all_atom=all_atom)
    top = None
    if weights:
        top = Topology(force_field=None)
        top.atom_types = weights
        ele = nx.get_node_attributes(mol.molecule, "element")
        nx.set_node_attributes(mol.molecule, ele, "atype")

    mol.relabel_from_cgsmiles_str(new_cgs, topology=top, all_atom=all_atom)

    new_resnames_high = nx.get_node_attributes(mol.molecule, "resname")
    assert new_resnames_high == expct_high_res
    new_resnames_low = nx.get_node_attributes(mol, "resname")
    assert new_resnames_low == expct_low_res
    new_resid_low = nx.get_node_attributes(mol, "resid")
    assert new_resid_low == resid_low
    new_resid_high = nx.get_node_attributes(mol.molecule, "resid")
    assert new_resid_high == resid_high

    for node_a, node_b in edges:
        assert mol.has_edge(node_a, node_b)
