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
import numpy as np
import networkx as nx
import vermouth.forcefield
import vermouth.molecule
import polyply.src.meta_molecule
import polyply.src.map_to_molecule
import polyply.src.polyply_parser
from polyply.src.meta_molecule import (MetaMolecule, Monomer)
from vermouth.molecule import Interaction

@pytest.mark.parametrize('lines, monomers, bonds, edges, nnodes, from_itp', (
    # single block
    ("""
    [ moleculetype ]
    ; name nexcl.
    PEO         1
    ;
    [ atoms ]
    1  SN1a    1   PEO   CO1  1   0.000  45
    2  SN1a    1   PEO   CO2  1   0.000  45
    3  SN1a    1   PEO   CO3  1   0.000  45
    4  SN1a    1   PEO   CO4  1   0.000  45
    [ bonds ]
    ; back bone bonds
    1  2   1   0.37 7000
    2  3   1   0.37 8000
    3  4   1   0.37 9000
    """,
    ["PEO", "PEO"],
    [Interaction(atoms=(0, 1), parameters=['1', '0.37', '7000'], meta={}),
     Interaction(atoms=(1, 2), parameters=['1', '0.37', '8000'], meta={}),
     Interaction(atoms=(2, 3), parameters=['1', '0.37', '9000'], meta={}),
     Interaction(atoms=(4, 5), parameters=['1', '0.37', '7000'], meta={}),
     Interaction(atoms=(5, 6), parameters=['1', '0.37', '8000'], meta={}),
     Interaction(atoms=(6, 7), parameters=['1', '0.37', '9000'], meta={})],
    [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)],
    8,
    {}),
    # test multiblock from-itp
    ("""
    [ moleculetype ]
    ; name nexcl.
    PEO         1
    ;
    [ atoms ]
    1  SN1a    1   PEO   CO1  1   0.000  45
    [ moleculetype ]
    ; name nexcl.
    MIX         1
    ;
    [ atoms ]
    1  SN1a    1   R1   C1  1   0.000  45
    2  SN1a    1   R1   C2  1   0.000  45
    3  SC1     2   R2   C1  2   0.000  45
    4  SC1     2   R2   C2  2   0.000  45
    [ bonds ]
    ; back bone bonds
    1  2   1   0.37 7000
    2  3   1   0.37 7000
    3  4   1   0.37 7000
    """,
    ["R1", "R2", "PEO"],
    [Interaction(atoms=(0, 1), parameters=['1', '0.37', '7000'], meta={}),
     Interaction(atoms=(1, 2), parameters=['1', '0.37', '7000'], meta={}),
     Interaction(atoms=(2, 3), parameters=['1', '0.37', '7000'], meta={})],
    [(0, 1), (1, 2), (2, 3)],
    5,
    {0: "MIX", 1: "MIX"}),
    # test multiblock from-itp reverse
    ("""
    [ moleculetype ]
    ; name nexcl.
    PEO         1
    ;
    [ atoms ]
    1  SN1a    1   PEO   CO1  1   0.000  45
    [ moleculetype ]
    ; name nexcl.
    MIX         1
    ;
    [ atoms ]
    1  SN1a    1   R1   C1  1   0.000  45
    2  SN1a    1   R1   C2  1   0.000  45
    3  SC1     2   R2   C1  2   0.000  45
    4  SC1     2   R2   C2  2   0.000  45
    [ bonds ]
    ; back bone bonds
    1  2   1   0.37 7000
    2  3   1   0.37 7000
    3  4   1   0.37 7000
    """,
    ["PEO", "R1", "R2"],
    [Interaction(atoms=(1, 2), parameters=['1', '0.37', '7000'], meta={}),
     Interaction(atoms=(2, 3), parameters=['1', '0.37', '7000'], meta={}),
     Interaction(atoms=(3, 4), parameters=['1', '0.37', '7000'], meta={})],
    [(1, 2), (2, 3), (3, 4)],
    5,
    {1: "MIX", 2: "MIX"}),
    # test two multiblock from-itp
    ("""
    [ moleculetype ]
    ; name nexcl.
    PEO         1
    ;
    [ atoms ]
    1  SN1a    1   PEO   CO1  1   0.000  45
    [ moleculetype ]
    ; name nexcl.
    MIX         1
    ;
    [ atoms ]
    1  SN1a    1   R1   C1  1   0.000  45
    2  SN1a    1   R1   C2  1   0.000  45
    3  SC1     2   R2   C1  2   0.000  45
    4  SC1     2   R2   C2  2   0.000  45
    [ bonds ]
    ; back bone bonds
    1  2   1   0.37 7000
    2  3   1   0.37 7000
    3  4   1   0.37 8000
    """,
    ["R1", "R2", "PEO", "R1", "R2"],
    [Interaction(atoms=(0, 1), parameters=['1', '0.37', '7000'], meta={}),
     Interaction(atoms=(1, 2), parameters=['1', '0.37', '7000'], meta={}),
     Interaction(atoms=(2, 3), parameters=['1', '0.37', '8000'], meta={}),
     Interaction(atoms=(5, 6), parameters=['1', '0.37', '7000'], meta={}),
     Interaction(atoms=(6, 7), parameters=['1', '0.37', '7000'], meta={}),
     Interaction(atoms=(7, 8), parameters=['1', '0.37', '8000'], meta={})],
    [(0, 1), (1, 2), (2, 3), (5, 6), (6, 7), (7, 8)],
    9,
    {0: "MIX", 1: "MIX", 3: "MIX", 4:"MIX"}),
    # test two consecutive multiblock fragments
    ("""
    [ moleculetype ]
    ; name nexcl.
    MIX         1
    ;
    [ atoms ]
    1  SN1a    1   R1   C1  1   0.000  45
    2  SN1a    1   R1   C2  1   0.000  45
    3  SC1     2   R2   C1  2   0.000  45
    4  SC1     2   R2   C2  2   0.000  45
    [ bonds ]
    ; back bone bonds
    1  2   1   0.37 7000
    2  3   1   0.37 7000
    3  4   1   0.37 8000
    """,
    ["R1", "R2", "R1", "R2"],
    [Interaction(atoms=(0, 1), parameters=['1', '0.37', '7000'], meta={}),
     Interaction(atoms=(1, 2), parameters=['1', '0.37', '7000'], meta={}),
     Interaction(atoms=(2, 3), parameters=['1', '0.37', '8000'], meta={}),
     Interaction(atoms=(4, 5), parameters=['1', '0.37', '7000'], meta={}),
     Interaction(atoms=(5, 6), parameters=['1', '0.37', '7000'], meta={}),
     Interaction(atoms=(6, 7), parameters=['1', '0.37', '8000'], meta={})],
    [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)],
    8,
    {0: "MIX", 1: "MIX", 2: "MIX", 3:"MIX"}),
    # test two consecutive multiblock fragments with single residues
    # and different mol-name
    ("""
    [ moleculetype ]
    ; name nexcl.
    OTR         1
    ;
    [ atoms ]
    1  SN1a    1   R2   C1  1   0.000  45
    2  SN1a    1   R2   C2  1   0.000  45
    [ bonds ]
    ; back bone bonds
    1  2   1   0.4 7000
    [ moleculetype ]
    ; name nexcl.
    MIX         1
    ;
    [ atoms ]
    1  SN1a    1   R1   C1  1   0.000  45
    2  SN1a    1   R1   C2  1   0.000  45
    [ bonds ]
    ; back bone bonds
    1  2   1   0.37 7000
    """,
    ["R1", "R2"],
    [Interaction(atoms=(0, 1), parameters=['1', '0.37', '7000'], meta={}),
     Interaction(atoms=(2, 3), parameters=['1', '0.4', '7000'], meta={}),],
    [(0, 1), (2, 3),],
    4,
    {0: "MIX", 1: "OTR"})
    ))
def test_multiresidue_block(lines, monomers, bonds, edges, nnodes, from_itp):
    """
    Test multiresidue blocks are correctly identified and parameters from
    the blocks added at the correct place in the fine-grained molecule.
    """
    lines = textwrap.dedent(lines).splitlines()
    ff = vermouth.forcefield.ForceField(name='test_ff')
    polyply.src.polyply_parser.read_polyply(lines, ff)
    # build the meta-molecule
    meta_mol = MetaMolecule(name="test", force_field=ff)
    meta_mol.add_monomer(0, monomers[0], [])
    for node, monomer in enumerate(monomers[1:]):
        meta_mol.add_monomer(node+1, monomer, [(node, node+1)])
    nx.set_node_attributes(meta_mol, from_itp, "from_itp")
    # map to molecule
    new_meta_mol = polyply.src.map_to_molecule.MapToMolecule(ff).run_molecule(meta_mol)
    # check that the disconnected molecule is properly done
    for node in new_meta_mol.nodes:
        assert len(new_meta_mol.nodes[node]['graph'].nodes) != 0

    assert len(new_meta_mol.molecule.nodes) == nnodes
    assert list(new_meta_mol.molecule.edges) == edges
    assert new_meta_mol.molecule.interactions['bonds'] == bonds

def test_multi_excl_block():
    lines = """
    [ moleculetype ]
    ; name nexcl.
    PEO         1
    ;
    [ atoms ]
    1  SN1a    1   PEO   CO1  1   0.000  45
    [ moleculetype ]
    ; name nexcl.
    MIX         2
    ;
    [ atoms ]
    1  SN1a    1   MIX  C1  1   0.000  45
    2  SN1a    1   MIX  C2  1   0.000  45
    3  SC1     1   MIX  C1  2   0.000  45
    4  SC1     1   MIX  C2  2   0.000  45
    [ bonds ]
    ; back bone bonds
    1  2   1   0.37 7000
    2  3   1   0.37 7000
    3  4   1   0.37 7000
    """
    lines = textwrap.dedent(lines).splitlines()
    ff = vermouth.forcefield.ForceField(name='test_ff')
    polyply.src.polyply_parser.read_polyply(lines, ff)
    meta_mol = MetaMolecule(name="test", force_field=ff)
    meta_mol.add_monomer(0,"PEO",[])
    meta_mol.add_monomer(1,"MIX",[(1,0)])

    new_meta_mol = polyply.src.map_to_molecule.MapToMolecule(ff).run_molecule(meta_mol)
    assert nx.get_node_attributes(new_meta_mol.molecule, "exclude") == {0: 1, 1: 2, 2: 2, 3: 2, 4: 2}


@pytest.mark.parametrize('lines, monomers, bonds, edges, nnodes, from_itp', (
    # linear multiblock last
    ("""
    [ moleculetype ]
    ; name nexcl.
    PEO         1
    ;
    [ atoms ]
    1  SN1a    1   PEO   CO1  1   0.000  45
    [ moleculetype ]
    ; name nexcl.
    MIX         1
    ;
    [ atoms ]
    1  SN1a    1   R1   C1  1   0.000  45
    2  SN1a    1   R1   C2  1   0.000  45
    3  SC1     2   R2   C1  2   0.000  45
    4  SC1     2   R2   C2  2   0.000  45
    [ bonds ]
    ; back bone bonds
    1  2   1   0.37 7000
    2  3   1   0.37 7000
    3  4   1   0.37 7000
    """,
    ["PEO", "MIX"],
    [Interaction(atoms=(1, 2), parameters=['1', '0.37', '7000'], meta={}),
     Interaction(atoms=(2, 3), parameters=['1', '0.37', '7000'], meta={}),
     Interaction(atoms=(3, 4), parameters=['1', '0.37', '7000'], meta={})],
    [(1, 2), (2, 3), (3, 4)],
    5,
    {}),
   # linear multi-block second
    ("""
    [ moleculetype ]
    ; name nexcl.
    PEO         1
    ;
    [ atoms ]
    1  SN1a    1   PEO   CO1  1   0.000  45
    [ moleculetype ]
    ; name nexcl.
    MIX         1
    ;
    [ atoms ]
    1  SN1a    1   R1   C1  1   0.000  45
    2  SN1a    1   R1   C2  1   0.000  45
    3  SC1     2   R2   C1  2   0.000  45
    4  SC1     2   R2   C2  2   0.000  45
    [ bonds ]
    ; back bone bonds
    1  2   1   0.37 7000
    2  3   1   0.37 7000
    3  4   1   0.37 7000
    """,
    ["MIX", "PEO"],
    [Interaction(atoms=(0, 1), parameters=['1', '0.37', '7000'], meta={}),
     Interaction(atoms=(1, 2), parameters=['1', '0.37', '7000'], meta={}),
     Interaction(atoms=(2, 3), parameters=['1', '0.37', '7000'], meta={})],
    [(0, 1), (1, 2), (2, 3)],
    5,
    {}),
    # linear branched
    ("""
    [ moleculetype ]
    ; name nexcl.
    PEO         1
    ;
    [ atoms ]
    1  SN1a    1   PEO   CO1  1   0.000  45
    [ moleculetype ]
    ; name nexcl.
    MIX         1
    ;
    [ atoms ]
    1  SN1a    1   R1   C1  1   0.000  45
    2  SC1     1   R1   C1  2   0.000  45
    3  SN1a    2   R2   C1  1   0.000  45
    4  SC1     2   R2   C1  2   0.000  45
    5  SN1a    3   R3   C1  1   0.000  45
    6  SC1     3   R3   C1  2   0.000  45
    7  SN1a    4   R2   C1  1   0.000  45
    8  SC1     4   R2   C1  2   0.000  45
    [ bonds ]
    ; back bone bonds
    1  2   1   0.44 7000
    1  3   1   0.37 7000
    3  4   1   0.44 7000
    3  5   1   0.37 7000
    3  7   1   0.37 7000
    5  6   1   0.44 7000
    7  8   1   0.44 7000
    """,
    ["PEO", "MIX"],
    [Interaction(atoms=(1, 2), parameters=['1', '0.44', '7000'], meta={}),
     Interaction(atoms=(1, 3), parameters=['1', '0.37', '7000'], meta={}),
     Interaction(atoms=(3, 4), parameters=['1', '0.44', '7000'], meta={}),
     Interaction(atoms=(3, 5), parameters=['1', '0.37', '7000'], meta={}),
     Interaction(atoms=(3, 7), parameters=['1', '0.37', '7000'], meta={}),
     Interaction(atoms=(5, 6), parameters=['1', '0.44', '7000'], meta={}),
     Interaction(atoms=(7, 8), parameters=['1', '0.44', '7000'], meta={})],
    [(1, 2), (1, 3), (3, 4), (3, 5), (3, 7), (5, 6), (7, 8)],
    9,
    {}),))
def test_riase_multiresidue_error(lines, monomers, bonds, edges, nnodes, from_itp):
    """
    When a single node in the meta_molecule corresponds to a multiresidue block
    but is not labelled with from_itp an error should be raised.
    """
    lines = textwrap.dedent(lines).splitlines()
    ff = vermouth.forcefield.ForceField(name='test_ff')
    polyply.src.polyply_parser.read_polyply(lines, ff)
    # build the meta-molecule
    meta_mol = MetaMolecule(name="test", force_field=ff)
    meta_mol.add_monomer(0, monomers[0], [])
    for node, monomer in enumerate(monomers[1:]):
        meta_mol.add_monomer(node+1, monomer, [(node, node+1)])
    nx.set_node_attributes(meta_mol, from_itp, "from_itp")
    with pytest.raises(IOError):
        new_meta_mol = polyply.src.map_to_molecule.MapToMolecule(ff).run_molecule(meta_mol)

@pytest.mark.parametrize('lines, monomers, from_itp', (
    # we are missing one residue in the multiblock graph
    ("""
    [ moleculetype ]
    ; name nexcl.
    MIX         1
    ;
    [ atoms ]
    1  SN1a    1   R1   C1  1   0.000  45
    2  SN1a    1   R1   C2  1   0.000  45
    3  SC1     2   R2   C1  2   0.000  45
    4  SC1     2   R2   C2  2   0.000  45
    [ bonds ]
    ; back bone bonds
    1  2   1   0.37 7000
    2  3   1   0.37 7000
    3  4   1   0.37 8000
    """,
    ["R1", "R2", "R1"],
    {0: "MIX", 1: "MIX", 2: "MIX"}),
    # we have a residue too many in the residue graph provided
    ("""
    [ moleculetype ]
    ; name nexcl.
    MIX         1
    ;
    [ atoms ]
    1  SN1a    1   R1   C1  1   0.000  45
    2  SN1a    1   R1   C2  1   0.000  45
    3  SC1     2   R2   C1  2   0.000  45
    4  SC1     2   R2   C2  2   0.000  45
    [ bonds ]
    ; back bone bonds
    1  2   1   0.37 7000
    2  3   1   0.37 7000
    3  4   1   0.37 8000
    """,
    ["R1", "R2", "R3", "R1", "R2"],
    {0: "MIX", 1: "MIX", 2: "MIX", 3:"MIX"})
    ))
def test_error_missing_residues_multi(lines, monomers, from_itp):
    """
    It can happen that there is a length mismatch between a multiblock as identified
    from the residue graph sequence and provided in the blocks file. We check that
    an error is raised.
    """
    lines = textwrap.dedent(lines).splitlines()
    ff = vermouth.forcefield.ForceField(name='test_ff')
    polyply.src.polyply_parser.read_polyply(lines, ff)
    # build the meta-molecule
    meta_mol = MetaMolecule(name="test", force_field=ff)
    meta_mol.add_monomer(0, monomers[0], [])
    for node, monomer in enumerate(monomers[1:]):
        meta_mol.add_monomer(node+1, monomer, [(node, node+1)])
    nx.set_node_attributes(meta_mol, from_itp, "from_itp")
    # map to molecule
    with pytest.raises(IOError):
        polyply.src.map_to_molecule.MapToMolecule(ff).run_molecule(meta_mol)
