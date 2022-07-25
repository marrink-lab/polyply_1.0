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
Test that build files are properly read.
"""
import textwrap
import pytest
import numpy as np
import networkx as nx
import vermouth
import vermouth.forcefield
import vermouth.molecule
import polyply
from polyply.src.meta_molecule import Monomer, MetaMolecule
from polyply.src.topology import Topology

@pytest.mark.parametrize('tokens, _type, expected', (
   # for cylinder
   (["PEO", "63", "154", "in", "1", "2", "3", "6", "7"],
    "cylinder",
    {"resname": "PEO", "start": 63, "stop": 154, "parameters":["in", np.array([1.0, 2.0, 3.0]), 6.0, 7.0, "cylinder"]})
   # for recangle
   ,(["PEO", "0", "10", "out", "11", "12", "13", "1", "2", "3"],
    "rectangle",
    {"resname": "PEO", "start": 0, "stop": 10, "parameters":["out", np.array([11.0, 12.0, 13.0]), 1.0, 2.0, 3.0, "rectangle"]})
   # for sphere
   ,(["PEO", "0", "10", "in", "11", "12", "13", "5"],
    "sphere",
    {"resname": "PEO", "start": 0, "stop": 10, "parameters":["in", np.array([11.0, 12.0, 13.0]), 5.0, "sphere"]})
   ))
def test_base_parser_geometry(tokens, _type, expected):
    result = polyply.src.build_file_parser.BuildDirector._base_parser_geometry(tokens, _type)
    assert result.keys() == expected.keys()
    for key in result:
        if key != "parameters":
            assert result[key] == expected[key]
        else:
            assert result[key][0] == expected[key][0]
            assert all(result[key][1] == expected[key][1])
            for result_param, expected_param in zip(result[key][2:], expected[key][2:]):
                assert result_param == expected_param

@pytest.fixture
def test_molecule():
    # dummy vermouth force-field
    force_field = vermouth.forcefield.ForceField(name='test_ff')
    # monomers used in the meta-molecule
    ALA = Monomer(resname="ALA", n_blocks=4)
    GLU = Monomer(resname="GLU", n_blocks=2)
    THR = Monomer(resname="THR", n_blocks=3)
    # two meta-molecules
    mol = MetaMolecule.from_monomer_seq_linear(force_field,
                                               [ALA, GLU, THR],
                                               "AA")
    return mol

@pytest.mark.parametrize('_type, option, expected', (
   # tag all ALA nodes
   ("cylinder",
    {"resname": "ALA", "start": 1, "stop": 4, "parameters":["in", np.array([5.0, 5.0, 5.0]), 5.0, 5.0]},
    [0, 1, 2, 3]),
   # tag all nodes 1-4 that have name ALA, GLU, THR should not get tagged
   ("cylinder",
    {"resname": "ALA", "start": 1, "stop": 9, "parameters":["in", np.array([5.0, 5.0, 5.0]), 5.0, 5.0]},
   [0, 1, 2, 3]),
   # tag only GLU nodes
   ("sphere",
    {"resname": "GLU", "start": 5, "stop": 6, "parameters":["in", np.array([10.0, 10.0, 10.0]), 5.0]},
   [4, 5]),
   # tag no nodes based on the resid rensame correspondance
   ("sphere",
    {"resname": "GLU", "start": 1, "stop": 4, "parameters":["in", np.array([10.0, 10.0, 10.0]), 5.0]},
   [])
   # tag no nodes based on no resname in molecule
   ,("sphere",
    {"resname": "PPI", "start": 1, "stop": 4, "parameters":["in", np.array([10.0, 10.0, 10.0]), 5.0]},
   [])
   ))
def test_tag_nodes(test_molecule, _type, option, expected):
    polyply.src.build_file_parser.BuildDirector._tag_nodes(test_molecule, _type, option)
    for node in test_molecule.nodes:
        if "restraints" in test_molecule.nodes[node]:
           assert node in expected

@pytest.fixture()
def test_system():
  """
  Create a dummy test system with three types of molecules AA, BB and
  NA. NA is the molecule to be used a ligand. AA and BB are composed
  of different residues.
  """
  # dummy vermouth force-field
  force_field = vermouth.forcefield.ForceField(name='test_ff')
  # monomers used in the meta-molecule
  ALA = Monomer(resname="ALA", n_blocks=4)
  GLU = Monomer(resname="GLU", n_blocks=2)
  THR = Monomer(resname="THR", n_blocks=3)
  # two meta-molecules
  meta_mol_A = MetaMolecule.from_monomer_seq_linear(force_field,
                                                    [ALA, GLU, THR],
                                                    "AA")
  meta_mol_B = MetaMolecule.from_monomer_seq_linear(force_field,
                                                    [GLU, ALA, THR],
                                                    "BB")
  NA = MetaMolecule()
  NA.add_monomer(current=0, resname="NA", connections=[])
  molecules = [meta_mol_A, meta_mol_A.copy(),
               meta_mol_B.copy(), NA, NA.copy(),
               NA.copy(), NA.copy()]
  top = Topology(force_field=force_field)
  top.molecules = molecules
  top.mol_idx_by_name = {"AA":[0, 1], "BB": [2], "NA":[3, 4, 5, 6]}
  top.nonbond_params = {frozenset(["O"]): {"nb1": 0.30},
                        frozenset(["C"]): {"nb1": 0.36},
                        frozenset(["H"]): {"nb1": 0.04}}
  return top

@pytest.mark.parametrize('line, expected', (
   # simple example
   ("ALA 0 4 5.0 3.0 2.0 50.0",
    {"resname": "ALA", "start": 0, "stop": 4, "parameters": [np.array([5.0, 3.0, 2.0]), 50.]})
   # other example
   ,("GLU 5 6 4.0 4.0 1.0 25.0",
    {"resname": "GLU", "start": 5, "stop": 6, "parameters": [np.array([4.0, 4.0, 1.0]), 25.0]})
   ))
def test_rw_restriction_parsing(test_system, line, expected):
    processor = polyply.src.build_file_parser.BuildDirector([], test_system)
    processor.current_molidxs = [1]
    processor.current_molname = "AA"
    processor._rw_restriction(line)
    result =  processor.rw_options[("AA", 1)]
    assert result.keys() == expected.keys()
    for key in processor.rw_options[("AA", 1)]:
        if key != "parameters":
            assert result[key] == expected[key]
        else:
            assert all(result[key][0] == expected[key][0])
            assert result[key][1] == expected[key][1]

@pytest.mark.parametrize('line, key, expected', (
   # simple example
   ("1 8 3.0",
    (1, 8),
    (3.0, 0.0)),
   ("1 8 3.0 1.0",
    (1, 8),
    (3.0, 1.0)),
   ))
def test_distance_restraints(test_system, line, key, expected):
    processor = polyply.src.build_file_parser.BuildDirector([], test_system)
    processor.current_molidxs = [0]
    processor.current_molname = "AA"
    processor._distance_restraints(line)
    result = test_system.distance_restraints[("AA", 0)]
    result[key] == expected


@pytest.mark.parametrize('line', (
   # basic test
   ("1 50 3.0",
    "50 1 3.0")))
def test_distance_restraints_error(test_system, line):
    processor = polyply.src.build_file_parser.BuildDirector([], test_system)
    processor.current_molidxs = [0]
    processor.current_molname = "AA"
    with pytest.raises(IOError):
        processor._distance_restraints(line)


@pytest.mark.parametrize('lines, tagged_mols, tagged_nodes', (
   # basic test
   ("""
   [ molecule ]
   ; name from to
   AA    0  2
   ;
   [ cylinder ]
   ; resname start stop  inside-out  x  y  z  r   z
   ALA   2    4  in  5  5  5  5  5
   """,
   [0, 1],
   [0, 1, 2, 3]),
   # test that a different molecule is tagged
   ("""
   [ molecule ]
   BB  2  3
   [ rectangle ]
   ; resname start stop  inside-out  x  y  z a b c
   ALA   3    6  in  5  5  5  5  5 5
   """,
   [2],
   [2, 3, 4, 5]),
   # test nothing is tagged based on the molname
   ("""
   [ molecule ]
   CC 1 6
   [ sphere ]
   ; resname start stop  inside-out  x  y  z r
   ALA   2    4  in  5  5  5  5
   """,
   [],
   []),
   ))
def test_parser(test_system, lines, tagged_mols, tagged_nodes):
   lines = textwrap.dedent(lines).splitlines()
   ff = vermouth.forcefield.ForceField(name='test_ff')
   top = Topology(ff)
   polyply.src.build_file_parser.read_build_file(lines,
                                                 test_system.molecules,
                                                 top)
   for idx, mol in enumerate(test_system.molecules):
       for node in mol.nodes:
           if "restraints" in mol.nodes[node]:
               assert node in tagged_nodes
               assert idx in tagged_mols


@pytest.mark.parametrize('lines, expected', (
   # basic test
   ("""
   [ molecule ]
   ; name from to
   AA    0  2
   ;
   [ persistence_length ]
   ; model  lp start stop
     WCM   4.0  0    8
   """,
   [["WCM", 4.0, 0, 8, np.array([0, 1])]]),
   # test two batches are recognized
   ("""
   [ molecule ]
   ; name from to
   AA    0  2
   ;
   [ persistence_length ]
   ; model  lp start stop
   WCM   4.0  0    8
   [ molecule ]
   ; name from to
   BB 2  4
   [ persistence_length ]
   WCM   2.0   0  8
   """,
   [["WCM", 4.0, 0, 8, np.array([0, 1])],
    ["WCM", 2.0, 0, 8, np.array([2, 3])]]
   )))
def test_persistence_parsers(test_system, lines, expected):
   lines = textwrap.dedent(lines).splitlines()
   ff = vermouth.forcefield.ForceField(name='test_ff')
   top = Topology(ff)
   polyply.src.build_file_parser.read_build_file(lines,
                                                 test_system.molecules,
                                                 top)
   for ref, new in zip(expected, top.persistences):
        print(ref, new)
        for info_ref, info_new in zip(ref[:-1], new[:-1]):
            assert info_ref == info_new
        assert all(ref[-1] == new[-1])

@pytest.mark.parametrize('line, expected', (
   (("""
     [ volumes ]
     PEO 0.47
     """,
     {"PEO": 0.47}),
    ("""
     [ volumes ]
     PEO 0.47
     P3HT 0.61
     """,
     {"PEO": 0.47, "P3HT": 0.61}),
)))
def test_volume_parsing(test_system, line, expected):
    lines = textwrap.dedent(line)
    lines = lines.splitlines()
    polyply.src.build_file_parser.read_build_file(lines,
                                                  test_system.molecules,
                                                  test_system)
    assert test_system.volumes == expected

@pytest.mark.parametrize('line, names, edges, positions, out_vol', (
   (
    # 1 - template but volume are provided before
    ("""[ volumes ]
    PEO 0.43
    [ template ]
    resname PEO
    [ atoms ]
    EC1 C 0.000 0.000 0.000
    O1  O 0.000 0.000 0.150
    EC2 C 0.000 0.000 0.300
    [ bonds ]
    EC1 O1
    O1 EC2
    """,
    ["PEO"],
    [[("EC1", "O1"), ("O1", "EC2")]],
    # template positions are defined as vectors
    # from the center of geometry internally
    [[("EC1", 0.0, 0.0, -0.150),
      ("O1", 0.000, 0.000, 0.000),
      ("EC2", 0.000, 0.000, 0.150)]],
    {"PEO": 0.43}),
    # 2 - template but volume are provided after
    ("""[ template ]
    resname PEO
    [ atoms ]
    EC1 C  0.000 0.000 0.000
    O1  O 0.000 0.000 0.150
    EC2 C 0.000 0.000 0.300
    [ bonds ]
    EC1 O1
    O1 EC2
    [ volumes ]
    PEO 0.43
    """,
    ["PEO"],
    [[("EC1", "O1"), ("O1", "EC2")]],
    # template positions are defined as vectors
    # from the center of geometry internally
    [[("EC1", 0.0, 0.0, -0.150),
      ("O1", 0.000, 0.000, 0.000),
      ("EC2", 0.000, 0.000, 0.150)]],
    {"PEO": 0.43}),

    # 3 - two templates and volumes are provided before
    ("""[ volumes ]
    PEO 0.43
    OH 0.22
    [ template ]
    resname PEO
    [ atoms ]
    EC1 C 0.000 0.000 0.000
    O1  O 0.000 0.000 0.150
    EC2 C 0.000 0.000 0.300
    [ bonds ]
    EC1 O1
    O1 EC2
    [ template ]
    resname OH
    [ atoms ]
    O1 O 0.0000 0.000 0.000
    H1 H 0.0000 0.000 0.100
    [ bonds ]
    O1 H1
    """,
    ["PEO", "OH"],
    [[("EC1", "O1"), ("O1", "EC2")], [("O1", "H1")]],
    # template positions are defined as vectors
    # from the center of geometry internally
    [[("EC1", 0.0, 0.0, -0.150),
      ("O1", 0.000, 0.000, 0.000),
      ("EC2", 0.000, 0.000, 0.150)],
     [("O1", 0.0000, 0.000, -0.050),
      ("H1", 0.0000, 0.000, 0.050)]],
    {"PEO": 0.43, "OH": 0.22}),
    # 4 - two templates but only 1 volume are provided
    ("""[ template ]
    resname PEO
    [ atoms ]
    EC1 C 0.000 0.000 0.000
    O1  O 0.000 0.000 0.150
    EC2 C 0.000 0.000 0.300
    [ bonds ]
    EC1 O1
    O1 EC2
    [ template ]
    resname OH
    [ atoms ]
    O1 O 0.0000 0.000 0.000
    H1 H 0.0000 0.000 0.100
    [ bonds ]
    O1 H1
    [ volumes ]
    OH 0.22
    """,
    ["PEO", "OH"],
    [[("EC1", "O1"), ("O1", "EC2")], [("O1", "H1")]],
    # template positions are defined as vectors
    # from the center of geometry internally
    [[("EC1", 0.0, 0.0, -0.150),
      ("O1", 0.000, 0.000, 0.000),
      ("EC2", 0.000, 0.000, 0.150)],
     [("O1", 0.0000, 0.000, -0.050),
      ("H1", 0.0000, 0.000, 0.050)]],
    {"PEO": 0.4164132562731403, "OH": 0.22}),
)))
def test_template_volume_parsing(test_system, line, names, edges, positions, out_vol):
    lines = textwrap.dedent(line)
    lines = lines.splitlines()
    polyply.src.build_file_parser.read_build_file(lines,
                                                  test_system.molecules,
                                                  test_system)
    for mol in test_system.molecules:
        assert len(mol.templates) == len(names)
        for idx, name in enumerate(names):
            template = mol.templates[name]
            for node_pos in positions[idx]:
                node = node_pos[0]
                assert np.all(np.array(node_pos[1:], dtype=float) == template[node])

    assert test_system.volumes == out_vol
