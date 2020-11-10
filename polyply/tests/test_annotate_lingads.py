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
Test that ligands are properly annotated.
"""
import pytest
import numpy as np
import networkx as nx
import vermouth
import polyply
from polyply.src.meta_molecule import Monomer, MetaMolecule
from polyply.src.topology import Topology

# <mol_name>#<mol_idx>-<resname>#<resid>
@pytest.mark.parametrize('spec, expected', [
    ('', {}),
    ('A-ALA#1', {'molname': 'A', 'resname': 'ALA', 'resid': 1}),
    ('A-ALA1', {'molname': 'A', 'resname': 'ALA1'}),
    ('-ALA1', {'resname': 'ALA1'}),
    ('-ALA#1', {'resname': 'ALA', 'resid':1}),
    ('ALA1', {'molname': 'ALA1'}),
    ('A-ALA', {'molname': 'A', 'resname': 'ALA'}),
    ('ALA', {'molname': 'ALA'}),
    ('ALA#5', {'molname': 'ALA', 'mol_idx':5}),
    ('A#1-ALA#1', {'molname': 'A', 'mol_idx':1, 'resname': 'ALA', 'resid': 1.0}),
    ('#1-ALA#1', {'mol_idx':1, 'resname': 'ALA', 'resid': 1.0}),
    ('#2', {'mol_idx': 2}),
    ('#2-ALA', {'mol_idx': 2, 'resname': 'ALA'}),
    ('#2-#1', {'mol_idx': 2, 'resid':1.0}),
])
def test_parse_residue_spec(spec, expected):
    found = polyply.src.annotate_ligands.parse_residue_spec(spec)
    assert found == expected

# <mol_name>#<mol_idx>-<resname>#<resid>
@pytest.mark.parametrize('spec', [
    '#-#',
    '#-#2',
    'A#-#',
    '-A#qds',
     ])
def test_parse_residue_spec_fail(spec):
    with pytest.raises(IOError):
         polyply.src.annotate_ligands.parse_residue_spec(spec)

@pytest.fixture
def example_molecule():
    mol = nx.Graph()
    nodes = [
        {'resname': 'ALA', 'resid': 1, 'atomname':'BB'},
        {'resname': 'ALA', 'resid': 1, 'atomname':'SC1'},
        {'resname': 'ALA', 'resid': 2, 'atomname':'BB'},
        {'resname': 'ALA', 'resid': 2, 'atomname':'SC1'},
        {'resname': 'GLY', 'resid': 3, 'atomname':'BB'},
        {'resname': 'GLY', 'resid': 3, 'atomname':'SC1'},
    ]
    mol.add_nodes_from(enumerate(nodes))
    mol.add_edges_from([(0, 1), (0, 2), (2, 3), (2, 4), (4, 5)])
    return mol

@pytest.mark.parametrize('mol_attrs, expected', (
    ({"resid": 1}, [0, 1]),
    ({"resid": 2}, [2, 3]),
    ({"resname": "ALA"}, [0, 1, 2, 3]),
    ({"resname": "GLY"}, [4, 5]),
    ({"resid": 1, "resname": "ALA"}, [0, 1]),
    ({"resid": 1, "resname": "GLU"}, []),
    ({}, [0, 1, 2, 3, 4, 5])))
def test_find_nodes(example_molecule, mol_attrs, expected):
    found = list(polyply.src.annotate_ligands._find_nodes(example_molecule, mol_attrs))
    assert found == expected

@pytest.fixture()
def example_system():
    """
    Create a dummy test system with three types of molecules AA, BB and
    NA. NA is the molecule to be used a ligand. AA and BB are composed
    of different residues.
    """
    # dummy vermouth force-field
    force_field = vermouth.forcefield.ForceField(name='test_ff')
    # monomers used in the meta-molecule
    ALA = Monomer(resname="ALA", n_blocks=2)
    GLU = Monomer(resname="GLU", n_blocks=1)
    THR = Monomer(resname="THR", n_blocks=1)
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
    return top

@pytest.mark.parametrize('lig_spec, expected_lig_defs', [
    (('AA-ALA#1', 'NA'), {0: [(0,
                               3,
                             {'molname': 'AA', 'resname': 'ALA', 'resid': 1.0},
                             {'molname': 'NA'})],
                          1: [(0,
                               4,
                             {'molname': 'AA', 'resname': 'ALA', 'resid': 1.0},
                             {'molname': 'NA'})]}),
    (('#0-ALA#1', 'NA'), {0: [(0,
                               3,
                             {'mol_idx': 0.0, 'resname': 'ALA', 'resid': 1.0},
                             {'molname': 'NA'})]}),
   (('#0-ALA#1', '#5-NA'), {0: [(0,
                               5,
                             {'mol_idx': 0.0, 'resname': 'ALA', 'resid': 1.0},
                             {'molname': 'NA'})]}),
    (('AA#1-ALA#1', 'NA'), {1: [(0,
                                 3,
                             {'mol_idx': 1, 'molname': 'AA', 'resname': 'ALA', 'resid': 1.0},
                             {'molname': 'NA'})]}),
    (('AA-GLU', 'NA'), {0: [(2,
                             3,
                            {'molname': 'AA', 'resname': 'GLU'},
                            {'molname': 'NA'})],
                        1: [(2,
                             4,
                            {'molname': 'AA', 'resname': 'GLU'},
                            {'molname': 'NA'})]}),
    (('-GLU', 'NA'),   {0: [(2,
                             3,
                            {'resname': 'GLU'},
                            {'molname': 'NA'})],
                        1: [(2,
                             4,
                            {'resname': 'GLU'},
                            {'molname': 'NA'})],
                        2: [(0,
                             5,
                            {'resname': 'GLU'},
                            {'molname': 'NA'})]}),
    (('AA-ALA', 'NA'), {0: [(0,
                             3,
                             {'molname': 'AA', 'resname': 'ALA'},
                             {'molname': 'NA'}),
                            (1,
                             4,
                             {'molname': 'AA', 'resname': 'ALA'},
                             {'molname': 'NA'})],
                        1: [(0,
                             5,
                             {'molname': 'AA', 'resname': 'ALA'},
                             {'molname': 'NA'}),
                            (1,
                             6,
                             {'molname': 'AA', 'resname': 'ALA'},
                             {'molname': 'NA'})]}),
    (('BB-ALA', 'NA'), {2: [(1,
                             3,
                             {'molname': 'BB', 'resname': 'ALA'},
                             {'molname': 'NA'})]})])
def test_init(example_system, lig_spec, expected_lig_defs):
    """
    Test if based on the ligand specs the molecules and correct nodes
    are added and annotated.
    """
    processor = polyply.src.annotate_ligands.AnnotateLigands(topology=example_system, ligands=[lig_spec])
    for mol_idx in processor.ligand_defs:
        for lig, lig_ref in zip(processor.ligand_defs[mol_idx], expected_lig_defs[mol_idx]):
            assert lig[0] == lig_ref[0]
            assert lig[1] == lig_ref[1]
            assert lig[2] == lig_ref[2]

@pytest.mark.parametrize('lig_spec', [
    ('AA#2-ALA#1', 'NA'),
    ('AA-ALA#1', '')])
def test_init_fail(example_system, lig_spec):
    """
    Test if based on the ligand specs the molecules and correct nodes
    are added and annotated.
    """
    with pytest.raises(IOError):
         polyply.src.annotate_ligands.AnnotateLigands(topology=example_system, ligands=[lig_spec])

@pytest.mark.parametrize('lig_spec, expected_mols', [
    (('AA-ALA#1', 'NA'), {0: [4],
                          1: [4]}),
    (('#0-ALA#1', 'NA'), {0: [4]}),
    (('AA#1-ALA#1', 'NA'), {1: [4]}),
    (('AA-GLU', 'NA'), {0: [4],
                        1: [4]}),
    (('AA-ALA', 'NA'), {0: [4, 5],
                        1: [4, 5]}),
    (('BB-ALA', 'NA'), {2: [4, 5]})])
def test_annotate_molecules(example_system, lig_spec, expected_mols):
    """
    Test if based on the ligand specs the molecules and correct nodes
    are added and annotated.
    """
    processor = polyply.src.annotate_ligands.AnnotateLigands(topology=example_system, ligands=[lig_spec])
    processor.run_system(example_system)
    for mol_idx, expected_nodes in expected_mols.items():
        for node in expected_nodes:
            assert "build" in example_system.molecules[mol_idx].nodes[node]
            assert "ligated" in example_system.molecules[mol_idx].nodes[node]

    for idx, mol in enumerate(example_system.molecules):
        for node in mol.nodes:
            if "ligated" in mol.nodes[node]:
                assert node in expected_mols[idx]

@pytest.mark.parametrize('lig_spec, expected_mols', [
    (('AA-ALA#1', 'NA'), {0: [4],
                          1: [4]}),
    (('#0-ALA#1', 'NA'), {0: [4]}),
    (('AA#1-ALA#1', 'NA'), {1: [4]}),
    (('AA-GLU', 'NA'), {0: [4],
                        1: [4]}),
    (('AA-ALA', 'NA'), {0: [4, 5],
                        1: [4, 5]}),
    (('BB-ALA', 'NA'), {2: [4, 5]})])
def test_split_molecules(example_system, lig_spec, expected_mols):
    """
    Test if based on the ligand specs the molecules and correct nodes
    are added and annotated.
    """
    processor = polyply.src.annotate_ligands.AnnotateLigands(topology=example_system, ligands=[lig_spec])
    processor.run_system(example_system)
    lig_count = 0
    for mol_idx, expected_nodes in expected_mols.items():
        for node in expected_nodes:
            lig_count += 1
            example_system.molecules[mol_idx].nodes[node]["position"] = np.array([1.0, 1.0, 1.0])

    processor.split_ligands()

    for mol_idx, expected_nodes in expected_mols.items():
        for node in expected_nodes:
            assert node not in example_system.molecules[mol_idx].nodes

    for idx in range(3, lig_count):
        assert "position" in example_system.molecules[idx].nodes[0]
