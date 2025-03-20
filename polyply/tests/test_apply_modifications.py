# Copyright 2024 University of Groningen
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
Test that termini modifications are applied
"""
import logging
from contextlib import nullcontext as does_not_raise
from .example_fixtures import example_meta_molecule
import pytest
import vermouth.forcefield
import vermouth.ffinput
from polyply.src.meta_molecule import MetaMolecule
from polyply.src.apply_modifications import _patch_protein_termini, apply_mod, ApplyModifications, modifications_finalising
from polyply import TEST_DATA
import polyply.src.ff_parser_sub
import networkx as nx
from vermouth.molecule import Interaction
from polyply.src.meta_molecule import MetaMolecule


@pytest.mark.parametrize('input_mods, expected',(
        (['N-ter', 'C-ter'],
         [({'resid': 1, 'resname': 'A'}, 'N-ter'),
          ({'resid': 3, 'resname': 'A'}, 'C-ter')]
        ),
        (['Zwitter'],
        [({'resid': 1, 'resname': 'A'}, 'Zwitter'),
         ({'resid': 3, 'resname': 'A'}, 'Zwitter')]
        )
))
def test_annotate_protein(example_meta_molecule, input_mods, expected):
    """"
    test that a protein sequence gets annotated correctly
    """

    assert expected == _patch_protein_termini(example_meta_molecule, input_mods)

@pytest.mark.parametrize(
    'ff_files, expected', (
            (
                    ['aminoacids.ff', 'modifications.ff'],
                    does_not_raise()
            ),
            (
                    ['aminoacids.ff'],
                    pytest.raises(AssertionError)
            ),
    )
)
def test_mods_in_ff(caplog, ff_files, expected):
    """
    test that modifications exist in the force field that can be applied
    """
    ff = vermouth.forcefield.ForceField(name='martini3')

    ff_lines = []
    for file in ff_files:
        with open(TEST_DATA/ "ff" / file) as f:
            ff_lines += f.readlines()

    polyply.src.ff_parser_sub.read_ff(ff_lines, ff)

    with expected:
        assert len(ff.modifications) > 0

@pytest.mark.parametrize('input_itp, molname, expected, text',
                         (
    (
    'ALA5.itp',
    'pALA',
    False,
    None),
    ('PEO.itp',
     'PEO',
    True,
     ("The resname of your target residue"
      " is not recognised as a protein resname."
      " Will not attempt to modify."))
))
def test_apply_mod(input_itp, molname, expected, caplog, text):
    """
    test that modifications get applied correctly
    """
    caplog.set_level(logging.INFO)
    #make the meta molecule from the itp and ff files
    file_name = TEST_DATA / "itp" / input_itp

    ff = vermouth.forcefield.ForceField(name='martini3')

    ff_lines = []
    for file in ['aminoacids.ff', 'modifications.ff']:
        with open(TEST_DATA/ "ff" / file) as f:
            ff_lines += f.readlines()
    polyply.src.ff_parser_sub.read_ff(ff_lines, ff)

    meta_mol = MetaMolecule.from_itp(ff, file_name, molname)

    #apply the mods
    termini = _patch_protein_termini(meta_mol)
    apply_mod(meta_mol, termini)

    if expected:
        for record in caplog.records:
            if record.message == text:
                assert True
                break
        else:
            assert False

    else:
        #for each mod applied, check that the mod atom and interactions have been changed correctly
        for target, modname in termini:
            mod = meta_mol.molecule.force_field.modifications[modname]
            names = nx.get_node_attributes(meta_mol.molecule, 'atomname')
            atypes = nx.get_node_attributes(meta_mol.molecule, 'atype')
            resids = nx.get_node_attributes(meta_mol.molecule, 'resid')

            _interaction_atoms = []
            for modatom in mod.atoms:
                for ind, (aname, atype, resid) in enumerate(zip(names.values(), atypes.values(), resids.values())):
                    if (resid == target['resid']) and (aname == modatom['atomname']):
                        if 'replace' in modatom.keys():
                            assert atype == modatom['replace']['atype']
                        _interaction_atoms.append(ind)

            for interaction_type in mod.interactions:
                for interaction in mod.interactions[interaction_type]:
                    for aname, atype, resid in zip(names.values(), atypes.values(), resids.values()):
                        if (resid == target['resid']):
                            _interaction = Interaction(atoms=tuple(_interaction_atoms),
                                                       parameters=interaction.parameters,
                                                       meta=interaction.meta)
                            assert _interaction in meta_mol.molecule.interactions[interaction_type]


@pytest.mark.parametrize('adding, expected',
                         (
    (True,
     True),
    (False,
     False)
                         ))
def test_from_itp(caplog, adding, expected):

    caplog.set_level(logging.INFO)
    #make the meta molecule from the itp and ff files
    file_name = TEST_DATA / "itp" / "ALA5.itp"

    ff = vermouth.forcefield.ForceField(name='martini3')

    ff_lines = []
    for file in ['aminoacids.ff', 'modifications.ff']:
        with open(TEST_DATA/ "ff" / file) as f:
            ff_lines += f.readlines()
    polyply.src.ff_parser_sub.read_ff(ff_lines, ff)

    meta_mol = MetaMolecule.from_itp(ff, file_name, "pALA")

    if adding:
        for node in meta_mol.nodes:
            meta_mol.nodes[node]['from_itp'] = 'True'

    termini = _patch_protein_termini(meta_mol)
    apply_mod(meta_mol, termini)

    found = False
    expected_msg = "meta_molecule has come from itp. Will not attempt to modify."
    for record in caplog.records:
        if record.message == expected_msg:
            found = True
            break
        else:
            continue

    assert found == expected

@pytest.mark.parametrize('modifications, expected, text',
     (
             (
                 [],
                 True,
                 "No modifications present in forcefield, none will be applied"
             ),

     ))
def test_ApplyModifications(example_meta_molecule, caplog, modifications, expected, text):

    caplog.set_level(logging.INFO)

    ApplyModifications(modifications=modifications,
                       meta_molecule=example_meta_molecule).run_molecule(example_meta_molecule)

    if expected:
        for record in caplog.records:
            if record.message == text:
                assert True
                break
        else:
            assert False
        assert any(rec.levelname == 'INFO' for rec in caplog.records)

@pytest.mark.parametrize('extras, expected',
                         (
                ([],
                          [({'resid': 1, 'resname': 'ALA'}, 'N-ter'),
                           ({'resid': 5, 'resname': 'ALA'}, 'C-ter')]),
                ([['ALA1', 'NH2-ter']],
                 [({'resid': 5, 'resname': 'ALA'}, 'C-ter'),
                  ({'resid': 1, 'resname': 'ALA'}, 'NH2-ter')])

                        ))
def test_multiple_modifications(extras, expected):

    #make the meta molecule from the itp and ff files
    file_name = TEST_DATA / "itp" / "ALA5.itp"

    ff = vermouth.forcefield.ForceField(name='martini3')

    ff_lines = []
    for file in ['aminoacids.ff', 'modifications.ff']:
        with open(TEST_DATA/ "ff" / file) as f:
            ff_lines += f.readlines()
    polyply.src.ff_parser_sub.read_ff(ff_lines, ff)

    meta_mol = MetaMolecule.from_itp(ff, file_name, "pALA")

    to_apply = modifications_finalising(meta_mol, extras)

    assert to_apply == expected
