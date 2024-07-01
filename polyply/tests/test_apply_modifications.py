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
from polyply.src.apply_modifications import _get_protein_termini, apply_mod, ApplyModifications
from polyply import TEST_DATA
import polyply.src.ff_parser_sub
import networkx as nx
from vermouth.molecule import Interaction

def test_annotate_protein(example_meta_molecule):
    """"
    test that a protein sequence gets annotated correctly
    """
    result = [({'resid': 1, 'resname': 'A'}, 'N-ter'),
               ({'resid': 3, 'resname': 'A'}, 'C-ter')]

    termini = _get_protein_termini(example_meta_molecule)
    assert termini == result

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

def test_apply_mod(caplog):
    """
    test that modifications get applied correctly
    """
    #make the meta molecule from the itp and ff files
    file_name = TEST_DATA / "itp" / "ALA5.itp"

    ff = vermouth.forcefield.ForceField(name='martini3')

    ff_lines = []
    for file in ['aminoacids.ff', 'modifications.ff']:
        with open(TEST_DATA/ "ff" / file) as f:
            ff_lines += f.readlines()
    polyply.src.ff_parser_sub.read_ff(ff_lines, ff)

    name = "pALA"
    meta_mol = MetaMolecule.from_itp(ff, file_name, name)

    #apply the mods
    termini = _get_protein_termini(meta_mol)
    apply_mod(meta_mol, termini)

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

@pytest.mark.parametrize('modifications, protein_termini, expected',
     (
             (
                 None,
                 False,
                 True
             ),

             (None,
              True,
              False)
     ))
def test_ApplyModifications(example_meta_molecule, caplog, modifications, protein_termini, expected):

    ApplyModifications(modifications=modifications,
                       protter=protein_termini).run_molecule(example_meta_molecule)

    if expected:
        print(caplog.records)
        assert any(rec.levelname == 'WARNING' for rec in caplog.records)

