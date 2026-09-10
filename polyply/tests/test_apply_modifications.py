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
    # a protein modification does not match a PEO residue, because the
    # residue has none of the atoms the modification describes
    ('PEO.itp',
     'PEO',
    True,
     ("Cannot apply N-ter to PEO1, because"
      " the residue has no atoms BB."))
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
    nx.set_node_attributes(meta_mol, False, 'from_itp')

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

    if not adding:
        for node in meta_mol.nodes:
            meta_mol.nodes[node]['from_itp'] = False

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


def test_apply_mod_adds_atoms():
    """
    A modification that describes atoms which are not part of the block
    must add those atoms, their edges, and the interactions to the
    molecule, also for a residue that is not a protein residue.
    """
    ff = vermouth.forcefield.ForceField(name='test')
    meta_mol = MetaMolecule.from_itp(ff, TEST_DATA / "itp" / "PEO.itp", "PEO")
    nx.set_node_attributes(meta_mol, False, 'from_itp')
    molecule = meta_mol.molecule

    # hang an extra bead off the EO bead of the first residue and change
    # the atomtype of that bead at the same time
    modification = vermouth.molecule.Modification(name="EO-OH")
    modification.add_node("EO", **{'atomname': 'EO', 'PTM_atom': False,
                                   'replace': {'atype': 'P4'}})
    modification.add_node("OH", **{'atomname': 'OH', 'PTM_atom': True,
                                   'atype': 'P1', 'charge': -0.3, 'mass': 17.0})
    modification.add_edge("EO", "OH")
    modification.interactions['bonds'].append(Interaction(atoms=["EO", "OH"],
                                                          parameters=['1', '0.30', '7000'],
                                                          meta={}))
    ff.modifications["EO-OH"] = modification

    apply_mod(meta_mol, [({'resid': 1, 'resname': 'PEO'}, "EO-OH")])

    # the new atom is added at the end of the molecule and inherits the
    # resid, resname, and charge group of the residue it belongs to
    assert len(molecule.nodes) == 4
    new_node = 3
    assert molecule.nodes[new_node]['atomname'] == 'OH'
    assert molecule.nodes[new_node]['atype'] == 'P1'
    assert molecule.nodes[new_node]['resid'] == 1
    assert molecule.nodes[new_node]['resname'] == 'PEO'
    assert molecule.nodes[new_node]['charge_group'] == molecule.nodes[0]['charge_group']
    # the attributes of the atoms described by the block are replaced
    assert molecule.nodes[0]['atype'] == 'P4'
    # the edge and interaction are added using the new atom
    assert molecule.has_edge(0, new_node)
    assert Interaction(atoms=(0, new_node), parameters=['1', '0.30', '7000'],
                       meta={}) in molecule.interactions['bonds']
    # the graph of the residue is kept in sync with the molecule
    residue = meta_mol.nodes[0]['graph']
    assert new_node in residue
    assert residue.has_edge(0, new_node)
    # the molecule is sorted such that the residues stay contiguous; the
    # node keys themselves are not changed
    assert list(molecule.nodes) == [0, new_node, 1, 2]
    assert [molecule.nodes[node]['resid'] for node in molecule.nodes] == [1, 1, 2, 3]


def test_apply_mod_undefined_modification():
    """
    Asking for a modification that the force-field does not define is an
    error, because the user explicitly asked for it.
    """
    ff = vermouth.forcefield.ForceField(name='test')
    meta_mol = MetaMolecule.from_itp(ff, TEST_DATA / "itp" / "PEO.itp", "PEO")
    nx.set_node_attributes(meta_mol, False, 'from_itp')
    ff.modifications["some-mod"] = vermouth.molecule.Modification(name="some-mod")

    with pytest.raises(IOError):
        apply_mod(meta_mol, [({'resid': 1, 'resname': 'PEO'}, "does-not-exist")])
