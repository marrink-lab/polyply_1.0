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

import pytest
import vermouth.forcefield
import vermouth.ffinput
from vermouth.molecule import Block
from polyply.src.meta_molecule import MetaMolecule
from polyply.src.apply_modifications import _get_protein_termini, apply_mod
from polyply import TEST_DATA
import polyply.src.ff_parser_sub
import networkx as nx

def example_meta_molecule(mods_dict = {}):
    """

    adapted from test_apply_links

    Example molecule with three residues each with
    different attributes for testing the assignment
    of links.

    Names:
    -------
    BB - BB - BB - BB - BB - BB - BB
         |         |              |
        SC1       SC1            SC1
         |         |              |
        SC2       SC2            SC2

    Nodes:
    ------
     0  - 1    4  - 5 - 6   9 - 10
          |         |           |
          2         7           11
          |         |           |
          3         8           12
    """
    force_field = vermouth.forcefield.ForceField("test")

    force_field.modifications.update(mods_dict)

    block_A = Block(force_field=force_field)
    block_A.add_nodes_from([(0, {'resid': 1,  'name': 'BB',  'atype': 'A', 'resname': 'GLY', 'other': 'A'}),
                            (1, {'resid': 1,  'name': 'BB', 'atype': 'B', 'resname': 'GLY', 'other': 'A'}),
                            (2, {'resid': 1,  'name': 'SC1', 'atype': 'C', 'resname': 'GLY', 'other': 'A'}),
                            (3, {'resid': 1,  'name': 'SC2', 'atype': 'D', 'resname': 'GLY', 'other': 'A'})])
    block_A.add_edges_from([(0, 1), (1, 2), (2, 3)])

    block_B = Block(force_field=force_field)
    block_B.add_nodes_from([(0, {'resid': 1,  'name': 'BB', 'atype': 'A', 'resname': 'GLY'}),
                            (1, {'resid': 1,  'name': 'BB', 'atype': 'B', 'resname': 'GLY'}),
                            (2, {'resid': 1,  'name': 'BB', 'atype': 'A', 'resname': 'GLY'}),
                            (3, {'resid': 1,  'name': 'SC1', 'atype': 'A', 'resname': 'GLY'}),
                            (4, {'resid': 1,  'name': 'SC2', 'atype': 'C', 'resname': 'GLY'})])
    block_B.add_edges_from([(0, 1), (1, 2), (2, 3), (1, 4)])

    molecule = block_A.to_molecule()
    molecule.merge_molecule(block_B)
    molecule.merge_molecule(block_A)
    molecule.add_edges_from([(1, 4), (8, 9)])

    graph = MetaMolecule._block_graph_to_res_graph(molecule)
    meta_mol = MetaMolecule(graph, force_field=force_field, mol_name="test")
    # before do links is called there are no edges between the residues
    # at lower level
    molecule.remove_edges_from([(1, 4), (8, 9)])
    meta_mol.molecule = molecule
    return meta_mol


@pytest.mark.parametrize('result',
    (
        [[({'resid': 1, 'resname': 'GLY'}, 'N-ter'),
          ({'resid': 3, 'resname': 'GLY'}, 'C-ter')]
        ]
    )
                         )
def test_annotate_protein(result):
    meta_mol = example_meta_molecule()
    termini = _get_protein_termini(meta_mol)
    assert termini == result

@pytest.mark.parametrize(
    'ff_files, expected', (
            (
                    ['aminoacids.ff', 'modifications.ff'],
                    False
            ),
            (
                    ['aminoacids.ff'],
                    True
            ),
    )
)
def test_apply_mod(caplog, ff_files, expected):
    file_name = TEST_DATA / "itp" / "GLY5.itp"

    ff = vermouth.forcefield.ForceField(name='martini3')

    ff_lines = []
    for file in ff_files:
        with open(TEST_DATA/ "ff" / file) as f:
            ff_lines += f.readlines()

    polyply.src.ff_parser_sub.read_ff(ff_lines, ff)

    name = "pGLY"
    meta_mol = MetaMolecule.from_itp(ff, file_name, name)

    termini = _get_protein_termini(meta_mol)

    apply_mod(meta_mol, termini)

    for target, modname in termini:
        try:
            mod = meta_mol.molecule.force_field.modifications[modname]
            names = nx.get_node_attributes(meta_mol.molecule, 'atomname')
            atypes = nx.get_node_attributes(meta_mol.molecule, 'atype')
            resids = nx.get_node_attributes(meta_mol.molecule, 'resid')

            for modatom in mod.atoms:
                for aname, atype, resid in zip(names.values(), atypes.values(), resids.values()):
                    if (resid == target['resid']) and (atype == modatom['atomname']):
                            assert atype == modatom['replace']['atype']
        except KeyError:
            pass

    if expected:
        assert any(rec.levelname == 'WARNING' for rec in caplog.records)
    else:
        assert caplog.records == []
