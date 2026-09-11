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
from polyply.src.apply_modifications import (_patch_protein_termini, apply_mod,
                                             ApplyModifications, modifications_finalising,
                                             _find_modifications, modification_anchors)
from polyply import TEST_DATA
from collections import defaultdict
from polyply.src.molecule_utils import handle_ptms, extract_links, find_termini_mods
from polyply.src.ffoutput import ForceFieldDirectiveWriter
from polyply.src.gen_itp import gen_params
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


def _build_capped_chain():
    """
    Build a molecule of four 'B' residues in a row, where the side chain
    of the third one is extended by an 'X' residue while the side chains
    of the others are capped with a hydrogen. The block of 'B' is the
    version without the cap, so the capped residues - one of them an
    interior residue - are described by a modification.

    The residues are numbered the way CGSmiles numbers the sequence
    `{[#B][#B][#B]([#X])[#B]}`, that is the branch comes before the rest
    of the chain, because the links are generated from the difference in
    resid.
    """
    force_field = vermouth.forcefield.ForceField(name='test')
    molecule = vermouth.molecule.Molecule(force_field=force_field)
    molecule.nrexcl = 1

    layout = [(1, 'B', [('BB', 'C'), ('SC', 'C'), ('HSC', 'H')]),
              (2, 'B', [('BB', 'C'), ('SC', 'C'), ('HSC', 'H')]),
              (3, 'B', [('BB', 'C'), ('SC', 'C')]),
              (4, 'X', [('X1', 'O')]),
              (5, 'B', [('BB', 'C'), ('SC', 'C'), ('HSC', 'H')])]
    node = 0
    nodes_of = {}
    for resid, resname, atoms in layout:
        nodes_of[resid] = {}
        for atomname, element in atoms:
            molecule.add_node(node, atomname=atomname, element=element, resname=resname,
                              resid=resid, atype='P1', mass={'C': 12.0, 'H': 1.0, 'O': 16.0}[element],
                              charge=0.0, charge_group=resid)
            nodes_of[resid][atomname] = node
            node += 1

    edges = []
    for resid in (1, 2, 3, 5):
        edges.append((nodes_of[resid]['BB'], nodes_of[resid]['SC']))
        if 'HSC' in nodes_of[resid]:
            edges.append((nodes_of[resid]['SC'], nodes_of[resid]['HSC']))
    edges += [(nodes_of[1]['BB'], nodes_of[2]['BB']),
              (nodes_of[2]['BB'], nodes_of[3]['BB']),
              (nodes_of[3]['BB'], nodes_of[5]['BB']),
              (nodes_of[3]['SC'], nodes_of[4]['X1'])]
    molecule.add_edges_from(edges)
    for idx, jdx in edges:
        molecule.interactions['bonds'].append(Interaction(atoms=(idx, jdx),
                                                          parameters=['1', '0.35', '7000'],
                                                          meta={}))
    return force_field, molecule


def _meta_molecule_of(force_field, molecule):
    meta_molecule = MetaMolecule(MetaMolecule._block_graph_to_res_graph(molecule),
                                 force_field=force_field, mol_name='test')
    meta_molecule.molecule = molecule
    return meta_molecule


def test_find_modifications_free_anchor():
    """
    A modification is applied to the residues whose anchor has no bond
    leaving the residue, that is those whose bonding operator is unused,
    no matter whether that residue is terminal or not.
    """
    force_field, molecule = _build_capped_chain()
    modification = vermouth.molecule.Modification(name='B-cap')
    modification.add_node('SC', **{'atomname': 'SC', 'element': 'C',
                                   'resname': 'B', 'PTM_atom': False})
    modification.add_node('HSC', **{'atomname': 'HSC', 'element': 'H', 'atype': 'P1',
                                    'mass': 1.0, 'charge': 0.0, 'PTM_atom': True})
    modification.add_edge('SC', 'HSC')
    force_field.modifications['B-cap'] = modification

    targets = _find_modifications(_meta_molecule_of(force_field, molecule))

    # residue 3 uses its side chain operator for the X residue, the other
    # B residues do not; residue 2 is an interior residue
    assert sorted(target['resid'] for target, _ in targets) == [1, 2, 5]
    assert {name for _, name in targets} == {'B-cap'}


def test_find_modifications_needs_all_anchors_free():
    """
    A modification only applies if exactly its own anchors are free, so
    that a residue with more than one unused bonding operator is not
    described by the modification of a single operator.
    """
    force_field, molecule = _build_capped_chain()
    # this modification caps the side chain and the backbone, so it only
    # applies where both are free, which is true for no residue here
    modification = vermouth.molecule.Modification(name='B-cap-both')
    for atomname in ('SC', 'BB'):
        modification.add_node(atomname, **{'atomname': atomname, 'element': 'C',
                                           'resname': 'B', 'PTM_atom': False})
        ptm_name = 'H' + atomname
        modification.add_node(ptm_name, **{'atomname': ptm_name, 'element': 'H',
                                           'atype': 'P1', 'mass': 1.0, 'charge': 0.0,
                                           'PTM_atom': True})
        modification.add_edge(atomname, ptm_name)
    force_field.modifications['B-cap-both'] = modification

    targets = _find_modifications(_meta_molecule_of(force_field, molecule))
    assert targets == []


def test_find_modifications_other_resname():
    """
    Only the modifications that describe the residue itself are considered.
    """
    force_field, molecule = _build_capped_chain()
    modification = vermouth.molecule.Modification(name='X-cap')
    modification.add_node('X1', **{'atomname': 'X1', 'element': 'O',
                                   'resname': 'X', 'PTM_atom': False})
    modification.add_node('HX', **{'atomname': 'HX', 'element': 'H', 'atype': 'P1',
                                   'mass': 1.0, 'charge': 0.0, 'PTM_atom': True})
    modification.add_edge('X1', 'HX')
    force_field.modifications['X-cap'] = modification

    targets = _find_modifications(_meta_molecule_of(force_field, molecule))
    # the X residue is bonded to the side chain of residue 3, so its only
    # atom is not free; no B residue may pick up a modification of X
    assert targets == []


def test_capped_residues_end_to_end(tmp_path):
    """
    A residue whose bonding operator is used in some places and capped in
    others is written as a block plus a modification, and generating a
    molecule from that force-field must cap exactly those residues that
    do not use the operator - including the interior ones.
    """
    force_field, molecule = _build_capped_chain()
    res_graph = MetaMolecule._block_graph_to_res_graph(molecule)

    # one representative residue graph per resname and graph hash, as
    # the fragment finder generates them
    unique_fragments = {}
    for res in res_graph:
        attrs = res_graph.nodes[res]
        ghash = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(attrs['graph'],
                                                                        node_attr='element')
        unique_fragments[(attrs['resname'], ghash)] = attrs['graph']
    assert len(unique_fragments) == 3

    modification_names = handle_ptms(None, unique_fragments, res_graph, molecule,
                                     force_field, {'B': 0, 'X': 0})
    assert len(force_field.modifications) == 1
    force_field.links += extract_links(molecule, force_field)
    find_termini_mods(res_graph, molecule, force_field, modification_names)

    ff_file = tmp_path / "capped.ff"
    with open(ff_file, "w") as filehandle:
        ForceFieldDirectiveWriter(forcefield=force_field, stream=filehandle).write()

    itp_file = tmp_path / "capped.itp"
    gen_params(inpath=[ff_file], seq=['{[#B][#B][#B]([#X])[#B]}'],
               outpath=itp_file, name="test")

    new_molecule = MetaMolecule.from_itp(vermouth.forcefield.ForceField('read'),
                                         itp_file, "test").molecule
    atoms = defaultdict(list)
    for _, attrs in new_molecule.nodes(data=True):
        atoms[attrs['resid']].append(attrs['atomname'])

    # residue 3 uses its side chain operator for the X residue, so it is
    # the only B residue that keeps the bare block
    assert atoms[1] == ['BB', 'SC', 'HSC']
    assert atoms[2] == ['BB', 'SC', 'HSC']
    assert atoms[3] == ['BB', 'SC']
    assert atoms[4] == ['X1']
    assert atoms[5] == ['BB', 'SC', 'HSC']
    # the caps are bonded to the side chain they belong to
    names = nx.get_node_attributes(new_molecule, 'atomname')
    for node, atomname in names.items():
        if atomname == 'HSC':
            assert [names[neigh] for neigh in new_molecule.neighbors(node)] == ['SC']
