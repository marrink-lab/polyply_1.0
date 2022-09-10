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
import networkx as nx
import vermouth.forcefield
import vermouth.ffinput
from vermouth.molecule import Block, Molecule, Interaction, Link
import polyply
from polyply.src.meta_molecule import (MetaMolecule, Monomer)
from polyply.src.apply_links import MatchError, ApplyLinks


@pytest.mark.parametrize('resids, order, expected',(
                        ([1, 1, 1], [1, 1, 1], True),
                        ([1, 2, 3], [1, 1, 1], False),
                        ([1, 2, 3], [0, 1, 2], True),
                        ([1, 2, 2], [0, 1, 1], True),
                        ([1, 2, 3], [0, '>', '>>'], True),
                        ([1, 2, 3], [0, '>', '>'], False),
                        ([1, 2, 3], ['<', '<<', 0], False),
                        ([1, 2, 3], ['<<', '<', 0], True),
                        ([1, 2, 3], ['<', '<', 0], False)))
def test_check_relative_orders(resids, order, expected):
    status = polyply.src.apply_links._check_relative_order(resids, order)
    assert status == expected

@pytest.mark.parametrize('attributes, ignore, expected',(
                        ({'resid': 1}, [], [0, 1]),
                        ({'resid': 1, 'type': 'A'}, [], [0]),
                        ({'type': 'A'}, [], [0, 2]),
                        ({}, [], [0, 1, 2, 3]),
                        ({'name': 'C'}, [], [2, 3]),
                        ({'order': 1, 'type': 'D'}, ['type'], [2, 3]),
                        ({'resid': 2, 'name': 'B'}, ['resid'], [1]),
                        ({'order': 0, 'name': 'A'}, ['name'], [0, 1])))
def test_find_atoms(attributes, ignore, expected):
    molecule = nx.Graph()
    molecule.add_nodes_from([(0 , {'resid': 1, 'order': 0, 'name': 'A', 'type': 'A'}),
                            (1 , {'resid': 1, 'order': 0, 'name': 'B', 'type': 'B'}),
                            (2 , {'resid': 2, 'order': 1, 'name': 'C', 'type': 'A'}),
                            (3 , {'resid': 3, 'order': 1, 'name': 'C', 'type': 'C'})])
    atoms = polyply.src.apply_links.find_atoms(molecule, ignore=ignore, **attributes)
    assert set(atoms) == set(expected)

@pytest.fixture
def example_meta_molecule():
    """
    Example molecule with three residues each with
    different attributes for testing the assignment
    of links.

    Names:
    -------
    BB - BB1 - BB - BB1 - BB2 - BB - BB1
          |          |                |
         SC1        SC1              SC1
          |          |                |
         SC2        SC2              SC2

    Nodes:
    ------
     0  - 1    4  - 5 - 6   9 - 10
          |         |           |
          2         7           11
          |         |           |
          3         8           12
    """
    force_field = vermouth.forcefield.ForceField("test")
    block_A = Block(force_field=force_field)
    block_A.add_nodes_from([(0 , {'resid': 1,  'name': 'BB',  'atype': 'A', 'charge': 0.0, 'other': 'A'}),
                            (1 , {'resid': 1,  'name': 'BB1', 'atype': 'B', 'charge': 0.0, 'other': 'A'}),
                            (2 , {'resid': 1,  'name': 'SC1', 'atype': 'C', 'charge': 0.0, 'other': 'A'}),
                            (3 , {'resid': 1,  'name': 'SC2', 'atype': 'D', 'charge': 1.0, 'other': 'A'})])
    block_A.add_edges_from([(0, 1), (1, 2), (2, 3)])

    block_B = Block(force_field=force_field)
    block_B.add_nodes_from([(0 , {'resid': 1,  'name': 'BB', 'atype': 'A', 'charge': 0.0}),
                            (1 , {'resid': 1,  'name': 'BB1', 'atype': 'B', 'charge': 0.0}),
                            (2 , {'resid': 1,  'name': 'BB2', 'atype': 'A', 'charge': 0.0}),
                            (3 , {'resid': 1,  'name': 'SC1', 'atype': 'A', 'charge': -0.5}),
                            (4 , {'resid': 1,  'name': 'SC2', 'atype': 'C', 'charge': 0.5})])
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

@pytest.mark.parametrize('link_nodes, link_to_resid, expected',(
                        # first and second residue name-based
                        ([(0, {'name': 'BB1'}),
                          (1, {'name': 'BB'})],
                          {0: 0, 1: 1},
                          {0: 1, 1: 4}
                        ),
                        # second third residue name-based
                        ([(0, {'name': 'BB2'}),
                          (1, {'name': 'BB'})],
                          {0: 1, 1: 2},
                          {0: 6, 1: 9}
                        ),
                        # third and first residue name-based
                        ([(0, {'name': 'BB'}),
                          (1, {'name': 'BB1'})],
                          {0: 0, 1: 2},
                          {0: 0, 1: 10}
                        ),
                        # selection type based
                        ([(0, {'name': 'BB', 'atype': 'A'}),
                          (1, {'atype': 'B'})],
                          {0: 0, 1: 2},
                          {0: 0, 1: 10}
                        ),
                        ))
def test_match_link_and_residue_atoms(example_meta_molecule,
                                      link_nodes,
                                      link_to_resid,
                                      expected):

    link = nx.Graph()
    link.add_nodes_from(link_nodes)
    match = polyply.src.apply_links.match_link_and_residue_atoms(example_meta_molecule,
                                                                 link,
                                                                 link_to_resid)
    assert match == expected

@pytest.mark.parametrize('link_nodes, link_to_resid',(
                        # one non-matching name
                        ([(0, {'name': 'QB1'}),
                          (1, {'name': 'BB'})],
                          {0: 0, 1: 1}
                        ),
                        # both names not matching
                        ([(0, {'name': 'QB2'}),
                          (1, {'name': 'QB'})],
                          {0: 1, 1: 2}
                        ),
                        # one additional attribute not matching
                        ([(0, {'name': 'BB', 'atype': 'Q'}),
                          (1, {'name': 'BB1', 'charge': -1})],
                          {0: 0, 1: 2}
                        ),
                        # more than one matching atom
                        ([(0, {'atype': 'A'}),
                          (1, {'other': 'A'})],
                          {0: 0, 1: 2}
                        ),
                        ))
def test_match_link_and_residue_atoms_fail(example_meta_molecule,
                                      link_nodes,
                                      link_to_resid):

    link = nx.Graph()
    link.add_nodes_from(link_nodes)

    with pytest.raises(polyply.src.apply_links.MatchError):
         polyply.src.apply_links.match_link_and_residue_atoms(example_meta_molecule,
                                                              link,
                                                              link_to_resid)

@pytest.mark.parametrize('''link_defs,
                          link_to_resids,
                          inter_types,
                          link_inters,
                          link_non_edges,
                          link_patterns,
                          expected_nodes,
                          expected_inters''',
                        (
                        # 0 - simple check single residue
                        ([[(0, {'name': 'BB1'}),
                          (1, {'name': 'BB'})]],
                         [{0: 0, 1: 1}],
                         ['bonds'],
                         [[vermouth.molecule.Interaction(atoms=(0, 1),
                                                        parameters=['1', '0.33', '500'],
                                                        meta={})]],
                         [[]],
                         [[]],
                         [[(1, 4, 1)]],
                         [[(0, 0)]]
                        ),
                        # 1 - simple check single residue reverse
                        ([[(0, {'name': 'BB'}),
                          (1, {'name': 'BB1'})]],
                         [{0: 1, 1: 0}],
                         ['bonds'],
                         [[vermouth.molecule.Interaction(atoms=(0, 1),
                                                        parameters=['1', '0.33', '500'],
                                                        meta={})]],
                         [[]],
                         [[]],
                         [[(4, 1, 1)]],
                         [[(0, 0)]]
                        ),
                        # 2 - add replace with node
                        ([[(0, {'name': 'BB1', 'replace': {'charge': 1.0}}),
                          (1, {'name': 'BB'})]],
                         [{0: 0, 1: 1}],
                         ['bonds'],
                         [[vermouth.molecule.Interaction(atoms=(0, 1),
                                                        parameters=['1', '0.33', '500'],
                                                        meta={})]],
                         [[]],
                         [[]],
                         [[(1, 4, 1)]],
                         [[(0, 0)]]
                        ),
                        # 3 - test multiple versions don't overwrite
                        ([[(0, {'name': 'BB1', 'replace': {'charge': 1.0}}),
                          (1, {'name': 'BB'})]],
                         [{0: 0, 1: 1}],
                         ['bonds'],
                         [[vermouth.molecule.Interaction(atoms=(0, 1),
                                                        parameters=['1', '0.33', '500'],
                                                        meta={"version": 1}),
                          vermouth.molecule.Interaction(atoms=(0, 1),
                                                        parameters=['1', '0.33', '600'],
                                                        meta={"version": 2})]],
                          [[]],
                          [[]],
                          [[(1, 4, 1), (1, 4, 2)]],
                          [[(0, 0), (0, 1)]]
                        ),
                        # 4 - test multiple links which don't overwrite
                        ([[(0, {'name': 'BB1', 'replace': {'charge': 1.0}}),
                           (1, {'name': 'BB'})],
                         [(0, {'name': 'BB2'}), (1, {'name': 'BB'})]],
                         [{0: 0, 1: 1},
                          {0: 1, 1: 2}],
                         ['bonds', 'bonds'],
                         [[vermouth.molecule.Interaction(atoms=(0, 1),
                                                        parameters=['1', '0.33', '500'],
                                                        meta={"version": 1})],
                          [vermouth.molecule.Interaction(atoms=(0, 1),
                                                        parameters=['1', '0.33', '600'],
                                                        meta={"version": 1})]],
                          [[], []],
                          [[], []],
                          [[(1, 4, 1)], [(6, 9, 1)]],
                          [[(0, 0), (1, 0)]]
                        ),
                        # 5 - test multiple links which do overwrite
                        ([[(0, {'name': 'BB1'}), (1, {'name': 'BB'})],
                          [(0, {'name': 'BB1'}), (1, {'name': 'BB'})]],
                         [{0: 0, 1: 1},
                          {0: 0, 1: 1}],
                         ['bonds', 'bonds'],
                         [[vermouth.molecule.Interaction(atoms=(0, 1),
                                                        parameters=['1', '0.33', '500'],
                                                        meta={"version": 1})],
                          [vermouth.molecule.Interaction(atoms=(0, 1),
                                                        parameters=['1', '0.33', '600'],
                                                        meta={"version": 1})]],
                          [[], []],
                          [[], []],
                          [[(1, 4, 1)]],
                          [[(1, 0)]]
                        ),
                        # 6 - test non-edges which is OK
                        ([[(0, {'name': 'BB1'}),
                           (1, {'name': 'BB'})],
                           [(0, {'name': 'BB1'}),
                           (1, {'name': 'BB'}),
                           (2, {'name': 'BB1'}),
                           (3, {'name': 'BB2'})]],
                         [{0: 0, 1: 1},
                          {0: 0, 1: 1, 2: 1, 3: 1}],
                         ['bonds', 'dihedrals'],
                         [[vermouth.molecule.Interaction(atoms=(0, 1),
                                                        parameters=['1', '0.33', '500'],
                                                        meta={"version": 1})],
                          [vermouth.molecule.Interaction(atoms=(0, 1, 2, 3),
                                                        parameters=['1', '0.33', '500'],
                                                        meta={})]],
                         [[], [[3 , {'atomname':'BB8' , 'order': 1}]]],
                         [[], []],
                         [[(1, 4, 1)],[(1, 4, 5, 6, 1)]],
                         [[(0, 0)], [(1, 0)]]
                        ),
                        # 7 - test pattern that works
                        ([[(0, {'name': 'BB1','replace': {'label': 'A'}}),
                           (1, {'name': 'BB', 'replace': {'label': 'A'}})],
                          [(0, {'name': 'BB2', 'replace': {'label': 'B'}}),
                           (1, {'name': 'BB', 'replace': {'label': 'B'}})],
                          [(0, {'name': 'BB1'}),
                           (1, {'name': 'BB'}),
                           (2, {'name': 'BB1'}),
                           (3, {'name': 'BB2'})]],
                         [{0: 0, 1: 1}, {0: 1, 1: 2}, {0: 0, 1: 1, 2: 1, 3: 1}],
                         ['bonds', 'bonds', 'dihedrals'],
                         [[vermouth.molecule.Interaction(atoms=(0, 1),
                                                        parameters=['1', '0.33', '500'],
                                                        meta={"version": 1})],
                          [vermouth.molecule.Interaction(atoms=(0, 1),
                                                        parameters=['1', '0.33', '500'],
                                                        meta={"version": 1})],
                          [vermouth.molecule.Interaction(atoms=(0, 1, 2, 3),
                                                        parameters=['1', '0.33', '500'],
                                                        meta={})]],
                         [[], [], []],
                         [[], [], [[[0, {'label': 'A', 'name': 'BB1'}], [1, {'label': 'A', 'name': 'BB'}],
                                   [2, {'name': 'BB1'}], [3, {'label': 'B', 'name': 'BB2'}]]]],
                         [[(1, 4, 1)], [(6, 9, 1)], [(1, 4, 5, 6, 1)]],
                         [[(0, 0)], [(1, 0)], [(2, 0)]]
                        ),
                        # 8 - simple check single residue but both nodes are the same
                        ([[(0, {'name': 'BB'}),
                           (1, {'name': 'BB1'}),
                           (2, {'name': 'SC1'})]],
                         [{0: 0, 1: 0, 2:0}],
                         ['angles'],
                         [[vermouth.molecule.Interaction(atoms=(0, 1, 2),
                                                        parameters=['1', '124', '500'],
                                                        meta={})]],
                         [[]],
                         [[]],
                         [[(0, 1, 2, 1)]],
                         [[(0, 0)]]
                        ),
                        # 9 - test node get scheduled for removal
                        ([[(0, {'name': 'BB'}),
                           (1, {'name': 'BB1'}),
                           (2, {'name': 'SC1'}),
                           (3, {'name': 'SC2', 'replace': {'atomname': None}})]],
                         [{0: 0, 1: 0, 2:0, 3:0}],
                         ['angles'],
                         [[vermouth.molecule.Interaction(atoms=(0, 1, 2),
                                                        parameters=['1', '124', '500'],
                                                        meta={})]],
                         [[]],
                         [[]],
                         [[(0, 1, 2, 1)]],
                         [[(0, 0)]]
                        ),
                        ))
def test_apply_link_to_residue(example_meta_molecule,
                               link_defs,
                               link_to_resids,
                               inter_types,
                               link_inters,
                               link_non_edges,
                               link_patterns,
                               expected_nodes,
                               expected_inters):
    links = []
    processor = ApplyLinks()
    for link_nodes, link_to_resid, interactions, non_edges, patterns, inter_type in zip(link_defs,
                                                                                        link_to_resids,
                                                                                        link_inters,
                                                                                        link_non_edges,
                                                                                        link_patterns,
                                                                                        inter_types):
        link = Link()
        link.add_nodes_from(link_nodes)
        link.interactions[inter_type] = interactions
        link.non_edges = non_edges
        link.patterns = patterns
        link.make_edges_from_interaction_type(inter_type)
        processor.apply_link_between_residues(example_meta_molecule,
                                              link,
                                              link_to_resid)

    for node in processor.nodes_to_remove:
        node_name = example_meta_molecule.molecule.nodes[node]['name']
        for node in link.nodes:
            if link.nodes[node]['name'] == node_name:
                assert link.nodes[node]['replace']['atomname'] is None

    for nodes, inter_idxs, inter_type in zip(expected_nodes, expected_inters, inter_types):
        for inter_nodes, inter_idx in zip(nodes, inter_idxs):
            interaction = link_inters[inter_idx[0]][inter_idx[1]]
            new_interactions = processor.applied_links[inter_type][inter_nodes][0]
            assert new_interactions.atoms == tuple(inter_nodes[:-1])
            assert new_interactions.parameters == interaction.parameters
            assert new_interactions.meta == interaction.meta


@pytest.mark.parametrize('''link_defs,
                          link_to_resids,
                          inter_types,
                          link_inters,
                          link_non_edges,
                          link_patterns,
                          ''',
                        (
                        # 1 - simple check single residue where atom-name is amiss
                        ([[(0, {'name': 'QB1'}),
                          (1, {'name': 'BB'})]],
                         [{0: 0, 1: 1}],
                         ['bonds'],
                         [[vermouth.molecule.Interaction(atoms=(0, 1),
                                                        parameters=['1', '0.33', '500'],
                                                        meta={})]],
                         [[]],
                         [[]],
                        ),
                        # 2 - test non-edges which doesn't apply
                        # -> here there is an edge present where the non-edge
                        #    is specified
                        ([[(0, {'name': 'BB1'}),
                           (1, {'name': 'BB'})],
                          [(0, {'name': 'BB2'}),
                           (1, {'name': 'BB'})],
                          [(0, {'name': 'BB1'}),
                           (1, {'name': 'BB'}),
                           (2, {'name': 'BB1'}),
                           (3, {'name': 'BB2'})]],
                         [{0: 0, 1: 1}, {0: 1, 1: 2}, {0: 0, 1: 1, 2: 1, 3: 1}],
                         ['bonds', 'bonds', 'dihedrals'],
                         [[vermouth.molecule.Interaction(atoms=(0, 1),
                                                        parameters=['1', '0.33', '500'],
                                                        meta={"version": 1})],
                          [vermouth.molecule.Interaction(atoms=(0, 1),
                                                        parameters=['1', '0.33', '500'],
                                                        meta={"version": 1})],
                          [vermouth.molecule.Interaction(atoms=(0, 1, 2, 3),
                                                        parameters=['1', '0.33', '500'],
                                                        meta={})]],
                         [[], [], [[3 , {'name':'BB' , 'order': 1}]]],
                         [[], [], []],
                        ),
                        # 3 - test pattern that fails because no match can be found
                        ([[(0, {'name': 'BB1','replace': {'label': 'A'}}),
                           (1, {'name': 'BB', 'replace': {'label': 'A'}})],
                          [(0, {'name': 'BB2', 'replace': {'label': 'B'}}),
                           (1, {'name': 'BB', 'replace': {'label': 'B'}})],
                          [(0, {'name': 'BB1'}),
                           (1, {'name': 'BB'}),
                           (2, {'name': 'BB1'}),
                           (3, {'name': 'BB2'})]],
                         [{0: 0, 1: 1}, {0: 1, 1: 2}, {0: 0, 1: 1, 2: 1, 3: 1}],
                         ['bonds', 'bonds', 'dihedrals'],
                         [[vermouth.molecule.Interaction(atoms=(0, 1),
                                                        parameters=['1', '0.33', '500'],
                                                        meta={"version": 1})],
                          [vermouth.molecule.Interaction(atoms=(0, 1),
                                                        parameters=['1', '0.33', '500'],
                                                        meta={"version": 1})],
                          [vermouth.molecule.Interaction(atoms=(0, 1, 2, 3),
                                                        parameters=['1', '0.33', '500'],
                                                        meta={})]],
                         [[], [], []],
                         [[], [], [[[0, {'label': 'C', 'name': 'BB1'}], [1, {'label': 'A', 'name': 'BB'}],
                                   [2, {'name': 'BB1'}], [3, {'label': 'B', 'name': 'BB2'}]]]],
                        )))
def test_apply_link_fail(example_meta_molecule,
                         link_defs,
                         link_to_resids,
                         inter_types,
                         link_inters,
                         link_non_edges,
                         link_patterns):
    links = []

    with pytest.raises(polyply.src.apply_links.MatchError):
        processor = ApplyLinks()
        for link_nodes, link_to_resid, interactions, non_edges, patterns, inter_type in zip(link_defs,
                                                                                            link_to_resids,
                                                                                            link_inters,
                                                                                            link_non_edges,
                                                                                            link_patterns,
                                                                                            inter_types):
            link = Link()
            link.add_nodes_from(link_nodes)
            link.interactions[inter_type] = interactions
            link.non_edges = non_edges
            link.patterns = patterns
            link.make_edges_from_interaction_type(inter_type)
            processor.apply_link_between_residues(example_meta_molecule,
                                                  link,
                                                  link_to_resid)

@pytest.mark.parametrize('links, interactions, edges, inttype',(
   ("""
   [ link ]
   [ molmeta ]
   by_atom_ID true
   [ bonds ]
   1   2  1  0.350  1250
   """,
   [vermouth.molecule.Interaction(atoms=(0, 1),
                                  parameters=['1', '0.350', '1250'],
                                  meta={})],
   1,
   'bonds'
   ),
   ("""
   [ link ]
   [ molmeta ]
   by_atom_ID true
   [ angles ]
   2  3  4  1  125  250
   """,
   [vermouth.molecule.Interaction(atoms=(1, 2, 3),
                                  parameters=['1', '125', '250'],
                                  meta={})],
   2,
   'angles')
   ))
def test_add_explicit_link(links, interactions, edges, inttype):
    lines = """
    [ moleculetype ]
    GLY  1
    [ atoms ]
    ;id  type resnr residu atom cgnr   charge
     1   SP2   1     GLY    BB     1      0
    [ moleculetype ]
    ALA  1
    [ atoms ]
    ;id  type resnr residu atom cgnr   charge
     1   SP2   1     ALA    BB     1      0
     2   SC2   1     ALA    SC1     1      0
    """
    lines = lines + links
    lines = textwrap.dedent(lines).splitlines()
    force_field = vermouth.forcefield.ForceField(name='test_ff')
    vermouth.ffinput.read_ff(lines, force_field)

    new_mol = force_field.blocks['GLY'].to_molecule()
    new_mol.merge_molecule(force_field.blocks['GLY'])
    new_mol.merge_molecule(force_field.blocks['ALA'])

    polyply.src.apply_links.apply_explicit_link(
        new_mol, force_field.links[0])
    assert new_mol.interactions[inttype] == interactions
    assert len(new_mol.edges) == edges

@pytest.mark.parametrize('links, error_type',
                         (("""
     [ link ]
     [ molmeta ]
     by_atom_ID true
     [ bonds ]
     BB  +BB  1  0.350  1250
     """,
                           ValueError
                           ),
                          ("""
     [ link ]
     [ molmeta ]
     by_atom_ID true
     [ angles ]
     1   8  1  125  250
     """,
                           IOError)
                          ))
def test_explicit_link_failure(links, error_type):
    lines = """
    [ moleculetype ]
    GLY  1
    [ atoms ]
    ;id  type resnr residu atom cgnr   charge
     1   SP2   1     GLY    BB     1      0
    [ moleculetype ]
    ALA  1
    [ atoms ]
    ;id  type resnr residu atom cgnr   charge
     1   SP2   1     ALA    BB     1      0
     2   SC2   1     ALA    SC1    1      0
    """
    lines = lines + links
    lines = textwrap.dedent(lines).splitlines()
    force_field = vermouth.forcefield.ForceField(name='test_ff')
    vermouth.ffinput.read_ff(lines, force_field)

    new_mol = force_field.blocks['GLY'].to_molecule()
    new_mol.merge_molecule(force_field.blocks['GLY'])
    new_mol.merge_molecule(force_field.blocks['ALA'])

    with pytest.raises(error_type):
        polyply.src.apply_links.apply_explicit_link(
            new_mol, force_field.links[0])

def test_expand_exclusions():
    mol = vermouth.Molecule()
    mol.nrexcl = 1
    mol.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 4)])
    nx.set_node_attributes(mol, {0:1, 1:2, 2:2, 3:2, 4:2}, "exclude")
    mol = polyply.src.apply_links.expand_excl(mol)
    ref_excl = [frozenset([0, 1]),
                frozenset([1, 2]),
                frozenset([2, 0]),
                frozenset([2, 3]),
                frozenset([2, 4]),
                frozenset([3, 4]),
                frozenset([3, 1]),
                frozenset([4, 1])]

    assert len(ref_excl) == len(mol.interactions["exclusions"])
    for excl in mol.interactions["exclusions"]:
        assert frozenset(excl.atoms) in ref_excl
