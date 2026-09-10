# Copyright 2024 Dr. Fabian Gruenewald
#
# Licensed under the PolyForm Noncommercial License 1.0.0;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://polyformproject.org/licenses/noncommercial/1.0.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test extraction of blocks and links from a molecule for the .ff format.
"""
import io
import textwrap
import pytest
import networkx as nx
import vermouth
from vermouth.molecule import Interaction
from vermouth.processors.do_links import match_link
import polyply
from polyply.src.meta_molecule import MetaMolecule
from polyply.src.molecule_utils import (extract_block, extract_links,
                                        find_termini_mods, find_minimal_residue)
from polyply.src.ffoutput import ForceFieldDirectiveWriter
from polyply.src.ff_parser_sub import read_ff
from polyply.src.gen_itp import gen_params
from collections import defaultdict
from .example_fixtures import example_meta_molecule

@pytest.mark.parametrize('lines, expected_bonds', (
    # simple block extraction; the template graph selects the first
    # two atoms (BB, SC1) of the four atom molecule
    ("""
     [ moleculetype ]
     test 1
     [ atoms ]
     1 P4 1 GLY BB 1
     2 P3 1 GLY SC1 2
     3 P2 1 ALA SC2 3
     4 P2 1 ALA SC3 3
     [ bonds ]
     1 2 1 0.2 100
     2 3 1 0.6 700
     3 4 1 0.2 700
     [ moleculetype ]
     GLY 1
     [ atoms ]
     1 P4 1 GLY BB 1
     2 P3 1 GLY SC1 2
     [ bonds ]
     1 2 1 0.2 100
     """,
     [Interaction(atoms=['BB', 'SC1'], parameters=['1', '0.2', '100'], meta={})]),
    # two bonds defined between the same pair of atoms; the second
    # one must be tagged with an incrementing version so that both
    # survive rather than the second silently overwriting the first
    ("""
     [ moleculetype ]
     test 1
     [ atoms ]
     1 P4 1 GLY BB 1
     2 P3 1 GLY SC1 2
     [ bonds ]
     1 2 1 0.2 100
     1 2 1 0.3 150
     [ moleculetype ]
     GLY 1
     [ atoms ]
     1 P4 1 GLY BB 1
     2 P3 1 GLY SC1 2
     """,
     [Interaction(atoms=['BB', 'SC1'], parameters=['1', '0.2', '100'], meta={}),
      Interaction(atoms=['BB', 'SC1'], parameters=['1', '0.3', '150'], meta={'version': 2})]),
))
def test_extract_block(lines, expected_bonds):
    lines = textwrap.dedent(lines).splitlines()
    ff = vermouth.forcefield.ForceField(name='test_ff')
    polyply.src.polyply_parser.read_polyply(lines, ff)
    molecule = ff.blocks['test'].to_molecule()
    template_graph = ff.blocks['GLY'].to_molecule()
    new_block = extract_block(molecule, template_graph, {})

    for node in ff.blocks["GLY"]:
        atomname = ff.blocks["GLY"].nodes[node]["atomname"]
        assert ff.blocks["GLY"].nodes[node] == new_block.nodes[atomname]

    assert new_block.interactions['bonds'] == expected_bonds


@pytest.mark.parametrize('inters, expected',(
    # simple bond spanning two residues
    ({'bonds':[Interaction(atoms=(0, 1), parameters=['1', '0.33', '500'], meta={}),
               Interaction(atoms=(1, 2), parameters=['1', '0.33', '500'], meta={}),
               Interaction(atoms=(1, 4), parameters=['1', '0.30', '500'], meta={}),
               Interaction(atoms=(4, 5), parameters=['1', '0.35', '500'], meta={}),]},
     {'bonds': [Interaction(atoms=['BB1', '+BB'],
                            parameters=['1', '0.30', '500'],
                            meta={'version': 1, 'comment': 'link'}),
               ]},
    ),
    # double version dihedral spanning two residues
    ({'dihedrals':[Interaction(atoms=(0, 1, 4, 5),
                               parameters=['9', '120', '4', '1'],
                               meta={}),
                   Interaction(atoms=(0, 1, 4, 5),
                               parameters=['9', '120', '4', '2'],
                               meta={}),
                   Interaction(atoms=(0, 1, 2, 3),
                               parameters=['9', '120', '4', '2'],
                               meta={})]
     },
     {'dihedrals': [Interaction(atoms=['BB', 'BB1', '+BB', '+BB1'],
                                parameters=['9', '120', '4', '1'],
                                meta={'version': 1, 'comment': 'link'}),
                    Interaction(atoms=['BB', 'BB1', '+BB', '+BB1'],
                                parameters=['9', '120', '4', '2'],
                                meta={'version': 2, 'comment': 'link'}),]
     },
    ),
    # four stacked GROMACS type 9 dihedrals on the same four atoms but
    # with different multiplicity/parameters; each must be kept and
    # given its own incrementing version rather than being collapsed
    ({'dihedrals':[Interaction(atoms=(0, 1, 4, 5),
                               parameters=['9', '180.00', '1.96', '1'],
                               meta={}),
                   Interaction(atoms=(0, 1, 4, 5),
                               parameters=['9', '0', '0.18', '2'],
                               meta={}),
                   Interaction(atoms=(0, 1, 4, 5),
                               parameters=['9', '0', '0.33', '3'],
                               meta={}),
                   Interaction(atoms=(0, 1, 4, 5),
                               parameters=['9', '0', '0.12', '4'],
                               meta={})]
     },
     {'dihedrals': [Interaction(atoms=['BB', 'BB1', '+BB', '+BB1'],
                                parameters=['9', '180.00', '1.96', '1'],
                                meta={'version': 1, 'comment': 'link'}),
                    Interaction(atoms=['BB', 'BB1', '+BB', '+BB1'],
                                parameters=['9', '0', '0.18', '2'],
                                meta={'version': 2, 'comment': 'link'}),
                    Interaction(atoms=['BB', 'BB1', '+BB', '+BB1'],
                                parameters=['9', '0', '0.33', '3'],
                                meta={'version': 3, 'comment': 'link'}),
                    Interaction(atoms=['BB', 'BB1', '+BB', '+BB1'],
                                parameters=['9', '0', '0.12', '4'],
                                meta={'version': 4, 'comment': 'link'}),]
     },
    ),
    # 1-5 pairs spanning 3 residues
    ({'pairs': [Interaction(atoms=(1, 9),
                            parameters=[1],
                            meta={})]},
    {'pairs': [Interaction(atoms=['BB1', '++BB'],
                           parameters=[1],
                           meta={'version': 1, 'comment': 'link'})]
    }),
    # redundant pair
    ({'pairs': [Interaction(atoms=(1, 5),
                            parameters=[1],
                            meta={}),
                Interaction(atoms=(5, 9),
                            parameters=[1],
                            meta={}),
               ],},
    {'pairs': [Interaction(atoms=['BB1', '+BB1'],
                           parameters=[1],
                           meta={'version': 1, 'comment': 'link'})]
    }),
    # the exact same bond is listed twice (e.g. picked up from two
    # separate interactions in the source topology); the duplicate
    # must be filtered out rather than kept as a second version
    ({'bonds': [Interaction(atoms=(1, 4),
                            parameters=['1', '0.30', '500'],
                            meta={}),
                Interaction(atoms=(1, 4),
                            parameters=['1', '0.30', '500'],
                            meta={}),
               ],},
    {'bonds': [Interaction(atoms=['BB1', '+BB'],
                           parameters=['1', '0.30', '500'],
                           meta={'version': 1, 'comment': 'link'})]
    }),
    # exclusions (like virtual sites) are not numbered with a version
    # or tagged with a comment, unlike regular bonded interactions
    ({'exclusions': [Interaction(atoms=(1, 4),
                                 parameters=[],
                                 meta={})],},
    {'exclusions': [Interaction(atoms=['BB1', '+BB'],
                                parameters=[],
                                meta={})]
    }),
))
def test_extract_links(example_meta_molecule, inters, expected):
    mol = example_meta_molecule.molecule
    mol.add_edges_from([(1, 4), (8, 9)])
    nx.set_node_attributes(mol, {0: "resA", 1: "resA", 2: "resA", 3: "resA",
                                 4: "resB", 5: "resB", 6: "resB", 7: "resB", 8: "resB",
                                 9: "resA", 10: "resA", 11: "resA", 12: "resA"}, "resname")
    nx.set_node_attributes(mol, {0: "BB", 1: "BB1", 2: "SC1", 3: "SC2",
                                 4: "BB", 5: "BB1", 6: "BB2", 7: "SC1", 8: "SC2",
                                 9: "BB", 10: "BB1", 11: "SC1", 12: "SC2"}, "atomname")
    mol.interactions.update(inters)
    link = extract_links(mol)[0]
    for inter_type in expected:
        assert expected[inter_type] == link.interactions[inter_type]


def _build_linear_tetramer(mod_resid_sc1_mass, bonds=()):
    """
    Build a small linear B-B-B-B molecule with realistic backbone
    connectivity (BB(res_i)-BB(res_i+1) forms the backbone, and each
    residue's SC1 hangs off its own BB), together with its residue
    graph and a force field holding the reference 'B' block. All four
    residues share the resname 'B' since termini are identified purely
    by graph degree, not by resname, so only one reference block is
    needed.

    The SC1 mass of the residue next to the left-hand terminus (resid
    2) is set to `mod_resid_sc1_mass`, while the residue next to the
    right-hand terminus (resid 3) always matches the reference block
    exactly (SC1 mass 2.0), so only the left terminus can produce a
    link with a replace statement. `bonds` are extra bond Interactions (using molecule atom
    indices) on top of the backbone/side-chain bonds, letting tests
    probe bonds inside the terminal residue itself, the bond directly
    connecting it to its neighbor, or both.
    """
    force_field = vermouth.forcefield.ForceField('test')

    molecule = vermouth.molecule.Molecule()
    molecule.add_node(0, resid=1, resname='B', atomname='BB', atype='P1', mass=1.0)
    molecule.add_node(1, resid=1, resname='B', atomname='SC1', atype='P2', mass=1.0)
    molecule.add_node(2, resid=2, resname='B', atomname='BB', atype='P1', mass=1.0)
    molecule.add_node(3, resid=2, resname='B', atomname='SC1', atype='P2', mass=mod_resid_sc1_mass)
    molecule.add_node(4, resid=3, resname='B', atomname='BB', atype='P1', mass=1.0)
    molecule.add_node(5, resid=3, resname='B', atomname='SC1', atype='P2', mass=2.0)
    molecule.add_node(6, resid=4, resname='B', atomname='BB', atype='P1', mass=1.0)
    molecule.add_node(7, resid=4, resname='B', atomname='SC1', atype='P2', mass=1.0)
    molecule.add_edges_from([(0, 2), (0, 1), (2, 4), (2, 3), (4, 6), (4, 5), (6, 7)])
    molecule.interactions['bonds'] = list(bonds)

    block_B = vermouth.molecule.Block(force_field=force_field)
    block_B.add_nodes_from([('BB', {'atype': 'P1', 'mass': 1.0}),
                            ('SC1', {'atype': 'P2', 'mass': 2.0})])
    block_B.add_edge('BB', 'SC1')
    force_field.blocks['B'] = block_B

    res_graph = MetaMolecule._block_graph_to_res_graph(molecule)
    return res_graph, molecule, force_field


def test_find_termini_mods_no_difference():
    # both neighbor residues match the reference block exactly; a link is
    # still generated for each of the two termini, because the interactions
    # at a terminus can differ from those in the chain even when the atoms
    # themselves do not, but none of the atoms gets a replace statement
    res_graph, molecule, force_field = _build_linear_tetramer(mod_resid_sc1_mass=2.0)
    find_termini_mods(res_graph, molecule, force_field)

    assert len(force_field.links) == 2
    for link in force_field.links:
        assert not any('replace' in attrs for _, attrs in link.nodes(data=True))
        assert not link.interactions


def test_find_termini_mods_with_difference():
    # the residue next to the left-hand terminus has a different SC1
    # mass than the reference block; both termini produce a link, but
    # only the left-hand one captures an atom replacement, as well as
    # all three kinds of bonds touching the modified residue: one
    # entirely inside the terminal residue itself, the bond directly
    # connecting the terminus to its neighbor, and one entirely inside
    # the (modified) neighbor
    bonds = [Interaction(atoms=(0, 1), parameters=['1', '0.20', '5000'], meta={}),
             Interaction(atoms=(0, 2), parameters=['1', '0.33', '1000'], meta={}),
             Interaction(atoms=(2, 3), parameters=['1', '0.30', '2000'], meta={})]
    res_graph, molecule, force_field = _build_linear_tetramer(mod_resid_sc1_mass=3.0, bonds=bonds)
    find_termini_mods(res_graph, molecule, force_field)

    assert len(force_field.links) == 2
    link = force_field.links[0]

    assert link.nodes['BB']['resname'] == 'B'
    assert 'replace' not in link.nodes['BB']
    assert link.nodes['SC1']['resname'] == 'B'
    assert 'replace' not in link.nodes['SC1']
    assert link.nodes['+SC1']['resname'] == 'B'
    assert link.nodes['+SC1']['replace'] == {'mass': 3.0}

    # the right-hand terminus matches the reference block, so its link
    # neither replaces an atom attribute nor overwrites an interaction
    other_link = force_field.links[1]
    assert not any('replace' in attrs for _, attrs in other_link.nodes(data=True))
    assert not other_link.interactions

    # without a non-edge, this link would also match at any interior
    # BB-BB junction that looks the same locally; the non-edge requires
    # that 'BB' has no further neighbor one residue further out (order
    # -1 relative to its own resid), which only holds at a genuine
    # chain terminus
    assert link.non_edges == [['BB', {'atomname': 'BB', 'order': -1}]]

    assert list(link.edges) == [('+BB', 'BB')]
    assert link.interactions['bonds'] == [Interaction(atoms=['BB', 'SC1'],
                                                       parameters=['1', '0.20', '5000'],
                                                       meta={}),
                                          Interaction(atoms=['BB', '+BB'],
                                                       parameters=['1', '0.33', '1000'],
                                                       meta={}),
                                          Interaction(atoms=['+BB', '+SC1'],
                                                       parameters=['1', '0.30', '2000'],
                                                       meta={})]


def _build_hexamer(mod_resid_sc1_mass):
    """
    Build a longer (six-residue) linear B chain, otherwise identical in
    style to `_build_linear_tetramer`, with bonds along the whole
    backbone and every side chain populated. Only the residue next to
    the left-hand terminus (resid 2) differs from the reference block;
    every other B-B junction in the chain is structurally identical,
    so this is used to check that a generated link only matches the
    genuine terminus and not any of the interior look-alikes.
    """
    force_field = vermouth.forcefield.ForceField('test')

    molecule = vermouth.molecule.Molecule()
    n_res = 6
    bonds = []
    for resid in range(1, n_res + 1):
        bb = (resid - 1) * 2
        sc = bb + 1
        sc_mass = 3.0 if resid == 2 else 2.0
        molecule.add_node(bb, resid=resid, resname='B', atomname='BB', atype='P1', mass=1.0)
        molecule.add_node(sc, resid=resid, resname='B', atomname='SC1', atype='P2', mass=sc_mass)
        if resid > 1:
            molecule.add_edge(bb - 2, bb)
            bonds.append(Interaction(atoms=(bb - 2, bb), parameters=['1', '0.33', '1000'], meta={}))
        molecule.add_edge(bb, sc)
        bonds.append(Interaction(atoms=(bb, sc), parameters=['1', '0.20', '5000'], meta={}))
    molecule.interactions['bonds'] = bonds

    block_B = vermouth.molecule.Block(force_field=force_field)
    block_B.add_nodes_from([('BB', {'atomname': 'BB', 'atype': 'P1', 'mass': 1.0, 'resid': 1,
                                    'resname': 'B', 'charge_group': 1, 'charge': 0.0}),
                            ('SC1', {'atomname': 'SC1', 'atype': 'P2', 'mass': 2.0, 'resid': 1,
                                    'resname': 'B', 'charge_group': 1, 'charge': 0.0})])
    block_B.add_edge('BB', 'SC1')
    block_B.nrexcl = 1
    force_field.blocks['B'] = block_B

    res_graph = MetaMolecule._block_graph_to_res_graph(molecule)
    return res_graph, molecule, force_field


def test_find_termini_mods_non_edge_rejects_interior_match():
    # a longer chain where every B-B junction looks alike apart from
    # the terminus; without the non-edge on the connecting atom, the
    # link generated for the left terminus would also match every
    # other (interior) junction, since 'replace' is ignored by graph
    # matching and every junction shares the same atomnames, resnames,
    # and local bond topology. Round-trips the force field through the
    # real .ff writer/reader (as gen_ff would) and reconstructs link
    # edges from the bonds the same way meta_molecule._make_edges does
    # downstream, so this exercises the non-edge exactly as it is used
    # in practice, not just as a raw attribute on the Link object
    res_graph, molecule, force_field = _build_hexamer(mod_resid_sc1_mass=3.0)
    find_termini_mods(res_graph, molecule, force_field)

    buf = io.StringIO()
    writer = ForceFieldDirectiveWriter(forcefield=force_field, stream=buf)
    writer.write_block_edges = False
    writer.write()

    reread_ff = vermouth.forcefield.ForceField('reread')
    read_ff(buf.getvalue().splitlines(), reread_ff)
    link = reread_ff.links[0]
    link.make_edges_from_interaction_type(type_='bonds')

    matches = list(match_link(molecule, link))
    assert matches == [{'BB': 0, 'SC1': 1, '+BB': 2, '+SC1': 3}]


def _build_variant_tetramer():
    """
    Build a linear B-B-B-B molecule in which both terminal residues have
    one extra atom that the two interior residues do not have; the extra
    atom is different at each terminus. All residues share the resname
    'B', so the interior residues define the block and the two terminal
    ones are described by a modification each.
    """
    force_field = vermouth.forcefield.ForceField('test')
    molecule = vermouth.molecule.Molecule(force_field=force_field)
    molecule.nrexcl = 1

    layout = [(1, [('BB', 'C', 'P1', 45.0), ('SC1', 'C', 'P2', 45.0), ('H1', 'H', 'P3', 1.0)]),
              (2, [('BB', 'C', 'P1', 45.0), ('SC1', 'C', 'P2', 45.0)]),
              (3, [('BB', 'C', 'P1', 45.0), ('SC1', 'C', 'P2', 45.0)]),
              (4, [('BB', 'C', 'P1', 45.0), ('SC1', 'C', 'P2', 45.0), ('O1', 'O', 'P4', 16.0)])]
    node = 0
    res_nodes = {}
    for resid, atoms in layout:
        res_nodes[resid] = []
        for atomname, element, atype, mass in atoms:
            molecule.add_node(node, atomname=atomname, element=element, atype=atype,
                              mass=mass, resid=resid, resname='B',
                              charge_group=resid, charge=0.0)
            res_nodes[resid].append(node)
            node += 1

    edges = [(0, 1), (0, 2), (3, 4), (5, 6), (7, 8), (7, 9),
             (0, 3), (3, 5), (5, 7)]
    molecule.add_edges_from(edges)
    for idx, jdx in edges:
        molecule.interactions['bonds'].append(Interaction(atoms=(idx, jdx),
                                                          parameters=['1', '0.35', '7000'],
                                                          meta={}))
    return force_field, molecule, res_nodes


def _make_variant_force_field():
    """
    Run the force-field generation steps on the molecule of
    `_build_variant_tetramer` and return the resulting force-field.
    """
    force_field, molecule, res_nodes = _build_variant_tetramer()
    res_graph = MetaMolecule._block_graph_to_res_graph(molecule)
    graphs = {resid: molecule.subgraph(nodes) for resid, nodes in res_nodes.items()}
    hashes = {resid: nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(graph,
                                                                             node_attr='element')
              for resid, graph in graphs.items()}

    fragment, modifications = find_minimal_residue([graphs[1], graphs[2], graphs[4]],
                                                   [hashes[1], hashes[2], hashes[4]])
    modification_names = {}
    for ghash, modification in modifications.items():
        force_field.modifications[modification.name] = modification
        modification_names[ghash] = modification.name

    block = extract_block(molecule, fragment, defines={})
    nx.set_node_attributes(block, 1, "resid")
    block.nrexcl = molecule.nrexcl
    force_field.blocks['B'] = block
    force_field.links += extract_links(molecule)
    find_termini_mods(res_graph, molecule, force_field, modification_names)
    return force_field, modification_names, hashes


def test_find_termini_mods_annotates_modifications():
    """
    A terminal residue that is described by a modification must be
    annotated with the name of that modification by the link, and the
    atoms of the modification must not be part of the link, because the
    link is applied before the modification.
    """
    force_field, modification_names, hashes = _make_variant_force_field()

    assert set(modification_names) == {hashes[1], hashes[4]}
    annotated = {}
    for link in force_field.links:
        for node, attrs in link.nodes(data=True):
            names = attrs.get('replace', {}).get('annotated_modifications', [])
            for name in names:
                annotated[name] = node
        # the extra atoms of the terminal residues are described by the
        # modifications, so no link may require them
        assert 'H1' not in link.nodes
        assert 'O1' not in link.nodes

    # both termini are annotated, each with its own modification
    assert set(annotated) == set(modification_names.values())
    assert annotated[modification_names[hashes[1]]] != annotated[modification_names[hashes[4]]]


def test_termini_modifications_end_to_end(tmp_path):
    """
    The modifications generated for the two termini must be applied to
    the correct terminus when a molecule is generated from the written
    force-field alone.
    """
    force_field, _, _ = _make_variant_force_field()
    ff_file = tmp_path / "variants.ff"
    with open(ff_file, "w") as filehandle:
        ForceFieldDirectiveWriter(forcefield=force_field, stream=filehandle).write()

    itp_file = tmp_path / "variants.itp"
    gen_params(inpath=[ff_file], seq=['B:4'], outpath=itp_file, name="test")

    new_force_field = vermouth.forcefield.ForceField('read')
    molecule = MetaMolecule.from_itp(new_force_field, itp_file, "test").molecule
    atoms = defaultdict(list)
    for node, attrs in molecule.nodes(data=True):
        atoms[attrs['resid']].append(attrs['atomname'])

    # only the terminal residues get the extra atom, and each terminus
    # gets the one that belongs to it
    assert atoms[1] == ['BB', 'SC1', 'H1']
    assert atoms[2] == ['BB', 'SC1']
    assert atoms[3] == ['BB', 'SC1']
    assert atoms[4] == ['BB', 'SC1', 'O1']
