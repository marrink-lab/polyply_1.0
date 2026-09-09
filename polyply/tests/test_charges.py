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
Test the charge modification functions used in itp_to_ff.
"""
import textwrap
from types import SimpleNamespace
import pytest
from pathlib import Path
import networkx as nx
import vermouth
import polyply
from polyply.src.meta_molecule import MetaMolecule
from polyply.src.charges import balance_charges, set_charges

def test_set_charges():
    # linear A-B-B-B-A chain; each of the three B residues carries
    # different charges, and set_charges must pick the one with the
    # highest betweenness centrality (the middle one, resid 3) rather
    # than just the first or last match
    molecule = vermouth.molecule.Molecule()
    charges_by_resid = {1: (0.0, 0.0), 2: (0.1, -0.1), 3: (0.9, -0.9),
                        4: (0.2, -0.2), 5: (0.0, 0.0)}
    for resid in range(1, 6):
        bb = (resid - 1) * 2
        sc = bb + 1
        resname = 'B' if resid in (2, 3, 4) else 'A'
        bb_charge, sc_charge = charges_by_resid[resid]
        molecule.add_node(bb, resid=resid, resname=resname, atomname='BB', charge=bb_charge)
        molecule.add_node(sc, resid=resid, resname=resname, atomname='SC1', charge=sc_charge)
        molecule.add_edge(bb, sc)
        if resid > 1:
            molecule.add_edge(bb - 2, bb)
    res_graph = MetaMolecule._block_graph_to_res_graph(molecule)

    block = vermouth.molecule.Block()
    block.add_nodes_from([('BB', {'atomname': 'BB', 'charge': 0.0}),
                          ('SC1', {'atomname': 'SC1', 'charge': 0.0})])

    set_charges(block, res_graph, 'B')

    assert block.nodes['BB']['charge'] == 0.9
    assert block.nodes['SC1']['charge'] == -0.9


def test_balance_charges_single_atom():
    # a block with fewer than two atoms is returned untouched, since
    # there is nothing to redistribute charge between
    block = vermouth.molecule.Block()
    block.add_node('BB', charge=0.37)
    result = balance_charges(block, charge=0.0)
    assert result.nodes['BB']['charge'] == 0.37


def test_balance_charges_already_balanced():
    # if the total charge already matches the target within `tol`, the
    # optimization is skipped entirely and charges are left untouched
    lines = """
    [ moleculetype ]
    test 1
    [ atoms ]
    1 P4 1 GLY BB  1
    2 P3 1 GLY SC1 2
    [ bonds ]
    1 2 1 0.2 100
    """
    lines = textwrap.dedent(lines).splitlines()
    ff = vermouth.forcefield.ForceField(name='test_ff')
    polyply.src.polyply_parser.read_polyply(lines, ff)
    block = ff.blocks['test']
    nx.set_node_attributes(block, {0: 0.4, 1: -0.4}, 'charge')

    balance_charges(block, charge=0.0, tol=10**-5)

    new_charges = nx.get_node_attributes(block, 'charge')
    assert new_charges == {0: 0.4, 1: -0.4}


@pytest.mark.parametrize('bondtype_key', (
    ('P4', 'P3'),  # same order _get_bonds looks atypes up in
    ('P3', 'P4'),  # reversed order - must fall back to the reversed lookup
))
def test_balance_charges_bondtypes_from_topology(bondtype_key):
    # some force fields (e.g. Charmm) define bonds by atom type rather
    # than giving an explicit length per bond; balance_charges must then
    # look the bond length up from `topology.types['bonds']` instead of
    # from the interaction's own parameters, trying both atype orders
    lines = """
    [ moleculetype ]
    test 1
    [ atoms ]
    1 P4 1 GLY BB  1
    2 P3 1 GLY SC1 2
    [ bonds ]
    1 2 1
    """
    lines = textwrap.dedent(lines).splitlines()
    ff = vermouth.forcefield.ForceField(name='test_ff')
    polyply.src.polyply_parser.read_polyply(lines, ff)
    block = ff.blocks['test']
    nx.set_node_attributes(block, {0: 0.3, 1: -0.1}, 'charge')

    # a plain namespace is enough - _get_bonds only reads topology.types,
    # but the truthiness check in _get_bonds would treat an *empty*
    # vermouth.molecule.Molecule as falsy (it is a graph with 0 nodes)
    topology = SimpleNamespace(types={'bonds': {bondtype_key: [(['1', '0.33', '1000'], {})]}})

    balance_charges(block, charge=0.0, topology=topology, tol=10**-5, decimals=5)

    new_charges = nx.get_node_attributes(block, 'charge')
    assert pytest.approx(sum(new_charges.values()), abs=0.0001) == 0.0


def test_balance_charges_bondtypes_missing_topology_raises():
    # same as above but without a topology to fall back on; the bond
    # length genuinely cannot be determined, so this must raise rather
    # than silently guessing
    lines = """
    [ moleculetype ]
    test 1
    [ atoms ]
    1 P4 1 GLY BB  1
    2 P3 1 GLY SC1 2
    [ bonds ]
    1 2 1
    """
    lines = textwrap.dedent(lines).splitlines()
    ff = vermouth.forcefield.ForceField(name='test_ff')
    polyply.src.polyply_parser.read_polyply(lines, ff)
    block = ff.blocks['test']
    nx.set_node_attributes(block, {0: 0.3, 1: -0.1}, 'charge')

    with pytest.raises(ValueError):
        balance_charges(block, charge=0.0, topology=None, tol=10**-5)


def test_balance_charges_bondtypes_not_in_topology_raises():
    # a topology is given, but it simply does not have an entry (in
    # either atype order) for this particular bond's atom types; this
    # must raise a clear error rather than an UnboundLocalError from an
    # unset `params` variable
    lines = """
    [ moleculetype ]
    test 1
    [ atoms ]
    1 P4 1 GLY BB  1
    2 P3 1 GLY SC1 2
    [ bonds ]
    1 2 1
    """
    lines = textwrap.dedent(lines).splitlines()
    ff = vermouth.forcefield.ForceField(name='test_ff')
    polyply.src.polyply_parser.read_polyply(lines, ff)
    block = ff.blocks['test']
    nx.set_node_attributes(block, {0: 0.3, 1: -0.1}, 'charge')

    topology = SimpleNamespace(types={'bonds': {('Q1', 'Q2'): [(['1', '0.33', '1000'], {})]}})

    with pytest.raises(ValueError):
        balance_charges(block, charge=0.0, topology=topology, tol=10**-5)


@pytest.mark.parametrize('charges, target',(
    ({0: 0.2, 1: -0.4, 2: 0.23, 3: 0.001},
     0.0,),
    ({0: 0.6, 1: -0.2, 2: 0.5, 3: 0.43},
     0.5,),
    ({0: -0.633, 1: -0.532, 2: 0.512, 3: 0.0},
     -0.6,),
))
def test_balance_charges(charges, target):
    lines = """
    [ moleculetype ]
    test 1
    [ atoms ]
    1 P4 1 GLY BB  1
    2 P3 1 GLY SC1 2
    3 P2 1 ALA SC2 3
    4 P2 1 ALA SC3 3
    [ bonds ]
    1 2 1 0.2 100
    2 3 1 0.6 700
    3 4 1 0.2 700
    """
    lines = textwrap.dedent(lines).splitlines()
    ff = vermouth.forcefield.ForceField(name='test_ff')
    polyply.src.polyply_parser.read_polyply(lines, ff)
    block = ff.blocks['test']
    nx.set_node_attributes(block, charges, 'charge')
    balance_charges(block, topology=None, charge=target, tol=10**-5, decimals=5)
    new_charges = nx.get_node_attributes(block, 'charge')
    assert pytest.approx(sum(new_charges.values()),abs=0.0001) == target
