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
Fixtures for tests
"""
import textwrap
import pytest
import numpy as np
import networkx as nx
import vermouth.forcefield
import vermouth.molecule
from vermouth.molecule import Block, Molecule
from polyply.src.meta_molecule import MetaMolecule

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
    block_A.add_nodes_from([(0 , {'resid': 1,  'atomname': 'BB',  'atype': 'A', 'charge': 0.0, 'other': 'A', 'resname': 'A'}),
                            (1 , {'resid': 1,  'atomname': 'BB1', 'atype': 'B', 'charge': 0.0, 'other': 'A', 'resname': 'A'}),
                            (2 , {'resid': 1,  'atomname': 'SC1', 'atype': 'C', 'charge': 0.0, 'other': 'A', 'resname': 'A'}),
                            (3 , {'resid': 1,  'atomname': 'SC2', 'atype': 'D', 'charge': 1.0, 'other': 'A', 'resname': 'A'})])
    block_A.add_edges_from([(0, 1), (1, 2), (2, 3)])

    block_B = Block(force_field=force_field)
    block_B.add_nodes_from([(0 , {'resid': 1,  'atomname': 'BB',  'atype': 'A', 'charge': 0.0, 'resname': 'B'}),
                            (1 , {'resid': 1,  'atomname': 'BB1', 'atype': 'B', 'charge': 0.0, 'resname': 'B'}),
                            (2 , {'resid': 1,  'atomname': 'BB2', 'atype': 'A', 'charge': 0.0, 'resname': 'B'}),
                            (3 , {'resid': 1,  'atomname': 'SC1', 'atype': 'A', 'charge': -0.5, 'resname': 'B'}),
                            (4 , {'resid': 1,  'atomname': 'SC2', 'atype': 'C', 'charge': 0.5, 'resname': 'B'})])
    block_B.add_edges_from([(0, 1), (1, 2), (2, 3), (1, 4)])

    molecule = block_A.to_molecule()
    molecule.merge_molecule(block_B)
    molecule.merge_molecule(block_A)
    molecule.add_edges_from([(1, 4), (6, 9)])

    graph = MetaMolecule._block_graph_to_res_graph(molecule)
    meta_mol = MetaMolecule(graph, force_field=force_field, mol_name="test")
    meta_mol.molecule = molecule
    return meta_mol
