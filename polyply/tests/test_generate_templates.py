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
Test linear algebra aux functions.
"""

import textwrap
import pytest
import numpy as np
import math
import networkx as nx
import vermouth
import polyply
from ..src.linalg_functions import center_of_geometry
from ..src.generate_templates import (_atoms_in_node, find_atoms,
                                            _expand_inital_coords,
                                            compute_volume, map_from_CoG,
                                            extract_block, GenerateTemplates)

class TestGenTemps:

      @staticmethod
      def test_atoms_in_node():
          G = nx.Graph()
          G.add_edges_from([(1, 2), (2, 3), (3, 4)])
          assert _atoms_in_node([1], G.nodes) == True
          assert _atoms_in_node([5], G.nodes) == False

      @staticmethod
      def test_find_atoms():
          G = nx.Graph()
          G.add_edges_from([(1, 2), (2, 3), (3, 4)])
          attrs = {1: {"resname": "test", "id": 'A'},
                   2: {"resname": "test", "id": 'B'},
                   3: {"resname": "testB", "id": 'A'},
                   4: {"resname": "testB", "id": 'B'}}
          nx.set_node_attributes(G, attrs)
          nodes = find_atoms(G, "resname", "test")
          assert nodes == [1, 2]
          nodes = find_atoms(G, "id", "A")
          assert nodes == [1, 3]

      @staticmethod
      def test_expand_inital_coords():
        lines = """
        [ moleculetype ]
        GLY 1

        [ atoms ]
        1 P4 1 ALA BB 1
        2 P3 1 ALA SC1 2
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
        block = ff.blocks['GLY']
        coords = _expand_inital_coords(block)
        assert len(coords) == 4
        for pos in coords.values():
            assert len(pos) == 3

      @staticmethod
      def test_compute_volume():
          lines = """
          [ moleculetype ]
          GLY 1

          [ atoms ]
          1 P1 1 ALA BB 1
          2 P1 1 ALA SC1 2
          3 P1 1 ALA SC2 3
          4 P1 1 ALA SC3 3

          [ bonds ]
          1 2 1 0.2 100
          2 3 1 0.6 700
          3 4 1 0.2 700
          """
          meta_mol = polyply.MetaMolecule()
          meta_mol.nonbond_params = {frozenset(["P1", "P1"]): {"nb1": 0.47, "nb2":0.5}}
          meta_mol.defaults = {"nbfunc": 2}

          lines = textwrap.dedent(lines).splitlines()
          ff = vermouth.forcefield.ForceField(name='test_ff')
          polyply.src.polyply_parser.read_polyply(lines, ff)
          block = ff.blocks['GLY']
          coords = _expand_inital_coords(block)
          vol = compute_volume(meta_mol, block, coords)
          assert vol > 0.

      @staticmethod
      def test_map_from_CoG():
         lines = """
         [ moleculetype ]
         GLY 1
         [ atoms ]
         1 P4 1 ALA BB 1
         2 P3 1 ALA SC1 2
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
         block = ff.blocks['GLY']
         coords = _expand_inital_coords(block)
         points = np.array(list(coords.values()))
         CoG = center_of_geometry(points)
         new_coords = map_from_CoG(coords)
         for coord in coords:
             coords[coord] == new_coords[coord] + CoG

      @staticmethod
      def test_extract_block():
         lines = """
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
         """
         lines = textwrap.dedent(lines).splitlines()
         ff = vermouth.forcefield.ForceField(name='test_ff')
         polyply.src.polyply_parser.read_polyply(lines, ff)
         block = ff.blocks['test']
         molecule = block.to_molecule()
         new_block = extract_block(molecule, "GLY", {})
         for node in ff.blocks["GLY"]:
             assert ff.blocks["GLY"].nodes[node] == new_block.nodes[node]
         for inter_type in ff.blocks["GLY"].interactions:
             len(ff.blocks["GLY"].interactions[inter_type]) == len(new_block.interactions[inter_type])

      @staticmethod
      def test_run_molecule():
          top = polyply.src.topology.Topology.from_gmx_topfile("test_data/topology_test/system.top", "test")
          top.gen_pairs()
          top.convert_nonbond_to_sig_eps()
          GenerateTemplates().run_molecule(top.molecules[0])
          assert "PMMA" in top.molecules[0].volumes
          assert "PMMA" in top.molecules[0].templates
