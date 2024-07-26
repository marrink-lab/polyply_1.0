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
from vermouth.molecule import Interaction
from vermouth.gmx.itp_read import read_itp
import polyply
from polyply import TEST_DATA
from polyply.src.linalg_functions import center_of_geometry
from polyply.src.generate_templates import (find_atoms,
                                            _expand_inital_coords,
                                            _relabel_interaction_atoms,
                                            compute_volume, map_from_CoG,
                                            extract_block, GenerateTemplates,
                                            find_interaction_involving,
                                            _extract_template_graphs)
from .example_fixtures import example_meta_molecule

class TestGenTemps:

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
        4 VS 1 ALA SC3 3

        [ bonds ]
        1 2 1 0.2 100
        2 3 1 0.6 700

        [ virtual_sitesn ]
        4 2 3 1
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
      @pytest.mark.parametrize('atom_defs', (
      [1, 2, 3],
      (1, 2, 3)))
      def test_relabel_interaction(atom_defs):
         interaction = Interaction(atom_defs, ["1", "ga_2"], {"test": "value"})
         mapping = {1: "A", 2: "B", 3: "C"}
         new_interaction = _relabel_interaction_atoms(interaction, mapping)
         assert new_interaction.atoms == ["A", "B", "C"]
         assert new_interaction.parameters == ["1", "ga_2"]
         assert new_interaction.meta == {"test": "value"}

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
         template_graph = ff.blocks['GLY'].to_molecule()
         new_block = extract_block(molecule, template_graph, {})
         for node in ff.blocks["GLY"]:
             atomname = ff.blocks["GLY"].nodes[node]["atomname"]
             assert ff.blocks["GLY"].nodes[node] == new_block.nodes[atomname]
         for inter_type in ff.blocks["GLY"].interactions:
             len(ff.blocks["GLY"].interactions[inter_type]) == len(new_block.interactions[inter_type])

      @staticmethod
      @pytest.mark.parametrize('volumes', (
                               None,
                               {"PMMA": 0.55},
      ))
      def test_run_molecule(volumes):
          top = polyply.src.topology.Topology.from_gmx_topfile(TEST_DATA / "topology_test" / "system.top", "test")
          top.gen_pairs()
          if volumes:
            top.volumes = volumes
          top.convert_nonbond_to_sig_eps()
          GenerateTemplates(topology=top, skip_filter=False, max_opt=10).run_molecule(top.molecules[0])
          graph = top.molecules[0].nodes[0]['graph']
          graph_hash = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(graph, node_attr='atomname')
          assert graph_hash in top.volumes
          if volumes:
            assert top.volumes[graph_hash] == volumes['PMMA']
          assert graph_hash in top.molecules[0].templates

      @staticmethod
      @pytest.mark.parametrize('lines, result', (
        ("""
         [ moleculetype ]
         test 1
         [ atoms ]
         1 P4 1 GLY BB 1
         2 P3 1 GLY SC1 2
         3 P2 1 ALA SC2 3
         [ bonds ]
         1 2 1 0.2 100
         2 3 1 0.2 100
         """,
         (False, Interaction(atoms=["1", "2"], parameters=["1", "0.2", "100"], meta={}),
          "bonds")),
        ("""
         [ moleculetype ]
         test 1
         [ atoms ]
         1 P4 1 GLY BB 1
         2 P3 1 GLY SC1 2
         3 VS 1 ALA SC2 3
         4 P3 1 ALA SC3 3
         [ virtual_sitesn ]
         3 1 4 2 1
         """,
         (True, Interaction(atoms=["3", "4", "2", "1"], parameters=["1"], meta={}),
          "virtual_sitesn"))))
      def test_find_interaction_involving(lines, result):
          lines = textwrap.dedent(lines).splitlines()
          ff = vermouth.forcefield.ForceField(name='test_ff')
          polyply.src.polyply_parser.read_polyply(lines, ff)
          block = ff.blocks['test']
          result = find_interaction_involving(block, 1, 2)
          assert result == result

      @staticmethod
      @pytest.mark.parametrize('lines', (
        """
         [ moleculetype ]
         test 1
         [ atoms ]
         1 P4 1 GLY BB 1
         2 P3 1 GLY SC1 2
         3 P2 1 ALA SC2 3
         [ bonds ]
         1 2 1 0.2 100
         """,
        """
         [ moleculetype ]
         test 1
         [ atoms ]
         1 P4 1 GLY BB 1
         2 P3 1 GLY SC1 2
         3 VS 1 ALA SC2 3
         4 P3 1 ALA SC3 3
         """))
      def test_find_interaction_involving_error(lines):
          lines = textwrap.dedent(lines).splitlines()
          ff = vermouth.forcefield.ForceField(name='test_ff')
          polyply.src.polyply_parser.read_polyply(lines, ff)
          block = ff.blocks['test']
          with pytest.raises(IOError):
               find_interaction_involving(block, 1, 2)

@pytest.mark.parametrize('lines, coords, volume',
    [("""
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
      """,
      np.array([[0.00, 0.0, 0.0],
                [0.47, 0.0, 0.0],
                [0.94, 0.0, 0.0],
                [1.41, 0.0, 0.0]]),
      0.9689298220201501),
    ("""
      [ moleculetype ]
      GLY 1
      [ atoms ]
      1 P2 1 ALA BB 1
      2 P1 1 ALA SC1 2
      3 P2 1 ALA SC2 3
      4 P1 1 ALA SC3 3
      [ bonds ]
      1 2 1 0.2 100
      2 3 1 0.6 700
      3 4 1 0.2 700
      """,
      np.array([[0.00, 0.0, 0.0],
                [0.47, 0.0, 0.0],
                [0.94, 0.0, 0.0],
                [1.41, 0.0, 0.0]]),
      0.861408729930223),
     ("""
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
      [ virtual_sitesn ]
      4 1 1
      """,
      np.array([[0.00, 0.0, 0.0],
                [0.47, 0.0, 0.0],
                [0.94, 0.0, 0.0],
                [0.00, 0.0, 0.0]]),
      0.8391178403537848),
     ("""
      [ moleculetype ]
      GLY 1
      [ atoms ]
      1 P1 1 ALA BB 1
      """,
      np.array([[0.00, 0.0, 0.0]]),
      0.47),
])
def test_compute_volume(lines, coords, volume):
    meta_mol = polyply.MetaMolecule()
    nonbond_params = {frozenset(["P1", "P1"]): {"nb1": 0.47, "nb2":0.5},
                      frozenset(["P2", "P2"]): {"nb1": 0.23, "nb2":0.5},
                      frozenset(["P2", "P1"]): {"nb1": 0.35, "nb2":0.5},}

    lines = textwrap.dedent(lines).splitlines()
    ff = vermouth.forcefield.ForceField(name='test_ff')
    read_itp(lines, ff)
    block = ff.blocks['GLY']
    coord_dict = {node: coord for node, coord in zip(block.nodes, coords)}
    new_vol = compute_volume(block, coord_dict, nonbond_params)
    print(new_vol)
    assert np.isclose(new_vol, volume, atol=0.000001)


@pytest.mark.parametrize('resnames, gen_template_graphs, use_resname, skip_filter', (
                        # two different residues no template_graphs
                        (['A', 'B', 'A'], [], False, False),
                        # two different residues no template_graphs
                        (['A', 'B', 'A'], [], False, True),
                        # two different residues one template_graphs
                        (['A', 'B', 'A'], [1], False, True),
                        # two different residues one template_graphs
                        (['A', 'B', 'A'], [1], False, False),
                        # here the template is indexed with the resname
                        # instead of the hash which needs to be cleared
                        (['A', 'B', 'A'], [1], True, True),
))
def test_extract_template_graphs(example_meta_molecule,
                                 resnames,
                                 gen_template_graphs,
                                 use_resname,
                                 skip_filter):
    # set the residue names
    for resname, node in zip(resnames, example_meta_molecule.nodes):
        example_meta_molecule.nodes[node]['resname'] = resname
        nx.set_node_attributes(example_meta_molecule.nodes[node]['graph'], resname, 'resname')

    # extract template graphs if needed
    template_graphs = {}
    for node in gen_template_graphs:
        graph = example_meta_molecule.nodes[node]['graph']
        nx.set_node_attributes(graph, True, 'template')
        graph_hash = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(graph, node_attr='atomname')
        if use_resname:
            resname =  example_meta_molecule.nodes[node]['resname']
            template_graphs[resname] = None
        else:
            template_graphs[graph_hash] = None

    # perfrom the grouping
    unique_graphs = _extract_template_graphs(example_meta_molecule, template_graphs, skip_filter)

    # check the outcome
    assert len(unique_graphs) == 2

    # assert that all nodes have the template attribute
    for node in example_meta_molecule.nodes:
        assert example_meta_molecule.nodes[node].get('template', False)
