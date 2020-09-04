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
import math
import vermouth.forcefield
import vermouth.molecule
import polyply.src.meta_molecule
from polyply import TEST_DATA
from polyply.src.topology import Topology

class TestTopology:

    @staticmethod
    def test_from_gmx_topfile():
        top = Topology.from_gmx_topfile(TEST_DATA+"/topology_test/system.top", "test")
        assert len(top.molecules) == 1

    @staticmethod
    def test_add_positions_from_gro():
        top = Topology.from_gmx_topfile(TEST_DATA + "/topology_test/system.top", "test")
        top.add_positions_from_file(TEST_DATA + "/topology_test/test.gro")
        for node in top.molecules[0].molecule.nodes:
            if node < 14:
                assert "position" in top.molecules[0].molecule.nodes[node].keys()

        for node in top.molecules[0].nodes:
            if node != 2:
                assert "position" in top.molecules[0].nodes[node].keys()
                assert top.molecules[0].nodes[node]["build"] == False
            else:
                assert top.molecules[0].nodes[node]["build"] == True

    @staticmethod
    def test_add_positions_from_pdb():
        top = Topology.from_gmx_topfile(TEST_DATA + "/topology_test/pdb.top", "test")
        top.add_positions_from_file(TEST_DATA + "/topology_test/test.pdb")
        for meta_mol in top.molecules:
            for node in meta_mol.molecule.nodes:
                assert "position" in meta_mol.molecule.nodes[node].keys()

        for meta_mol in top.molecules:
            for node in meta_mol.nodes:
                    assert "position" in meta_mol.nodes[node].keys()
                    assert meta_mol.nodes[node]["build"] == False

    @staticmethod
    def test_convert_to_vermouth_system():
        top = Topology.from_gmx_topfile(TEST_DATA + "/topology_test/system.top", "test")
        system = top.convert_to_vermouth_system()
        assert isinstance(system, vermouth.system.System)
        assert len(system.molecules) == 1

    @staticmethod
    @pytest.mark.parametrize('lines, outcome', (
        (
        """
        [ defaults ]
        1.0   1.0   yes  1.0     1.0
        [ atomtypes ]
        O       8 0.000 0.000  A   2.7106496e-03  9.9002500e-07
        C       8 0.000 0.000  A   1.7106496e-03  9.9002500e-07
        """,
        {frozenset(["O", "O"]): {"f": 1,
                               "nb1": 2.7106496e-03,
                               "nb2": 9.9002500e-07},
         frozenset(["C", "C"]): {"f": 1,
                               "nb1": 1.7106496e-03,
                               "nb2": 9.9002500e-07},
         frozenset(["C", "O"]): {"f": 1,
                               "nb1": 0.0022106496,
                               "nb2": 9.9002500e-07}}
        ),
        ("""
        [ defaults ]
        1.0   3.0    yes  0.5     0.5
        [ atomtypes ]
        C   C   6      12.01100     0.500       A    3.75000e-01  4.39320e-01 ; SIG
        O   O   8      15.99940    -0.500       A    2.96000e-01  8.78640e-01 ; SIG
        """,
        {frozenset(["C", "C"]): {"f": 1,
                               "nb1": 3.75000e-01,
                               "nb2": 4.39320e-01},
         frozenset(["O", "O"]): {"f": 1,
                               "nb1": 2.96000e-01,
                               "nb2": 8.78640e-01},
         frozenset(["O", "C"]): {"f": 1,
                               "nb1": 0.3355,
                               "nb2": 0.6212923022217481}}
        ),
        (
        """
        [ defaults ]
        1.0   1.0   yes  1.0     1.0
        [ atomtypes ]
        O       8 0.000 0.000  A   2.7106496e-03  9.9002500e-07
        C       8 0.000 0.000  A   1.7106496e-03  9.9002500e-07
        [ nonbond_params ]
        C    O    1     2.0     4.0
        """,
        {frozenset(["O", "O"]): {"f": 1,
                               "nb1": 2.7106496e-03,
                               "nb2": 9.9002500e-07},
         frozenset(["C", "C"]): {"f": 1,
                               "nb1": 1.7106496e-03,
                               "nb2": 9.9002500e-07},
         frozenset(["C", "O"]): {"f": 1,
                               "nb1": 2.0,
                               "nb2": 4.0}}
        )))
    def test_gen_pairs(lines, outcome):
        new_lines = textwrap.dedent(lines)
        new_lines = new_lines.splitlines()
        force_field = vermouth.forcefield.ForceField(name='test_ff')
        top =  Topology(force_field, name="test")
        polyply.src.top_parser.read_topology(new_lines, top)
        top.gen_pairs()
        for atom_pair in outcome:
            assert atom_pair in top.nonbond_params
            nb1 = top.nonbond_params[atom_pair]["nb1"]
            nb2 = top.nonbond_params[atom_pair]["nb2"]
            nb1_ref = outcome[atom_pair]["nb1"]
            nb2_ref = outcome[atom_pair]["nb2"]
            assert math.isclose(nb1, nb1_ref)
            assert math.isclose(nb2, nb2_ref)

    @staticmethod
    @pytest.mark.parametrize('lines, outcome', (
        (
        """
        [ defaults ]
        1.0   1.0   yes  1.0     1.0
        [ atomtypes ]
        O       8 0.000 0.000  A   2.7106496e-03  9.9002500e-07
        C       8 0.000 0.000  A   1.7106496e-03  9.9002500e-07
        #define ga_1  100  250
        [ moleculetype ]
        test 3
        [ atoms ]
        1 CH3   1 test C1 1   0.0 14.0
        2 CH2   1 test C2 2   0.0 12.0
        3 CH3   1 test C3 3   0.0 12.0
        [ angles ]
        1 2  3 2 ga_1
        [ system ]
        some title
        [ molecules ]
        test 1
        """,
        ["2", "100", "250"]
        ),
        # different location of define statement
        (
        """
        [ defaults ]
        1.0   1.0   yes  1.0     1.0
        #define ga_1  100  250
        [ atomtypes ]
        O       8 0.000 0.000  A   2.7106496e-03  9.9002500e-07
        C       8 0.000 0.000  A   1.7106496e-03  9.9002500e-07
        [ moleculetype ]
        test 3
        [ atoms ]
        1 CH3   1 test C1 1   0.0 14.0
        2 CH2   1 test C2 2   0.0 12.0
        3 CH3   1 test C3 3   0.0 12.0
        [ angles ]
        1 2  3 2 ga_1
        [ system ]
        some title
        [ molecules ]
        test 1
        """,
        ["2", "100", "250"]
        ),
        # two defines for one statement
        (
        """
        [ defaults ]
        1.0   1.0   yes  1.0     1.0
        #define ga_1  100
        #define gk_1  250
        [ atomtypes ]
        O       8 0.000 0.000  A   2.7106496e-03  9.9002500e-07
        C       8 0.000 0.000  A   1.7106496e-03  9.9002500e-07
        [ moleculetype ]
        test 3
        [ atoms ]
        1 CH3   1 test C1 1   0.0 14.0
        2 CH2   1 test C2 2   0.0 12.0
        3 CH3   1 test C3 3   0.0 12.0
        [ angles ]
        1 2  3 2 ga_1 gk_1
        [ system ]
        some title
        [ molecules ]
        test 1
        """,
        ["2", "100", "250"]
        )
	))
    def test_replace_defines(lines, outcome):
        new_lines = textwrap.dedent(lines)
        new_lines = new_lines.splitlines()
        force_field = vermouth.forcefield.ForceField(name='test_ff')
        top =  Topology(force_field, name="test")
        polyply.src.top_parser.read_topology(new_lines, top)
        top.replace_defines()
        assert top.molecules[0].molecule.interactions["angles"][0].parameters == outcome

    @staticmethod
    def test_convert_nonbond_to_sig_eps():
        """
        Simply test if the conversion from C6 C12 to simga epsilon
        is done properly.
        """

        force_field = vermouth.forcefield.ForceField(name='test_ff')
        top =  Topology(force_field, name="test")
        top.nonbond_params = {frozenset(["EO", "EO"]):
                             {"nb1":6.44779031E-02 , "nb2": 4.07588234E-04}}
        top.convert_nonbond_to_sig_eps()
        assert math.isclose(top.nonbond_params[frozenset(["EO", "EO"])]["nb1"], 0.43)
        assert math.isclose(top.nonbond_params[frozenset(["EO", "EO"])]["nb2"], 3.4*0.75)

