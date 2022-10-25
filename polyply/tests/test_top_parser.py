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

from collections import defaultdict
import textwrap
import pytest
import vermouth.forcefield
import vermouth.molecule
import polyply
from polyply.src.topology import Topology


class TestTopParsing:
    @staticmethod
    @pytest.mark.parametrize('lines, attr, value', (
        # Refers, by index, to an atom out of range
        ("""
        [ defaults ]
        1.0   1.0   no   1.0     1.0
        """,
         "defaults",
         {"nbfunc": 1.0,
          "comb-rule": 1.0,
          "gen-pairs": "no",
          "fudgeLJ": 1.0,
          "fudgeQQ": 1.0}
         ),
        #martini style defaults
        #gen-pairs set by default
        ("""
        [ defaults ]
        1.0   1.0
        """,
         "defaults",
         {"nbfunc": 1.0,
          "comb-rule": 1.0,
          "gen-pairs": "no"}
         ),
        # martini like atomtypes
        ("""
        [ atomtypes ]
        P6 72.0 0.000 A 0.0 0.0
        """,
         "atom_types",
         {"P6": {"mass": 72.0,
                "atom_num": None,
                "charge": 0.000,
                "ptype": "A",
                "bond_type": None,
                "nb1": 0.0,
                "nb2": 0.0}}
         ),
        # amber/charrm like atomtypes
        ("""
        [ atomtypes ]
        H0           1       1.008   0.0000  A   2.47135e-01  6.56888e-02
        """,
        "atom_types",
        {"H0": {"mass": 1.008,
                "atom_num": 1.0,
                "charge": 0.000,
                "ptype": "A",
                "bond_type": None,
                "nb1": 2.47135e-01,
                "nb2": 6.56888e-02}}
         ),
        # check OPLS type atom defs.
        ("""
        [ atomtypes ]
        opls_001   C   6      12.01100     0.500       A    3.75000e-01  4.39320e-01 ; SIG
        """,
         "atom_types",
        {"opls_001":  {"mass": 12.01100,
                       "atom_num": 6,
                       "charge": 0.500,
                       "ptype": "A",
                       "bond_type": "C",
                       "nb1": 3.75000e-01,
                       "nb2": 4.39320e-01}}
         ),
        ("""
        [ nonbond_params ]
        OM      O         1    1.9670816e-03  8.5679450e-07
        """,
        "nonbond_params",
        {frozenset(["OM", "O"]):{"f": 1,
                                 "nb1": 1.9670816e-03,
                                 "nb2": 8.5679450e-07}}
        ),
        ("""
        [ bondtypes ]
        OM      O         1    1.9670816e-03  8.5679450e-07
        """,
        "types",
        {"bonds": {("OM", "O"): [(["1", "1.9670816e-03", "8.5679450e-07"], None)]}}
         ),
        ("""
        [ dihedraltypes ]
        CEL1    CEL1    CTL2    CTL2     9  1.800000e+02  3.807440e+00      1
        CEL1    CEL1    CTL2    CTL2     9  1.800000e+02  7.531200e-01      2
        CEL1    CEL1    CTL2    CTL2     9  1.800000e+02  7.112800e-01      3
        """,
        "types",
        {"dihedrals": {("CEL1", "CEL1", "CTL2", "CTL2"): [(["9", "1.800000e+02", "3.807440e+00", "1"], None),
                                                          (["9", "1.800000e+02", "7.531200e-01", "2"], None),
                                                          (["9", "1.800000e+02", "7.112800e-01", "3"], None)]}}
        ),

        ("""
        [ system ]
        some multiline
        description of a system
        """,
         "description",
         ["some multiline", "description of a system"]
         )))
    def test_directives(lines, attr, value):
        new_lines = textwrap.dedent(lines)
        new_lines = new_lines.splitlines()
        force_field = vermouth.forcefield.ForceField(name='test_ff')
        top = Topology(force_field, name="test")
        polyply.src.top_parser.read_topology(new_lines, top)
        assert getattr(top, attr) == value

    @staticmethod
    @pytest.mark.parametrize('def_statements, bonds',
         (("""#ifdef FLEXIBLE
         [ bonds ]
         1  2
         2  3
         #endif""",
         [vermouth.molecule.Interaction(
             atoms=[0, 1], parameters=[], meta={"ifdef": "FLEXIBLE"}),
          vermouth.molecule.Interaction(
              atoms=[1, 2], parameters=[], meta={"ifdef": "FLEXIBLE"})]),
         ("""#ifndef FLEXIBLE
         [ bonds ]
         1   2
         2   3
         #endif""",
         [vermouth.molecule.Interaction(
             atoms=[0, 1], parameters=[], meta={"ifndef": "FLEXIBLE"}),
          vermouth.molecule.Interaction(
             atoms=[1, 2], parameters=[], meta={"ifndef": "FLEXIBLE"})]),
         ("""[ bonds ]
         1   2
         #ifdef FLEXIBLE
         2   3
         #endif
         3  4""",
         [vermouth.molecule.Interaction(
             atoms=[0, 1], parameters=[], meta={}),
          vermouth.molecule.Interaction(
             atoms=[1, 2], parameters=[], meta={"ifdef": "FLEXIBLE"}),
          vermouth.molecule.Interaction(
             atoms=[2, 3], parameters=[], meta={})])
         ))
    def test_ifdefs(def_statements, bonds):
        """
        test the handling if ifdefs and ifndefs at
        different positions in itp-file
        """
        lines = """
        [ moleculetype ]
        GLY 1

        [ atoms ]
        1 P4 1 ALA BB 1
        2 P4 1 ALA SC1 1
        3 P4 1 ALA SC2 1
        4 P4 1 ALA SC3 1
        """
        new_lines = lines + def_statements
        new_lines = textwrap.dedent(new_lines)
        new_lines = new_lines.splitlines()
        force_field = vermouth.forcefield.ForceField(name='test_ff')
        top = Topology(force_field, name="test")
        polyply.src.top_parser.read_topology(new_lines, top)
        print(top.force_field.blocks["GLY"].interactions['bonds'])
        assert top.force_field.blocks["GLY"].interactions['bonds'] == bonds

    @staticmethod
    @pytest.mark.parametrize('def_statements',
          ("""
          #ifdef random
          #include "random.itp"
          #endif
          """,
          """
          #define random
          #ifndef random
          #include "random.itp"
          #endif
          """))
    def test_ifdefs_files(def_statements):
        """
        test files in ifdef/ifndef statements are only
        read if a define has been read before
        """
        new_lines = textwrap.dedent(def_statements)
        new_lines = new_lines.splitlines()
        force_field = vermouth.forcefield.ForceField(name='test_ff')
        top = Topology(force_field, name="test")
        polyply.src.top_parser.read_topology(new_lines, top)
        assert True

    @staticmethod
    @pytest.mark.parametrize('def_statements, result',
        (("""#define pH5""",
          {"pH5": True}
         ),
        ("""#define ga25 1.025
         """,
          {"ga25": ["1.025"]}
        ),
        ("""
         [ moleculetype ]
         GLY 1
         [ atoms ]
         1 P4 1 ALA BB 1
         #define FLEXIBLE
         """,
        {"FLEXIBLE": True}
        )))
    def test_define(def_statements, result):
        """
        test the handling of define statements in
        """
        new_lines = textwrap.dedent(def_statements)
        new_lines = new_lines.splitlines()
        force_field = vermouth.forcefield.ForceField(name='test_ff')
        top = Topology(force_field, name="test")
        polyply.src.top_parser.read_topology(new_lines, top)
        assert top.defines == result

    @staticmethod
    def test_itp_handling():
       """
       test if multiple itps are read correctly
       """
       lines = """
       [ defaults ]
       1   1   no   1.0     1.0
       [ moleculetype ]
       GLY 1
       [ atoms ]
       1 P4 1 ALA BB 1
       [ moleculetype ]
       LYS 1
       [ atoms ]
       1 P4 1 ALA BB 1
       [ molecules ]
       LYS  2
       """
       new_lines = textwrap.dedent(lines).splitlines()
       force_field = vermouth.forcefield.ForceField(name='test_ff')
       top = Topology(force_field, name="test")
       polyply.src.top_parser.read_topology(new_lines, top)
       assert top.defaults == {"nbfunc": 1,
                               "comb-rule": 1,
                               "gen-pairs": "no",
                               "fudgeLJ": 1.0,
                               "fudgeQQ": 1.0}
       # check that there are two LYS in topology
       assert len(top.molecules) == 2
       for mol in top.molecules:
           assert mol.mol_name == "LYS"
       # check that both are not the same instance of meta_molecule
       assert top.molecules[0] is not top.molecules[1]
       # also check that the high-res molecule is not the same
       assert top.molecules[0].molecule is not top.molecules[1].molecule
       # we also should have each molecule as block in the force
       # field. Again we only check that it is there because
       # the actual parsing is done elsewhere
       assert len(force_field.blocks) == 2

    @staticmethod
    @pytest.mark.parametrize('lines', (
        """
         #fdef FLEXIBLE
         #define A
         #endif
         """,
        """
         #ifdef FLEXIBLE
         """,
        """
         #endif
         """,
        """
         #ifdef FLEXIBLE
         #define A
         #ifdef RANDOM
         #define B
         """,
        """
         #if
         """))
    def test_def_fails(lines):
        """
        test if incorrectly formatted ifdefs raise
        appropiate error
        """
        new_lines = textwrap.dedent(lines)
        new_lines = new_lines.splitlines()
        force_field = vermouth.forcefield.ForceField(name='test_ff')
        top = Topology(force_field, name="test")
        with pytest.raises(IOError):
            polyply.src.top_parser.read_topology(new_lines, top)

    @staticmethod
    def test_buckingham_fail():
        lines = """
        [ defaults ]
        2   1   no   1.0     1.0
        """
        new_lines = textwrap.dedent(lines)
        new_lines = new_lines.splitlines()
        force_field = vermouth.forcefield.ForceField(name='test_ff')
        top = Topology(force_field, name="test")
        with pytest.raises(IOError):
            polyply.src.top_parser.read_topology(new_lines, top)

    @staticmethod
    def test_atom_type_fail():
        lines = """
        [ atomtypes ]
        opls_001   C   6      12.01100     0.500       A    3.75000e-01  4.39320e-01 random; SIG
        """
        new_lines = textwrap.dedent(lines)
        new_lines = new_lines.splitlines()
        force_field = vermouth.forcefield.ForceField(name='test_ff')
        top = Topology(force_field, name="test")
        with pytest.raises(IOError):
            polyply.src.top_parser.read_topology(new_lines, top)

    @staticmethod
    @pytest.mark.parametrize('lines', (
        """
        ********
        [ defaults ]
        1   1   no   1.0     1.0
        """,
        """
        [ defaults ]
        1   1   no   1.0     1.0
        [ cmaptypes ]
        something something
        """,
        """
        [ defaults ]
        1   1   no   1.0     1.0
        [implicit_genborn_params]
        something something
        """
         ))
    def test_skip_directives(lines):
        """
        Test that directives which currently cannot be read and
        or interpreted are simply skipped and don't cause an
        error when reading the topology file.
        """
        new_lines = textwrap.dedent(lines)
        new_lines = new_lines.splitlines()
        force_field = vermouth.forcefield.ForceField(name='test_ff')
        top = Topology(force_field, name="test")
        polyply.src.top_parser.read_topology(new_lines, top)
        assert top.defaults == {"nbfunc":1.0,
                                "comb-rule":1.0,
                                "gen-pairs":'no',
                                "fudgeLJ":1.0,
                                "fudgeQQ":1.0}

def test_consistency():
    """
    This test checks that all interaction formats defined
    in TOPDirector.atom_idxs also have a corresponding
    method in ITPDirector.METH_DICT.
    """
    ff = vermouth.forcefield.ForceField(name='test_ff')
    top = Topology(ff, name="test")
    top_director = polyply.src.top_parser.TOPDirector(top)
    for inter_type in top_director.atom_idxs:
        assert (tuple(['moleculetype', inter_type]) in top_director.METH_DICT or \
                tuple([inter_type]) in top_director.METH_DICT)
