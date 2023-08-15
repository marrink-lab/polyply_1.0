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
import numpy as np
import vermouth.forcefield
import vermouth.molecule
import polyply.src.polyply_parser

class TestITP:
    @staticmethod
    def test_moleculetype():
        lines = """
        [ moleculetype ]
        GLY  3
        """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        polyply.src.polyply_parser.read_polyply(lines, ff)
        assert 'GLY' in ff.blocks
        assert ff.blocks['GLY'].nrexcl == 3

    @staticmethod
    def test_multiple_moleculetype():
        lines = """
            [ moleculetype ]
            GLY  3

            [ moleculetype ]
            VAL 2
            """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        polyply.src.polyply_parser.read_polyply(lines, ff)
        assert set(ff.blocks) == set(['GLY', 'VAL'])
        assert ff.blocks['GLY'].nrexcl == 3
        assert ff.blocks['VAL'].nrexcl == 2

    @staticmethod
    def test_atoms():
        lines = """
        [ moleculetype ]
        GLY 1

        [ atoms ]
        1 P4  1  ALA  BB  1
        2 P3\t1\tALA  SC1 2 -3
        3 P2  1  ALA  SC2 3 -3 72
        """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        polyply.src.polyply_parser.read_polyply(lines, ff)
        block = ff.blocks['GLY']
        assert len(block.nodes) == 3
        assert block.nodes[0] == {'index':1,'atomname': 'BB', 'atype': 'P4',
                                     'resname': 'ALA', 'resid': 1,
                                     'charge_group': 1}
        assert block.nodes[1] == {'index':2,'atomname': 'SC1', 'atype': 'P3',
                                      'resname': 'ALA', 'resid': 1,
                                      'charge_group': 2, 'charge': -3.0}
        assert block.nodes[2] == {'index':3,'atomname': 'SC2', 'atype': 'P2',
                                      'resname': 'ALA', 'resid': 1,
                                      'charge_group': 3,'charge':-3,'mass':72}

    @staticmethod
    def test_fixed_number_interaction():
        """
        Define an interaction for which the number of atoms required is known.
        """
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
        2 3 4 0.6 700
        [ angles ]
        1  2  3 params
        2  3  4 params
        """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        polyply.src.polyply_parser.read_polyply(lines, ff)
        block = ff.blocks['GLY']

        bonds = [
            vermouth.molecule.Interaction(
                atoms=[0, 1], parameters=['1', '0.2', '100'], meta={},
            ),
            vermouth.molecule.Interaction(
                atoms=[1, 2], parameters=['4', '0.6', '700'],meta={},
            ),
        ]

        angles = [
            vermouth.molecule.Interaction(
                atoms=[0, 1, 2], parameters=['params'], meta={},
            ),
            vermouth.molecule.Interaction(
                atoms=[1, 2, 3], parameters=['params'],meta={},
            ),
        ]
        assert block.interactions['bonds'] == bonds
        assert block.interactions['angles'] == angles

    @staticmethod
    def test_interaction_by_index():
        """
        Create an interaction and refer to atoms by index.
        """
        lines = """
        [ moleculetype ]
        GLY 1

        [ atoms ]
        1 P4 1 ALA BB 1
        2 P3 1 ALA SC1 2
        3 P2 1 ALA SC2 3

        [ bonds ]
        1 2   1 0.2 100
        1 3   4 0.6 700
        """

        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        polyply.src.polyply_parser.read_polyply(lines, ff)
        block = ff.blocks['GLY']

        bonds = [
            vermouth.molecule.Interaction(
                atoms=[0, 1], parameters=['1', '0.2', '100'], meta={},
            ),
            vermouth.molecule.Interaction(
                atoms=[0, 2], parameters=['4', '0.6', '700'],
                meta={},
            ),

        ]
        assert block.interactions['bonds'] == bonds

    @staticmethod
    @pytest.mark.parametrize('interaction_lines', (
        # Refers, by index, to an atom out of range
        """
        [ bonds ]
        1 6
        """,
        """
        [ bonds ]
        1 -6
        """,
        # Refers, by name, to a non-defined atom (SC2 does not exist)
        """
        [ bonds ]
        BB SC2
        """,
        # One missing atom in the definition (refers to only one atom instead of 2)
        """
        [ bonds ]
        BB
        """,
        # Prefixed atom in a block interaction
        """
        [ bonds ]
        BB +SC1
        """,
        """
        [ bonds ]
        BB -SC1
        """,
        """
        [ bonds ]
        BB <SC1
        """,
        """
        [ bonds ]
        BB >SC1
        """,
    ))
    def test_interaction_fail_reference(interaction_lines):
        """
        Faulty atom references in interactions lead to an exception.
        """
        lines = """
        [ moleculetype ]
        GLY 1

        [ atoms ]
        1 P4 1 ALA BB 1
        ; The following atoms have names that should not be valid
        2 P4 2 TRP +SC1 1
        3 P4 2 TRP -SC1 1
        4 P4 2 TRP <SC1 1
        5 P4 2 TRP >SC1 1
        """
        lines = textwrap.dedent(lines)
        lines = lines.splitlines()
        lines += textwrap.dedent(interaction_lines).splitlines()

        ff = vermouth.forcefield.ForceField(name='test_ff')
        with pytest.raises(IOError):
            vermouth.ffinput.read_ff(lines, ff)

    @staticmethod
    @pytest.mark.parametrize('section_name', (
         'non-edges', 'patterns', 'features', '!bonds', '!angles'
     ))
    def test_invalid_section_fails(section_name):
        """
        Using the sections that are invalid in a block fails.
        """
        lines = """
        [ moleculetype ]
        GLY 1

        [ atoms ]
        1 P4 1 ALA BB 1
        2 P4 1 ALA SC1 1

        [ {} ]
        BB SC1
        """
        lines = textwrap.dedent(lines).format(section_name)
        lines = lines.splitlines()

        ff = vermouth.forcefield.ForceField(name='test_ff')
        with pytest.raises(IOError):
             polyply.src.polyply_parser.read_polyply(lines, ff)

    @staticmethod
    def test_atomname_not_unique():
        """
        Atom names are unique in an FF block but not an itp.
        """
        lines = """
        [ moleculetype ]
        GLY 1

        [ atoms ]
        1 P4 1 ALA BB 1
        2 P4 1 ALA BB 1
        """
        lines = textwrap.dedent(lines)
        lines = lines.splitlines()

        ff = vermouth.forcefield.ForceField(name='test_ff')
        polyply.src.polyply_parser.read_polyply(lines, ff)

    @staticmethod
    def test_index_unique():
        """
        Atom indices need to be unique.
        """
        lines = """
        [ moleculetype ]
        GLY 1

        [ atoms ]
        1 P4 1 ALA BB 1
        1 P4 1 ALA SC1 1
        """
        lines = textwrap.dedent(lines)
        lines = lines.splitlines()

        ff = vermouth.forcefield.ForceField(name='test_ff')
        with pytest.raises(IOError):
             polyply.src.polyply_parser.read_polyply(lines, ff)

    @staticmethod
    def test_virtual_sitesn():
        """
        test if index and atoms are curretly distinguished for
        this type of interaction
        """
        lines = """
        [ moleculetype ]
        GLY 1

        [ atoms ]
        1 P4 1 ALA BB 1
        2 P4 1 ALA SC1 1
        3 P4 1 ALA SC2 1
        4 P4 1 ALA SC3 1
        5 P4 1 ALA SC4 1
        6 VS 1 ALA SC5 1

        [ virtual_sitesn ]
        6  2  1 2 3 4
        6  10  1 2 3 4
        """
        lines = textwrap.dedent(lines)
        lines = lines.splitlines()

        ff = vermouth.forcefield.ForceField(name='test_ff')
        polyply.src.polyply_parser.read_polyply(lines, ff)
        VS1 = vermouth.molecule.Interaction(
                atoms=[5, 0, 1, 2, 3], parameters=['2'], meta={},
            )

        VS2 = vermouth.molecule.Interaction(
                atoms=[5, 0, 1, 2, 3], parameters=['10'], meta={},
            )

        assert ff.blocks['GLY'].interactions['virtual_sitesn'][0] == VS1
        assert ff.blocks['GLY'].interactions['virtual_sitesn'][1] == VS2

    @staticmethod
    @pytest.mark.parametrize('def_statements, bonds',
        (("""#ifdef FLEXIBLE
         [ bonds ]
         1  2
         2  3
         #endif""",
         [vermouth.molecule.Interaction(
               atoms=[0, 1], parameters=[], meta={"ifdef":"FLEXIBLE"}),
          vermouth.molecule.Interaction(
               atoms=[1, 2], parameters=[], meta={"ifdef":"FLEXIBLE"})]),
        ("""#ifndef FLEXIBLE
         [ bonds ]
         1   2
         2   3
         #endif""",
         [vermouth.molecule.Interaction(
             atoms=[0, 1], parameters=[], meta={"ifndef":"FLEXIBLE"}),
          vermouth.molecule.Interaction(
             atoms=[1, 2], parameters=[], meta={"ifndef":"FLEXIBLE"})]),
        ("""[ bonds ]
         1   2
         #ifdef FLEXIBLE
         2   3
         #endif
         3  4""",
         [vermouth.molecule.Interaction(
              atoms=[0, 1], parameters=[], meta={}),
          vermouth.molecule.Interaction(
              atoms=[1, 2], parameters=[],meta={"ifdef":"FLEXIBLE"}),
          vermouth.molecule.Interaction(
              atoms=[2, 3], parameters=[],meta={})])
        ))

    def test_defs(def_statements, bonds):
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
        ff = vermouth.forcefield.ForceField(name='test_ff')
        polyply.src.polyply_parser.read_polyply(new_lines, ff)
        assert ff.blocks['GLY'].interactions['bonds'] == bonds

    @staticmethod
    @pytest.mark.parametrize('def_fail_statements',(
         """[ atoms ]
         1 P4 1 ALA BB 1
         2 P4 1 ALA SC1 1
         [ bonds ]
         1   2
         #fdef FLEXIBLE
         2   3
         #endif
         3  4""",
         """
         [ atoms ]
         1 P4 1 ALA BB 1
         2 P4 1 ALA SC1 1
         3 P4 1 ALA SC2 1
         4 P4 1 ALA SC3 1
         [ bonds ]
         1   2
         #ifdef FLEXIBLE
         2   3
         3  4""",
         """[ atoms ]
         1   2
         #ifdef FLEXIBLE
         2   3
         #endif
         3  4""",
         """[ atoms ]
         1 P4 1 ALA SC1 1
         2 P4  1 ALA SC1 1
         [ bonds ]
         #include random
         """,
         """[atoms]
         1 P4  1 ALA SC1 1
         2 P4  1 ALA SC1 1
         [ bonds ]
         1   2   A B
         #endif
         """,
         """[ atoms ]
         1 P4  1 ALA SC1 1
         2 P4  1 ALA SC1 1
         3 P4  1 ALA SC1 1
         [ bonds ]
         #ifdef FLEXIBLE
         1   2   A  B
         #ifdef RANDOM
         1    3  A   B
         """,
         """[ atoms ]
         1 P4 1 ALA SC1 1
         2 P4  1 ALA SC1 1
         [ bonds ]
         1   2 A B
         #if
         """))
    def test_def_fails(def_fail_statements):
        """
        test if incorrectly formatted ifdefs raise
        appropiate error
        """
        lines = """
        [ moleculetype ]
        GLY 1
        """
        new_lines = lines + def_fail_statements
        new_lines = textwrap.dedent(new_lines)
        new_lines = new_lines.splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        with pytest.raises(IOError):
             polyply.src.polyply_parser.read_polyply(new_lines, ff)

    @staticmethod
    def test_multiple_interactions():
        """
        test if we can multiple interactions
        using the same atom indices but different
        parameters.
        """
        lines = """
        [ moleculetype ]
        GLY 1

        [ atoms ]
        1 P4 1 ALA BB 1
        2 P4 1 ALA SC1 1
        3 P4 1 ALA SC2 1
        4 P4 1 ALA SC3 1
        [ dihedrals ]
        1   2   3   4  1  A
        1   2   3   4  1  B
        1   2   3   4  1  C
        """

        lines = textwrap.dedent(lines)
        lines = lines.splitlines()

        ff = vermouth.forcefield.ForceField(name='test_ff')
        polyply.src.polyply_parser.read_polyply(lines, ff)

        dih = [vermouth.molecule.Interaction(
                atoms=[0, 1, 2, 3], parameters=['1','A'], meta={}),
              vermouth.molecule.Interaction(
                atoms=[0, 1, 2, 3], parameters=['1','B'], meta={}),
              vermouth.molecule.Interaction(
                atoms=[0, 1, 2, 3], parameters=['1','C'], meta={})]

        assert ff.blocks['GLY'].interactions['dihedrals'] == dih

    @staticmethod
    def test_excluions():
        """
        test if can read exclusions with
        variable number of excluded atoms properly.
        """
        lines = """
        [ moleculetype ]
        GLY 1

        [ atoms ]
        1 P4 1 ALA BB 1
        2 P4 1 ALA SC1 1
        3 P4 1 ALA SC2 1
        4 P4 1 ALA SC3 1
        [ exclusions ]
        1   2   3   4
        2   3   4
        3   4
        """

        lines = textwrap.dedent(lines)
        lines = lines.splitlines()

        ff = vermouth.forcefield.ForceField(name='test_ff')
        polyply.src.polyply_parser.read_polyply(lines, ff)

        excl = [vermouth.molecule.Interaction(
                 atoms=[0, 1, 2, 3], parameters=[], meta={}),
               vermouth.molecule.Interaction(
                 atoms=[1, 2, 3], parameters=[], meta={}),
               vermouth.molecule.Interaction(
                 atoms=[2, 3], parameters=[], meta={})]

        assert ff.blocks['GLY'].interactions['exclusions'] == excl


    @staticmethod
    @pytest.mark.parametrize('test_lines',(
         """[ atoms ]
         1 P4 1 ALA BB 1
         1 P4 1 ALA BB 1
         """,
         """[ atoms ]
         -1   P4 1 ALA BB 1
         """,
         """[ atoms ]
         AA   P4   ALA BB 1
         """))
    def test_atom_block_errors(test_lines):
        """
        test if incorrectly formatted ifdefs raise
        appropiate error
        """
        lines = """
        [ moleculetype ]
        GLY 1
        """
        new_lines = lines + test_lines
        new_lines = textwrap.dedent(new_lines)
        new_lines = new_lines.splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        with pytest.raises(IOError):
             polyply.src.polyply_parser.read_polyply(new_lines, ff)

