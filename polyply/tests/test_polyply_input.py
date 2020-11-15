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

# TODO:
# add test for when version tagging is important

import textwrap
import pytest
import numpy as np
import vermouth.forcefield
import vermouth.molecule
import polyply.src.polyply_parser

class TestPolyply:
    @staticmethod
    def test_dangling_bond():
        lines = """
        [ moleculetype ]
        ; name nexcl.
        PEO         1
        ;
        [ atoms ]
        1  SN1a    1   PEO   COC  1   0.000  45
        [ bonds ]
        ; back bone bonds
        1  2   1   0.37 7000
        """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        polyply.src.polyply_parser.read_polyply(lines, ff)

        link_bond = vermouth.molecule.Interaction(
                atoms=["COC", "+COC"], parameters=['1', '0.37', '7000'], meta={"version":1},)
        assert ff.blocks['PEO'].interactions['bonds'] == []
        assert ff.links[0].interactions['bonds'][0] == link_bond

    @staticmethod
    def test_multiple_interactions():
        lines = """
        [ moleculetype ]
        ; name nexcl.
        PEO         1
        ;
        [ atoms ]
        1  SN1a    1   PEO   COC  1   0.000  45
        [ dihedrals ]
        1  2  3  4     1    180.00    1.96   1
        1  2  3  4     1     0        0.18   2
        1  2  3  4     1     0        0.33   3
        """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        polyply.src.polyply_parser.read_polyply(lines, ff)

        dih  = [vermouth.molecule.Interaction(
                atoms=["COC", "+COC", "++COC", "+++COC"], parameters=['1','180.00','1.96','1'], meta={'version':3}),
               vermouth.molecule.Interaction(
                atoms=["COC", "+COC", "++COC", "+++COC"], parameters=['1','0','0.18','2'], meta={'version':2}),
               vermouth.molecule.Interaction(
                atoms=["COC", "+COC", "++COC", "+++COC"], parameters=['1','0','0.33','3'], meta={'version':1})]

        assert len(ff.links) == 1
        assert ff.links[0].interactions['dihedrals'] == dih

    @staticmethod
    def test_exclusions():
        lines = """
        [ moleculetype ]
        ; name nexcl.
        PEO         1
        ;
        [ atoms ]
        1  SN1a    1   PEO   COC  1   0.000  45
        [ exclusions ]
        1  3
        """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        polyply.src.polyply_parser.read_polyply(lines, ff)

        excl = [vermouth.molecule.Interaction(
                atoms=["COC", "++COC"], parameters=[], meta={"version":1},)]

        assert ff.links[0].interactions['exclusions'] == excl


    @staticmethod
    @pytest.mark.parametrize('lines, total, blocks, links', (
        ("""
        [ moleculetype ]
        ; name nexcl.
        PEO         1
        ;
        [ atoms ]
        1  SN1a    1   PEO   EO1  1   0.000  45
        2  SN1a    1   PEO   EO2  1   0.000  45
        [ bonds ]
        ; back bone bonds
        1  2   1   0.37 7000
        2  3   1   0.37 7000
        """,
        1,
        {"bonds":[ vermouth.molecule.Interaction(
                atoms=[0, 1], parameters=['1', '0.37', '7000'], meta={},)]},
        {"bonds":[vermouth.molecule.Interaction(
                atoms=["EO2", "+EO1"], parameters=['1', '0.37', '7000'], meta={"version":1},)]}
        ),
        ("""
         [ moleculetype ]
         PS 1
         [ atoms ]
            1    STY            1  PS       R1       1     0.00000E+00   45
            2    STY            1  PS       R2       2     0.00000E+00   45
            3    STY            1  PS       R3       3     0.00000E+00   45
            4    SCY            1  PS       B        4     0.00000E+00   45
         [ bonds ]
            1     4   1     0.27 8000
            4     5   1     0.27 8000

         [ angles ]
           4     1     2    1   136  100
           4     1     3    1   136  100
           ; links
           4     5     6    1   136  100
           4     5     7    1   136  100
           1     4     5    1   120   25
           4     5     8    1    52  550
        """,
        5,
        {"bonds": [vermouth.molecule.Interaction(
                atoms=[0, 3], parameters=['1', '0.27', '8000'], meta={},)],
         "angles":[vermouth.molecule.Interaction(
                atoms=[3, 0, 1], parameters=['1', '136', '100'], meta={},),
                   vermouth.molecule.Interaction(
                atoms=[3, 0, 2], parameters=['1', '136', '100'], meta={},)]},
        {"bonds": [vermouth.molecule.Interaction(
                atoms=["B", "+R1"], parameters=['1', '0.27', '8000'], meta={'version':1},)],
         "angles":[
                vermouth.molecule.Interaction(
                atoms=["B", "+R1", "+R2"], parameters=['1', '136', '100'], meta={'version':1},),
                vermouth.molecule.Interaction(
                atoms=["B", "+R1", "+R2"], parameters=['1', '136', '100'], meta={'version':1},),
                vermouth.molecule.Interaction(
                atoms=["R1", "B", "+R1"], parameters=['1', '120', '25'], meta={'version':1},),
                vermouth.molecule.Interaction(
                atoms=["B", "+R1", "+B"], parameters=['1', '52', '550'], meta={'version':1},)]},
        )))

    def test_split_link_block(lines, total, blocks, links):
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        polyply.src.polyply_parser.read_polyply(lines, ff)
        # test the correct total number of links is produced
        assert len(ff.links) == total
        # check if each link has the length one
        for link in ff.links:
            assert len(link.interactions) == 1
            key = list(link.interactions.keys())[0]
            values = link.interactions[key]
            print(values)
            assert len(values) == 1
            assert len(link.edges) != 0
