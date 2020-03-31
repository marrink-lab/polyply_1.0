# Copyright 2018 University of Groningen
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
import polyply.src.parsers

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
        polyply.src.parsers.read_polyply(lines, ff)

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
        polyply.src.parsers.read_polyply(lines, ff)

        dih  = [vermouth.molecule.Interaction(
                atoms=["COC", "+COC", "++COC", "+++COC"], parameters=['1','180.00','1.96','1'], meta={'version':3}),
               vermouth.molecule.Interaction(
                atoms=["COC", "+COC", "++COC", "+++COC"], parameters=['1','0','0.18','2'], meta={'version':2}),
               vermouth.molecule.Interaction(
                atoms=["COC", "+COC", "++COC", "+++COC"], parameters=['1','0','0.33','3'], meta={'version':1})]

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
        polyply.src.parsers.read_polyply(lines, ff)

        excl = [vermouth.molecule.Interaction(
                atoms=["COC", "++COC"], parameters=[], meta={"version":1},)]

        assert ff.links[0].interactions['exclusions'] == excl


    @staticmethod
    def test_split_link_block():
        lines = """
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
        """
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        polyply.src.parsers.read_polyply(lines, ff)

        block_bond = vermouth.molecule.Interaction(
                atoms=[0, 1], parameters=['1', '0.37', '7000'], meta={},)

        link_bond = vermouth.molecule.Interaction(
                atoms=["EO2", "+EO1"], parameters=['1', '0.37', '7000'], meta={"version":1},)

        assert ff.blocks['PEO'].interactions['bonds'][0] == block_bond
        assert ff.links[0].interactions['bonds'][0] == link_bond
