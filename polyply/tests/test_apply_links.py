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
import networkx as nx
import vermouth.forcefield
import vermouth.molecule
import polyply.src.meta_molecule
import polyply.src.map_to_molecule
import polyply.src.parsers
import polyply.src.apply_links
from polyply.src.meta_molecule import (MetaMolecule, Monomer)
from vermouth.molecule import Interaction

class TestApplyLinks:
    @staticmethod
    @pytest.mark.parametrize('lines, monomers, interactions', (
        ("""[ moleculetype ]
        ; name nexcl.
        PEO         1
        ;
        [ atoms ]
        1  SN1a    1   PEO   CO1  1   0.000  45
        2  SN1a    1   PEO   CO2  1   0.000  45
        3  SN1a    1   PEO   CO3  1   0.000  45
        4  SN1a    1   PEO   CO4  1   0.000  45
        [ bonds ]
        ; back bone bonds
        1  2   1   0.37 7000
        2  3   1   0.37 7000
        2  4   1   0.37 7000
        4  5   1   0.37 7000
        """,
        [Monomer(resname="PEO",n_blocks=2)],
        {"bonds":[Interaction(atoms=(0, 1), parameters=['1', '0.37', '7000'], meta={}),
         Interaction(atoms=(1, 2), parameters=['1', '0.37', '7000'], meta={}),
         Interaction(atoms=(1, 3), parameters=['1', '0.37', '7000'], meta={}),
         Interaction(atoms=(3, 4), parameters=['1', '0.37', '7000'], meta={"version":1}),
         Interaction(atoms=(4, 5), parameters=['1', '0.37', '7000'], meta={}),
         Interaction(atoms=(5, 6), parameters=['1', '0.37', '7000'], meta={}),
         Interaction(atoms=(5, 7), parameters=['1', '0.37', '7000'], meta={})]}
         ),
         ("""
         [ moleculetype ]
         PS 1
         [ atoms ]
            1    STY            1  STYR       R1       1     0.00000E+00   45
            2    STY            1  STYR       R2       2     0.00000E+00   45
            3    STY            1  STYR       R3       3     0.00000E+00   45
            4    SCY            1  STYR       B        4     0.00000E+00   45
         [ bonds ]
            1     4   1     0.270000 8000
            4     5   1     0.270000 8000
         [ constraints ]
            2     3    1     0.270000
            3     1    1     0.270000
            1     2    1     0.270000
         [ angles ]
            4     1     2    1   136  100
            4     1     3    1   136  100
            4     5     6    1   136  100
            4     5     7    1   136  100
            1     4     5    1   120   25
            4     5     8    1    52  550
         [ exclusions ]
         1 2
         1 3
         1 4
         1 5
         1 6
         1 7
         1 8
         2 3
         2 4
         2 5
         3 5
         3 4
         4 5
         4 6
         4 7
         4 8
         4 9
         """,
         [Monomer(resname="PS",n_blocks=2)],
         {"bonds":[
         Interaction(atoms=(0, 3), parameters=['1', '0.270000', '8000'], meta={}),
         Interaction(atoms=(3, 4), parameters=['1', '0.270000', '8000'], meta={}),
         Interaction(atoms=(4, 7), parameters=['1', '0.270000', '8000'], meta={})],
         "angles":[
         Interaction(atoms=(0, 3, 4), parameters=['1', '120', '25'], meta={}),
         Interaction(atoms=(3, 0, 1), parameters=['1', '136', '100'], meta={}),
         Interaction(atoms=(3, 0, 2), parameters=['1', '136', '100'], meta={}),
         Interaction(atoms=(3, 4, 5), parameters=['1', '136', '100'], meta={}),
         Interaction(atoms=(3, 4, 6), parameters=['1', '136', '100'], meta={}),
         Interaction(atoms=(3, 4, 7), parameters=['1', '52', '550'], meta={}),
         Interaction(atoms=(7, 4, 5), parameters=['1', '136', '100'], meta={}),
         Interaction(atoms=(7, 4, 6), parameters=['1', '136', '100'], meta={})],
         "constraints":[
         Interaction(atoms=(0, 1), parameters=['1', '0.270000'], meta={}),
         Interaction(atoms=(1, 2), parameters=['1', '0.270000'], meta={}),
         Interaction(atoms=(2, 0), parameters=['1', '0.270000'], meta={}),
         Interaction(atoms=(4, 5), parameters=['1', '0.270000'], meta={}),
         Interaction(atoms=(5, 6), parameters=['1', '0.270000'], meta={}),
         Interaction(atoms=(6, 4), parameters=['1', '0.270000'], meta={})],
         "exclusions":[
         Interaction(atoms=(0, 1), parameters=[], meta={}),
         Interaction(atoms=(0, 2), parameters=[], meta={}),
         Interaction(atoms=(0, 3), parameters=[], meta={}),
         Interaction(atoms=(0, 4), parameters=[], meta={}),
         Interaction(atoms=(0, 5), parameters=[], meta={}),
         Interaction(atoms=(0, 6), parameters=[], meta={}),
         Interaction(atoms=(0, 7), parameters=[], meta={}),
         Interaction(atoms=(1, 2), parameters=[], meta={}),
         Interaction(atoms=(1, 3), parameters=[], meta={}),
         Interaction(atoms=(1, 4), parameters=[], meta={}),
         Interaction(atoms=(2, 3), parameters=[], meta={}),
         Interaction(atoms=(2, 4), parameters=[], meta={}),
         Interaction(atoms=(3, 4), parameters=[], meta={}),
         Interaction(atoms=(3, 5), parameters=[], meta={}),
         Interaction(atoms=(3, 6), parameters=[], meta={}),
         Interaction(atoms=(3, 7), parameters=[], meta={}),
         Interaction(atoms=(4, 5), parameters=[], meta={}),
         Interaction(atoms=(4, 6), parameters=[], meta={}),
         Interaction(atoms=(4, 7), parameters=[], meta={}),
         Interaction(atoms=(5, 6), parameters=[], meta={}),
         Interaction(atoms=(5, 7), parameters=[], meta={}),
         Interaction(atoms=(6, 7), parameters=[], meta={})]}),
         ("""[ moleculetype ]
           P3HT               1
          [ atoms ]
              1   SC5    1    P3HT     S1    1        0       45
              2   SC5    1    P3HT     C2    2        0       45
              3   SC5    1    P3HT     C3    3        0       45
              4    VS    1    P3HT     V4    4        0        0
              5   SC3    1    P3HT     C5    5        0       45
              6   SC3    1    P3HT     C6    6        0       45
          [ bonds ]
              5    6    1    0.360         5000
              4   10    1    0.380        50000
           [ exclusions ]
              1    7
              2    7
              3    7
          [ constraints ]
              1    2    1   0.240
              1    3    1   0.240
              2    3    1   0.240
              3    5    1   0.285
          [ angles ]
              1    3    5    2       180      250
              3    5    6    1       155       25
              2    3   10    1       160      180
              4    8    9    1       160      180
              4   10   16    1       158      180
          [ dihedrals ]
              1    4 7  10        9       0.0      1.8  1
              1    4 7  10        9       0.0     -9.5  2
           [ virtual_sitesn ]
             4    2    1   2   3""",
          [Monomer(resname="P3HT", n_blocks=2)],
           {"bonds":[
           Interaction(atoms=(3, 9), parameters=['1', '0.380', '50000'], meta={'version': 1}),
           Interaction(atoms=(4, 5), parameters=['1', '0.360', '5000'], meta={}),
           Interaction(atoms=(10, 11), parameters=['1', '0.360', '5000'], meta={})],
           "angles":[
           Interaction(atoms=(0, 2, 4), parameters=['2', '180', '250'], meta={}),
           Interaction(atoms=(1, 2, 6), parameters=['1', '160', '180'], meta={}),
           Interaction(atoms=(2, 4, 5), parameters=['1', '155', '25'], meta={}),
           Interaction(atoms=(3, 7, 8), parameters=['1', '160', '180'], meta={}),
           Interaction(atoms=(6, 8, 10), parameters=['2', '180', '250'], meta={}),
           Interaction(atoms=(8, 10, 11), parameters=['1', '155', '25'], meta={})],
           "constraints":[
           Interaction(atoms=(0, 1), parameters=['1', '0.240'], meta={}),
           Interaction(atoms=(0, 2), parameters=['1', '0.240'], meta={}),
           Interaction(atoms=(1, 2), parameters=['1', '0.240'], meta={}),
           Interaction(atoms=(2, 4), parameters=['1', '0.285'], meta={}),
           Interaction(atoms=(6, 7), parameters=['1', '0.240'], meta={}),
           Interaction(atoms=(6, 8), parameters=['1', '0.240'], meta={}),
           Interaction(atoms=(7, 8), parameters=['1', '0.240'], meta={}),
           Interaction(atoms=(8, 10), parameters=['1', '0.285'], meta={})],
           "exclusions":[
           Interaction(atoms=(0, 6), parameters=[], meta={'version':1}),
           Interaction(atoms=(1, 6), parameters=[], meta={'version':1}),
           Interaction(atoms=(2, 6), parameters=[], meta={'version':1})],
           "virtual_sitesn":[
           Interaction(atoms=(3, 0, 1, 2), parameters=['2'], meta={}),
           Interaction(atoms=(9, 6, 7, 8), parameters=['2'], meta={})],
           "dihedrals":[
           Interaction(atoms=(0, 3, 9, 6), parameters=['9','0.0','-9.5','2'], meta={}),
           Interaction(atoms=(0, 3, 9, 6), parameters=['9','0.0','1.8','1'], meta={'version':2})]})
         ))

    def test_polyply_input(lines, monomers, interactions):
        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        polyply.src.parsers.read_polyply(lines, ff)
        meta_mol = MetaMolecule.from_monomer_seq_linear(ff, monomers, "test")
        new_meta_mol = polyply.src.map_to_molecule.MapToMolecule().run_molecule(meta_mol)
        new_meta_mol = polyply.src.apply_links.ApplyLinks().run_molecule(meta_mol)

        for key in new_meta_mol.molecule.interactions:
            for interaction in new_meta_mol.molecule.interactions[key]:
                #print(interaction)
                assert interaction in interactions[key]
