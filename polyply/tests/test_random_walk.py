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
import numpy as np
from numpy.linalg import norm
import networkx as nx
import vermouth
import polyply
from polyply import MetaMolecule
from ..src.random_walk import (RandomWalk, _take_step, _is_overlap, update_positions)

class TestRandomWalk:

    @staticmethod
    def test__take_step():
        coord = np.array([0, 0, 0])
        step_length = 0.5
        vectors = polyply.src.linalg_functions.norm_sphere(50)
        new_coord, idx = _take_step(vectors, step_length, coord)
        assert math.isclose(norm(new_coord - coord), step_length)

    @staticmethod
    @pytest.mark.parametrize('new_point, tol, fudge, reference', (
    (np.array([0, 0, 1.0]),
     0.4,
     1.0,
     False),
    (np.array([0, 0, 0.6]),
     0.5,
     1.0,
     True),
    (np.array([0, 0, 0.75]),
     0.5,
     0.4,
     False)
    ))
    def test__is_overlap(new_point, tol, fudge, reference):
      ff = vermouth.forcefield.ForceField(name='test_ff')
      meta_mol = MetaMolecule(name="test", force_field=ff)
      meta_mol.add_monomer(0,"PEO",[])
      meta_mol.add_monomer(1,"PEO",[(1,0)])
      meta_mol.nodes[0]["position"] = np.array([0, 0, 0])
      meta_mol.nodes[1]["position"] = np.array([0, 0, 0.5])
      result =  _is_overlap(meta_mol, new_point, tol, fudge)
      assert result == reference

    @staticmethod
    def test_update_positions():
      ff = vermouth.forcefield.ForceField(name='test_ff')
      meta_mol = MetaMolecule(name="test", force_field=ff)
      meta_mol.add_monomer(0,"PEO",[])
      meta_mol.add_monomer(1,"PEO",[(1,0)])
      meta_mol.add_monomer(2,"PEO",[(1,2)])
      meta_mol.nodes[0]["position"] = np.array([0, 0, 0])
      meta_mol.nodes[1]["position"] = np.array([0, 0, 0.5])
      meta_mol.volumes = {}
      meta_mol.volumes["PEO"] = 0.5
      vectors = polyply.src.linalg_functions.norm_sphere(50)
      update_positions(vectors, meta_mol, 2, 1)
      assert "position" in meta_mol.nodes[2]

    @staticmethod
    def test_run_molecule():
       ff = vermouth.forcefield.ForceField(name='test_ff')
       meta_mol = MetaMolecule(name="test", force_field=ff)
       meta_mol.add_monomer(0,"PEO",[])
       meta_mol.add_monomer(1,"PEO",[(1,0)])
       meta_mol.add_monomer(2,"PEO",[(1,2)])
       meta_mol.volumes = {}
       meta_mol.volumes["PEO"] = 0.5
       RandomWalk().run_molecule(meta_mol)
       for node in meta_mol.nodes:
           assert "position" in meta_mol.nodes[node]
