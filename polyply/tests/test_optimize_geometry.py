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
Test geometry optimizer
"""
import textwrap
import math
import pytest
import numpy as np
import vermouth
import polyply
from polyply.src.generate_templates import _expand_inital_coords
from polyply.src.linalg_functions import angle
from polyply.src.minimizer import optimize_geometry
from polyply.src.virtual_site_builder import construct_vs

@pytest.mark.parametrize('lines', (
     """
     [ moleculetype ]
     test 1
     [ atoms ]
     1 P4 1 GLY BB 1
     2 P3 1 GLY SC1 2
     [ bonds ]
     1  2   1   0.2   100
     """,
     """
     [ moleculetype ]
     test 1
     [ atoms ]
     1 P4 1 GLY BB 1
     2 P3 1 GLY SC1 2
     3 P3 1 GLY SC2 2
     [ bonds ]
     1  2   1   0.2   100
     2  3   1   0.2   100
     [ angles ]
     1  2  3  1   90  50
     """,
     """
     [ moleculetype ]
     test               1
     [ atoms ]
     1   SC5    1    P3HT     S1    1        0       45
     2   SC5    1    P3HT     C2    2        0       45
     3   SC5    1    P3HT     C3    3        0       45
     4    VS    1    P3HT     V4    4        0        0
     5   SC3    1    P3HT     C5    5        0       45
     6   SC3    1    P3HT     C6    6        0       45
     [ bonds ]
     5    6    1    0.360         5000
     [ constraints ]
     1    2    1   0.240
     1    3    1   0.240
     2    3    1   0.240
     3    5    1   0.285
     [ angles ]
     1    3    5    2       180      250
     3    5    6    1       155       25
     [ virtual_sitesn ]
     4    2    1   2   3
     """))
def test_optimize_geometry(lines):
    """
    Tests if the geometry optimizer performs correct optimization
    of simple geometries. This guards against changes in scipy
    optimize that might effect optimization.
    """
    lines = textwrap.dedent(lines).splitlines()
    force_field = vermouth.forcefield.ForceField(name='test_ff')
    polyply.src.polyply_parser.read_polyply(lines, force_field)
    block = force_field.blocks['test']
    init_coords = _expand_inital_coords(block)
    status, coords = optimize_geometry(block, init_coords)

    assert status

    for bond in block.interactions["bonds"]:
        ref = float(bond.parameters[1])
        dist = np.linalg.norm(coords[bond.atoms[0]] - coords[bond.atoms[1]])
        assert math.isclose(dist, ref, abs_tol=0.1)

    for bond in block.interactions["constraints"]:
        ref = float(bond.parameters[1])
        dist = np.linalg.norm(coords[bond.atoms[0]] - coords[bond.atoms[1]])
        assert math.isclose(dist, ref, abs_tol=0.05)

    for inter in block.interactions["angles"]:
        ref = float(inter.parameters[1])
        ang = angle(coords[inter.atoms[0]], coords[inter.atoms[1]], coords[inter.atoms[2]])
        assert math.isclose(ang, ref, abs_tol=20)

    for virtual_side in block.interactions["virutal_sitesn"]:
        ref_coord = construct_vs("virutal_sitesn", virtual_side, coords)
        vs_coords = coords[virtual_side.atoms[0]]
        assert all(np.isclose(ref_coord, vs_coords))
