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
Test constructions of virutal-sites. Reference values come from
a GROMACS energy minimization.
"""

import numpy as np
import pytest
from vermouth.molecule import Interaction
from ..src.virtual_site_builder import construct_vs


@pytest.mark.parametrize('vs_type, interaction, positions, result', (
    # vsn
    ("virtual_sitesn",
     Interaction(atoms=[6, 1, 2, 3, 4, 5], parameters=["1"], meta={}),
     {1: np.array([4.924, 0.353, 0.036]), 2: np.array([5.191, 0.023, 0.208]),
      3: np.array([5.190, -0.231, -0.162]), 4: np.array([4.734, -0.268, -0.131]),
      5: np.array([4.548, 0.088, 0.046])},
     np.array([4.917, -0.007, -0.001])
     ),
    #  vs2
    ("virtual_sites2",
     Interaction(atoms=[3, 1, 2], parameters=["1", "0.5"], meta={}),
     {1: np.array([0.000, 0.000, 0.000]), 2: np.array([-0.012, 0.002, 0.024])},
     np.array([-0.006, 0.001, 0.012])
     ),
    #  vs31
    ("virtual_sites3",
     Interaction(atoms=[4, 1, 2, 3], parameters=[
                 "1", "0.223", "0.865"], meta={}),
     {1: np.array([4.717, 0.333, 4.974]), 2: np.array([4.983, -0.028, 5.098]),
      3: np.array([5.307, -0.324, 4.951])},
     np.array([5.287, -0.316, 4.982])
     ),
    #  vs32
    ("virtual_sites3",
        Interaction(atoms=[4, 1, 2, 3], parameters=[
                    "2", "0.223", "0.865"], meta={}),
        {1: np.array([4.717, 0.333, 4.974]), 2: np.array([4.983, -0.028, 5.098]),
         3: np.array([5.307, -0.324, 4.951])},
        np.array([5.247, -0.336, 5.117])
     ),
    #  vs33
    ("virtual_sites3",
        Interaction(atoms=[4, 1, 2, 3], parameters=[
                    "3", "150", "0.22"], meta={}),
        {1: np.array([4.717, 0.333, 4.974]), 2: np.array([4.983, -0.028, 5.098]),
         3: np.array([5.307, -0.324, 4.951])},
        np.array([4.652, 0.479, 4.823])
     ),
    #  vs34
    ("virtual_sites3",
     Interaction(atoms=[4, 1, 2, 3], parameters=[
                 "4", "0.2", "0.3", "1.5"], meta={}),
     {1: np.array([4.717, 0.333, 4.974]), 2: np.array([4.983, -0.028, 5.098]),
      3: np.array([5.307, -0.324, 4.951])},
     np.array([5.081, 0.182, 5.049])
     ),
    # vs4
    ("virtual_sites4",
        Interaction(atoms=[5, 1, 2, 3, 4], parameters=[
                    "2", "0.22", "0.45", "0.6"], meta={}),
        {1: np.array([4.917, 0.223, 0.192]), 2: np.array([5.064, 0.143, -0.214]),
         3: np.array([5.185, -0.244, 0.044]), 4: np.array([4.730, -0.149, -0.012])},
        np.array([5.156, -0.327, 0.215])
     )
))
def test(vs_type, interaction, positions, result):
    """
    Test if the six virtual sites are constructed properly.
    """
    vs_coord = construct_vs(vs_type, interaction, positions)
    assert all(np.isclose(vs_coord, result, atol=10**-3.))


def test_IOError():
    """
    Tests if VS1 with function type 1 raises error.
    """
    with pytest.raises(IOError):
        construct_vs("virtual_sites4", Interaction(
            atoms=[], parameters=["1"], meta={}), {})
