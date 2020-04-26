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
from polyply.src.linalg_functions import *

class TestLinAlg:

    @staticmethod
    def test_u_vect():
        v1 = np.array([0, 0, 0])
        v2 = np.array([0, 0, 2])
        u = u_vect(v1-v2)
        assert math.isclose(norm(u), 1.)

    @staticmethod
    def test_angle():
        v1 = np.array([0, 0, 0])
        v2 = np.array([0, 0, 2])
        v3 = np.array([0, 2, 2])
        assert math.isclose(angle(v1, v2, v3), 90)

    @staticmethod
    def test_dih():
        v1 = np.array([0, 0, 0])
        v2 = np.array([0, 0, 2])
        v3 = np.array([0, 2, 2])
        v4 = np.array([0,-2, 0])
        assert math.isclose(dih(v1,v2,v3,v4), 0)

    @staticmethod
    def test_geometrical_center():
       coords = np.array([[0, 0, 1],
                          [0, 1, 0],
                          [1, 0, 0],
                          [-1, 0, 0],
                          [0, -1, 0],
                          [0, 0, -1]])

       center = center_of_geometry(coords)
       assert math.isclose(norm(center),0)

    @staticmethod
    def test_radius_of_gyration():
        coords = np.array([[0, 0, 1],
                           [0, 1, 0],
                           [1, 0, 0],
                           [-1, 0, 0],
                           [0, -1, 0],
                           [0, 0, -1]])

        rg = radius_of_gyration(coords)
        assert math.isclose(rg, np.sqrt(2))
