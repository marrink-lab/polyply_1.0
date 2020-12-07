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
from polyply.src.linalg_functions import (_u_vect, _angle,
                                         _dih, _radius_of_gyration,
                                         center_of_geometry, _vector_angle_degrees)

class TestLinAlg:

    @staticmethod
    def test_vector_angle_degrees():
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([0.0, 0.0, 2.0])
        v3 = np.array([0.0, 2.0, 2.0])
        assert math.isclose(_vector_angle_degrees(v1-v2, v2-v3), 90.0)

    @staticmethod
    def test_u_vect():
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([0.0, 0.0, 2])
        u = _u_vect(v1-v2)
        assert math.isclose(np.linalg.norm(u), 1.)

    @staticmethod
    def test_angle():
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([0.0, 0.0, 2.0])
        v3 = np.array([0.0, 2.0, 2.0])
        assert math.isclose(_angle(v1, v2, v3), 90.0)

    @staticmethod
    def test_dih():
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([0.0, 0.0, 2.0])
        v3 = np.array([0.0, 2.0, 2.0])
        v4 = np.array([0.0,-2.0, 0.0])
        assert math.isclose(_dih(v1,v2,v3,v4), 0.0)

    @staticmethod
    def test_geometrical_center():
       coords = np.array([[0.0, 0.0, 1.0],
                          [0.0, 1.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [-1.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0],
                          [0.0, 0.0, -1.0]])

       center = center_of_geometry(coords)
       assert math.isclose(np.linalg.norm(center), 0.0)

    @staticmethod
    def test_radius_of_gyration():
        coords = np.array([[0.0, 0.0, 1.0],
                           [0.0, 1.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0],
                           [0.0, 0.0, -1.0]])

        rg = _radius_of_gyration(coords)
        assert math.isclose(rg, 1.0)
