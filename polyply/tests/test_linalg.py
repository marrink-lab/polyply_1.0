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
import polyply
from polyply.src.linalg_functions import (_u_vect, _angle,
                                         _dih, _radius_of_gyration,
                                         center_of_geometry, _vector_angle_degrees)


def test_vector_angle_degrees():
    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([0.0, 0.0, 2.0])
    v3 = np.array([0.0, 2.0, 2.0])
    assert math.isclose(_vector_angle_degrees(v1-v2, v2-v3), 90.0)

def test_u_vect():
    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([0.0, 0.0, 2])
    u = _u_vect(v1-v2)
    assert math.isclose(np.linalg.norm(u), 1.)

def test_angle():
    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([0.0, 0.0, 2.0])
    v3 = np.array([0.0, 2.0, 2.0])
    assert math.isclose(_angle(v1, v2, v3), 90.0)

def test_dih():
    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([0.0, 0.0, 2.0])
    v3 = np.array([0.0, 2.0, 2.0])
    v4 = np.array([0.0,-2.0, 0.0])
    assert math.isclose(_dih(v1, v2, v3, v4), 0.0)

def test_geometrical_center():
   coords = np.array([[0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [-1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0],
                      [0.0, 0.0, -1.0]])

   center = center_of_geometry(coords)
   assert math.isclose(np.linalg.norm(center), 0.0)

def test_radius_of_gyration():
    coords = np.array([[0.0, 0.0, 1.0],
                       [0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0, 0.0, -1.0]])

    rg = _radius_of_gyration(coords)
    assert math.isclose(rg, 1.0)

@pytest.mark.parametrize('vectors, expected',(
                        # single column vector with floats
                        ([np.array([[2.5], [2.5], [2.5]])],
                         np.array([[15.0], [37.5], [60.0]])),
                        # simple 2x3 matrix with floats
                        ([np.array([[2.5, 3.5], [2.5, 3.5] , [2.5, 3.5]])],
                          np.array([[15.0, 21.0], [37.5, 52.5], [60.0 , 84.0]])),
                        # multiple column vectors
                        ([np.array([[2.5], [2.5], [2.5]]),
                          np.array([[3.5], [3.5], [3.5]])],
                         np.array([[52.5], [131.25], [210.0]])),
                        # multiple matrices
                        ([np.array([[2.5, 4.5], [2.5, 4.5], [2.5, 4.5]]),
                          np.array([[3.5, 5.5], [3.5, 5.5], [3.5, 5.5]])],
                         np.array([[147.0,  231.0], [367.5, 577.5], [588.0, 924.0]])),
                        ))
def test_matrix_multiplication(vectors, expected):
        matrix = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        result = polyply.src.linalg_functions._matrix_multiplication(matrix, *vectors)
        assert result.shape == expected.shape
        assert np.allclose(expected, result)

@pytest.mark.parametrize('vectors, angles, expected',(
                        # single vector rotation around x
                        (np.array([[2.0], [3.5], [4.5]]),
                         np.array([90, 0.0, 0.0]),
                         np.array([[2.0], [-4.5], [3.5]])),
                        # single vector rotation around y
                        (np.array([[2.5], [3.5], [4.5]]),
                         np.array([0.0, 90.0, 0.0]),
                         np.array([[4.5], [3.5], [-2.5]])),
                        # single vector rotation around z
                        (np.array([[2.5], [3.5], [4.5]]),
                         np.array([0.0, 0.0, 90.0]),
                         np.array([[-3.5], [2.5], [4.5]])),
                        # single vector rotation around y, z
                        (np.array([[2.5], [3.5], [4.5]]),
                         np.array([0.0, 90.0, 90.0]),
                         np.array([[-3.5], [4.5], [-2.5]])),
                        # multiple points rotation around x, y
                        (np.array([[2.5, 5.5], [3.5, 6.5], [4.5, 7.5]]),
                         np.array([90.0, 90.0, 0.0]),
                         np.array([[3.5, 6.5], [-4.5, -7.5], [-2.5, -5.5]])),
                        ))
def test_rotation_matrix(vectors, angles, expected):
        result = polyply.src.linalg_functions._rotate_xyz(vectors, *np.deg2rad(angles))
        assert result.shape == expected.shape
        assert np.allclose(expected, result)

@pytest.mark.parametrize('object_xyz, vect, expected',(
                        # rotate Id. matrix 2pi around z-axis
                        (np.array([[1., 0., 0.],
                                   [0., 1., 0.],
                                   [0., 0., 1.]]),
                         np.array([0., 0., 2 * np.pi]),
                         np.array([[1., 0., 0.],
                                   [0., 1., 0.],
                                   [0., 0., 1.]])),
                        # rotate unit x-vector pi radian around z-axis
                        (np.array([[1., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]]),
                         np.array([0., 0., np.pi]),
                         np.array([[-1., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]])),
                        # rotate rotate random matrix around vector
                        (np.array([[1., 2., 3.],
                                   [4., 5., 6.],
                                   [7., 8., 9.]]),
                         np.array([np.pi, np.pi, np.pi]),
                         np.array([[0.70978341, 1.70978341, 2.70978341],
                                   [6.58364764, 7.58364764, 8.58364764],
                                   [4.70656895, 5.70656895, 6.70656895]])),
                        ))
def test_rotate_from_vect(object_xyz, vect, expected):
        result = polyply.src.linalg_functions._rotate_from_vect(object_xyz, vect)
        assert result.shape == expected.shape
        assert np.allclose(expected, result)

        inv_results = polyply.src.linalg_functions._rotate_from_vect(result, -vect)
        assert np.allclose(inv_results, object_xyz)


@pytest.mark.parametrize('coords, expected',(
                        # Points on straight line
                        (np.array([[0., 0., 0.],
                                   [1., 1., 1.],
                                   [2., 2., 2.],
                                   [3., 3., 3.]]),
                         np.array([[1. ,1., 1.],
                                   [1., 1., 1.],
                                   [1., 1., 1.],
                                   [1., 1., 1.]])),
                        # Points on circle
                        (np.array([[ 1.0000000e+00,  0.0000000e+00, 0],
                                   [ 6.1232340e-17,  1.0000000e+00, 0],
                                   [-1.0000000e+00,  1.2246468e-16, 0],
                                   [-1.8369702e-16, -1.0000000e+00, 0]]),
                         np.array([[-1. ,1., 0.],
                                   [-1., 0., 0],
                                   [ 0., -1, 0],
                                   [1., -1., 0.]])),
                        ))
def test_finite_difference_O1(coords, expected):
        result = polyply.src.linalg_functions._finite_difference_O1(coords)
        assert result.shape == expected.shape
        assert np.allclose(expected, result)

@pytest.mark.parametrize('coords, expected',(
                        # Points on straight line
                        (np.array([[0., 0., 0.],
                                   [1., 1., 1.],
                                   [2., 2., 2.],
                                   [3., 3., 3.],
                                   [4., 4., 4.],
                                   [5., 5., 5.]]),
                         np.array([[1. ,1., 1.],
                                   [1., 1., 1.],
                                   [1., 1., 1.],
                                   [1., 1., 1.],
                                   [1., 1., 1.],
                                   [1., 1., 1.]])),
                        # Points on circle
                        (np.array([[ 1.00000000e+00,  0.00000000e+00],
                                   [ 5.00000000e-01,  8.66025404e-01],
                                   [-5.00000000e-01,  8.66025404e-01],
                                   [-1.00000000e+00,  1.22464680e-16],
                                   [-5.00000000e-01, -8.66025404e-01],
                                   [ 5.00000000e-01, -8.66025404e-01]]),
                         np.array([[ 0.20833333,  1.08253175],
                                   [-0.95833333,  0.50518149],
                                   [-0.875     , -0.50518149],
                                   [ 0.        , -1.01036297],
                                   [ 0.91666667, -0.57735027],
                                   [ 0.83333333,  0.72168784]])),
                        ))
def test_finite_difference_O5(coords, expected):
        result = polyply.src.linalg_functions._finite_difference_O5(coords)
        assert result.shape == expected.shape
        assert np.allclose(expected, result)
