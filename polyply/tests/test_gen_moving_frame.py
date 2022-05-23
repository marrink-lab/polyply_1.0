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
Test moving frame generator.
"""

from itertools import combinations
import textwrap
import pytest
import numpy as np
import math
import networkx as nx
import polyply
from polyply.src.linalg_functions import u_vect
from polyply.src.gen_moving_frame import gen_next_normal

@pytest.mark.parametrize('curve_coords, expected',(
                        # points on straight line, O1/2
                        (np.array([[0., 0., 0.],
                                   [1., 1., 1.],
                                   [2., 2., 2.]]),
                         np.array([[1. ,1., 1.],
                                   [1., 1., 1.],
                                   [1., 1., 1.]])),
                        # points on straight line, O5
                        (np.array([[0., 0., 0.],
                                   [1., 1., 1.],
                                   [2., 2., 2.],
                                   [3., 3., 3.],
                                   [4., 4., 4.]]),
                         np.array([[1. ,1., 1.],
                                   [1., 1., 1.],
                                   [1., 1., 1.],
                                   [1., 1., 1.],
                                   [1., 1., 1.]])),
                        ))
def test_calc_tangents(curve_coords, expected):
    result = polyply.src.gen_moving_frame.calc_tangents(curve_coords)
    assert result.shape == expected.shape
    assert np.allclose(expected, result)

@pytest.mark.parametrize('current_frame, next_frame, expected',(
                        # frames on straight line
                        (np.array([[0., 0., 0.],
                                   [0., 0., 1.],
                                   [0., 1., 0.]]),
                         np.array([[1. ,1., 1.],
                                   [0., 0., 1.]]),
                         np.array([0., 1., 0.])),
                        # frame on circle
                        (np.array([[1., 0., 0.],
                                   [0., 1., 0.],
                                   [0., 0., 1.]]),
                         np.array([[0. ,1., 0.],
                                   [-1., 0., 0.]]),
                         np.array([0., 0., 1.])),
                        # frame after "sharp" turn
                        (np.array([[0., 0., 0.],
                                   [1., 0., 0.],
                                   [0., 0., 1.]]),
                         np.array([[1. ,0., 0.],
                                   [1/np.sqrt(2), 1/np.sqrt(2), 0.]]),
                         np.array([0., 0., 1.])),
                        ))
def test_gen_next_normal(current_frame, next_frame, expected):
    result = polyply.src.gen_moving_frame.gen_next_normal(current_frame, next_frame)
    assert result.shape == expected.shape
    assert np.allclose(expected, result)

@pytest.mark.parametrize('curve_coords',(
                        # straight curve on x-axis
                        (np.array([[0., 0., 0.],
                                   [1., 0., 0.],
                                   [2., 0., 0.],
                                   [3., 0., 0.],
                                   [4., 0., 0.],
                                   [5., 0., 0.]])),
                        # straight curve on y-axis
                        (np.array([[0., 0., 0.],
                                   [0., 1., 0.],
                                   [0., 2., 0.],
                                   [0., 3., 0.],
                                   [0., 4., 0.],
                                   [0., 5., 0.]])),
                        # straight curve on z-axis
                        (np.array([[0., 0., 0.],
                                   [0., 0., 1.],
                                   [0., 0., 2.],
                                   [0., 0., 3.],
                                   [0., 0., 4.],
                                   [0., 0., 5.]])),
                        # random curve
                        (np.array([[0., 0., 0.],
                                   [1.3, 4.0, 1.],
                                   [2.2, 3.9, 2.],
                                   [3.1, 3.8, 3.],
                                   [4.0, 3.2, 4.],
                                   [5.9, 3.3, 5.]])),
                        ))
def test_rotation_min_frame(curve_coords):
    results = polyply.src.gen_moving_frame.rotation_min_frame(curve_coords)
    # Check if shapes are correct
    assert [bundle.shape == curve_coords.shape for bundle in results]

    # Check if all reference vectors are normalised
    norms = [np.linalg.norm(bundle, axis=1) for bundle in results]
    assert np.allclose(norms, 1)

    # Determine if frames are orthogonal
    inproduct = [np.sum(bundles[0] * bundles[1]) for bundles
                in combinations(results, 2)]
    assert np.allclose(inproduct, 0)

@pytest.mark.parametrize('curve_coords',(
                        # straight curve on x-axis
                        (np.array([[0., 0., 0.],
                                   [0.33, 0., 0.],
                                   [0.66, 0., 0.],
                                   [0.99, 0., 0.],
                                   [1.32, 0., 0.],
                                   [1.65, 0., 0.]])),
                        # Points on Circle
                        (np.array([[ 8.09e-01,  5.87e-01, 0.00e+00],
                                   [ 3.09e-01,  9.51e-01, 0.00e+00],
                                   [-3.09e-01,  9.51e-01, 0.00e+00],
                                   [-8.09e-01,  5.87e-01, 0.00e+00],
                                   [-1.00e+00,  1.22e-16, 0.00e+00],
                                   [-8.09e-01, -5.87e-01, 0.00e+00],
                                   [-3.09e-01, -9.51e-01, 0.00e+00],
                                   [ 3.09e-01, -9.51e-01, 0.00e+00],
                                   [ 8.09e-01, -5.87e-01, 0.00e+00],
                                   [ 1.00e+00, -2.44e-16, 0.00e+00]])),
                        ))
def test_close_frame(curve_coords):
    rotation_per_bp = 0.59
    input_frames = polyply.src.gen_moving_frame.rotation_min_frame(curve_coords)
    tangents, normals, binormals = input_frames

    results = polyply.src.gen_moving_frame.close_frame(curve_coords,
                                                       *input_frames,
                                                       rotation_per_bp)
    tangents, normals, binormals = results
    # Determine the minimal rotating normal vector wrt. last reference frame
    target_normal = gen_next_normal((curve_coords[-1], tangents[-1], normals[-1]),
                                    (curve_coords[0], tangents[0]))
    # Calculate rotation needed to match the normals
    phi = np.arccos(target_normal @ normals[0]) - rotation_per_bp

    assert np.isclose(phi, 0)

@pytest.mark.parametrize('meta_mol, curve_coords',(
                        # straight curve on x-axis
                        (nx.path_graph(6),
                         np.array([[0., 0., 0.],
                                   [0.33, 0., 0.],
                                   [0.66, 0., 0.],
                                   [0.99, 0., 0.],
                                   [1.32, 0., 0.],
                                   [1.65, 0., 0.]])),
                        # Points on Circle
                        (nx.cycle_graph(10),
                         np.array([[ 8.09e-01,  5.87e-01, 0.00e+00],
                                   [ 3.09e-01,  9.51e-01, 0.00e+00],
                                   [-3.09e-01,  9.51e-01, 0.00e+00],
                                   [-8.09e-01,  5.87e-01, 0.00e+00],
                                   [-1.00e+00,  1.22e-16, 0.00e+00],
                                   [-8.09e-01, -5.87e-01, 0.00e+00],
                                   [-3.09e-01, -9.51e-01, 0.00e+00],
                                   [ 3.09e-01, -9.51e-01, 0.00e+00],
                                   [ 8.09e-01, -5.87e-01, 0.00e+00],
                                   [ 1.00e+00, -2.44e-16, 0.00e+00]])),
                        ))

def test_dna_frame(meta_mol, curve_coords):
    rotation_per_bp = 0.59
    results = polyply.src.gen_moving_frame.dna_frame(meta_mol,
                                                     curve_coords,
                                                     rotation_per_bp)
    tangents, normals, _ = results
    start = 1 if nx.cycle_basis(meta_mol) else 1
    strand_rotation = []
    for i in range(start, len(curve_coords)):
        # Determine the minimal rotating normal vector wrt. last reference frame
        target_normal = gen_next_normal((curve_coords[i-1], tangents[i-1], normals[i-1]),
                                        (curve_coords[i], tangents[i]))
        # Calculate rotation needed to match the normals
        phi = np.arccos(target_normal @ normals[i]) - rotation_per_bp
        strand_rotation.append(phi)

    if nx.cycle_basis(meta_mol):
        # Check if overall under/overtwisting is uniform
        assert np.allclose(strand_rotation, phi)
    else:
        # Check that there is no under/overtwisting
        assert np.allclose(strand_rotation, 0)
