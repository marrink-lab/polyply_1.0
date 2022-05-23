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

import textwrap
import pytest
import numpy as np
import math
import networkx as nx
import polyply
from polyply.src.linalg_functions import u_vect
from polyply.src.gen_moving_frame import gen_next_normal

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
def test_gen_template_frame(template, base, force_field):
    result = polyply.src.gen_moving_frame.calc_tangents(curve_coords)

    # Check if shapes are correct
    assert [bundle.shape == curve_coords.shape for bundle in results]

    # Check if all reference vectors are normalised
    norms = [np.linalg.norm(bundle, axis=1) for bundle in results]
    assert np.allclose(norms, 1)

    # Determine if frames are orthogonal
    inproduct = [np.sum(bundles[0] * bundles[1]) for bundles
                in combinations(results, 2)]
    assert np.allclose(inproduct, 0)
