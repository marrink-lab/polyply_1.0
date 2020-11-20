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
import math
import pytest
import numpy as np
from numpy.linalg import norm
import vermouth
import polyply
from polyply import TEST_DATA
from polyply.src.meta_molecule import Monomer, MetaMolecule
from polyply.src.topology import Topology
from polyply.src.nonbond_matrix import NonBondMatrix
from polyply.src.build_system import (_compute_box_size,
                                      _filter_by_molname
                                     )


@pytest.fixture
def topology():
    top_lines ="""
    [ defaults ]
    1   1   no   1.0     1.0
    [ atomtypes ]
    N0 72.0 0.000 A 0.0 0.0
    [ nonbond_params ]
    N0   N0   1 4.700000e-01    3.700000e+00
    [ moleculetype ]
    testA 1
    [ atoms ]
    1    N0  1   GLY    BB   1 0.00     45
    2    N0  1   GLY    SC1  1 0.00     45
    3    N0  1   GLY    SC2  1 0.00     45
    4    N0  2   GLU    BB   2 0.00     45
    5    N0  2   GLU    SC1  2 0.00     45
    6    N0  2   GLU    SC2  2 0.00     45
    [ bonds ]
    1    2    1  0.47 2000
    2    3    1  0.47 2000
    1    4    1  0.47 2000
    4    5    1  0.47 2000
    4    6    1  0.47 2000
    [ moleculetype ]
    testB 1
    [ atoms ]
    1    N0  1   ASP    BB   1 0.00     45
    2    N0  1   ASP    SC1  1 0.00     45
    3    N0  1   ASP    SC2  1 0.00     45
    [ bonds ]
    1    2    1  0.47 2000
    2    3    1  0.47 2000
    [ system ]
    test system
    [ molecules ]
    testA 2
    testB 1
    """
    lines = textwrap.dedent(top_lines)
    lines = lines.splitlines()
    force_field = ForceField("test")
    topology = Topology(force_field)
    read_topology(lines=lines, topology=topology, cwdir="./")
    topology.preprocess()
    topology.volumes = {"GLY": 0.53, "GLU": 0.67, "ASP": 0.43}
    return topology

@pytest.mark.parametrize('density, result', (
    (1000.0,
     4.2119903964305125
     ),
    (600.0,
     4.993866813213379
     )))
def test_compute_box_size(density, result):
    top = create_topology(100)
    assert np.isclose(_compute_box_size(top, density), result)



@pytest.mark.parametrize('ignore', (
    ["NA"],
    ["AA", "NA"],
     ))
def test_filer_by_molname(test_system, ignore):
    molecules = _filter_by_molname(test_system.molecules, ignore)
    for molecule in molecules:
        assert molecule.mol_name not in ignore


@pytest.mark.parametrize('ignore', (
      density,
      start_dict,
      max_force=10**3,
      grid_spacing=0.2,
      maxiter=800,
      maxiter_random=50,
      box=[],
      step_fudge=1,
      push=[],
      ignore=[],
      grid=None):

def test_run_molecule(test_system, defaults):
   # iterate over
   # density vs box
   # grid_spacind vs grid
   # start and ignore
   BuildSystem(test_system,
               start_dict=start_dict,
               density=args.density,
               max_force=args.max_force,
               grid_spacing=args.grid_spacing,
               maxiter=args.maxiter,
               maxiter_random=args.maxiter_random,
               box=box,
               step_fudge=args.step_fudge,
               push=args.push,
               ignore=args.ignore,
               grid=grid).run_system(topology.molecules)
