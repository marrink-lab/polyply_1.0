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
Test that inequivalent residues are identified.
"""
import textwrap
import pytest
from vermouth.forcefield import ForceField
import polyply
from polyply.src.topology import Topology
from polyply.src.top_parser import read_topology

@pytest.mark.parametrize('top_lines', (
    # residue in two different molecules not equivalent by
    # number of atoms
    ("""
    [ defaults ]
    1   1   no   1.0     1.0
    [ atomtypes ]
    N0 72.0 0.000 A 0.0 0.0
    N1 72.0 0.000 A 0.0 0.0
    [ nonbond_params ]
    N0   N0   1 4.700000e-01    3.700000e+00
    N1   N1   1 4.700000e-01    3.700000e+00
    [ moleculetype ]
    testA 1
    [ atoms ]
    1    N0   1   GLY    BB   1 0.00     45
    2    N0   1   GLY    SC1  1 0.00     45
    3    N0   1   GLY    SC2  1 0.00     45
    4    N1   2   GLU    BB   2 0.00     45
    5    N1   2   GLU    SC1  2 0.00     45
    6    N1   2   GLU    SC2  2 0.00     45
    [ bonds ]
    1     2    1  0.47 2000
    2     3    1  0.47 2000
    3     4    1  0.47 2000
    4     5    1  0.47 2000
    5     6    1  0.47 2000
    [ moleculetype ]
    testB 1
    [ atoms ]
    1    N0   1   ASP    BB   1 0.00     45
    2    N0   1   ASP    SC3  1 0.00     45
    3    N0   1   ASP    SC4  1 0.00     45
    4    N1   2   GLU    BB   2 0.00     45
    5    N1   2   GLU    SC1  2 0.00     45
    [ bonds ]
    1     2    1  0.47 2000
    2     3    1  0.47 2000
    3     4    1  0.47 2000
    4     5    1  0.47 2000
    [ system ]
    test system
    [ molecules ]
    testA 1
    testB 1
    """),
    # residue in two different molecules not equivalent by
    # connectivity
    ("""
    [ defaults ]
    1   1   no   1.0     1.0
    [ atomtypes ]
    N0 72.0 0.000 A 0.0 0.0
    N1 72.0 0.000 A 0.0 0.0
    [ nonbond_params ]
    N0   N0   1 4.700000e-01    3.700000e+00
    N1   N1   1 4.700000e-01    3.700000e+00
    [ moleculetype ]
    testA 1
    [ atoms ]
    1    N0   1   GLY    BB   1 0.00     45
    2    N0   1   GLY    SC1  1 0.00     45
    3    N0   1   GLY    SC2  1 0.00     45
    4    N1   2   GLU    BB   2 0.00     45
    5    N1   2   GLU    SC1  2 0.00     45
    6    N1   2   GLU    SC2  2 0.00     45
    7    N1   2   GLU    SC3  2 0.00     45
    [ bonds ]
    1     2    1  0.47 2000
    2     3    1  0.47 2000
    3     4    1  0.47 2000
    4     5    1  0.47 2000
    5     6    1  0.47 2000
    6     7    1  0.47 2000
    [ moleculetype ]
    testB 1
    [ atoms ]
    1    N0   1   ASP    BB   1 0.00     45
    2    N0   1   ASP    SC3  1 0.00     45
    3    N0   1   ASP    SC4  1 0.00     45
    4    N1   2   GLU    BB   2 0.00     45
    5    N1   2   GLU    SC1  2 0.00     45
    6    N1   2   GLU    SC2  2 0.00     45
    7    N1   2   GLU    SC3  2 0.00     45
    [ bonds ]
    1     2    1  0.47 2000
    2     3    1  0.47 2000
    3     4    1  0.47 2000
    4     5    1  0.47 2000
    5     6    1  0.47 2000
    5     7    1  0.47 2000
    [ system ]
    test system
    [ molecules ]
    testA 1
    testB 1
    """)))
def test_raise_residue_error(top_lines):
    lines = textwrap.dedent(top_lines)
    lines = lines.splitlines()
    force_field = ForceField("test")
    topology = Topology(force_field)
    read_topology(lines=lines, topology=topology, cwdir="./")
    topology.preprocess()
    topology.volumes = {"GLY": 0.53, "GLU": 0.67, "ASP": 0.43}
    with pytest.raises(IOError):
        polyply.src.check_residue_equivalence.check_residue_equivalence(topology)

def test_raise_residue_no_error():
    """
    This test makes sure that we actually skip molecules that are defined
    in the top file but not used in the actual system.
    """
    top_lines="""
    [ defaults ]
    1   1   no   1.0     1.0
    [ atomtypes ]
    N0 72.0 0.000 A 0.0 0.0
    N1 72.0 0.000 A 0.0 0.0
    [ nonbond_params ]
    N0   N0   1 4.700000e-01    3.700000e+00
    N1   N1   1 4.700000e-01    3.700000e+00
    [ moleculetype ]
    testA 1
    [ atoms ]
    1    N0   1   GLY    BB   1 0.00     45
    2    N0   1   GLY    SC1  1 0.00     45
    3    N0   1   GLY    SC2  1 0.00     45
    4    N1   2   GLU    BB   2 0.00     45
    5    N1   2   GLU    SC1  2 0.00     45
    6    N1   2   GLU    SC2  2 0.00     45
    7    N1   2   GLU    SC3  2 0.00     45
    [ bonds ]
    1     2    1  0.47 2000
    2     3    1  0.47 2000
    3     4    1  0.47 2000
    4     5    1  0.47 2000
    5     6    1  0.47 2000
    6     7    1  0.47 2000
    [ moleculetype ]
    testB 1
    [ atoms ]
    1    N0   1   ASP    BB   1 0.00     45
    2    N0   1   ASP    SC3  1 0.00     45
    3    N0   1   ASP    SC4  1 0.00     45
    4    N1   2   GLU    BB   2 0.00     45
    5    N1   2   GLU    SC1  2 0.00     45
    6    N1   2   GLU    SC2  2 0.00     45
    7    N1   2   GLU    SC3  2 0.00     45
    [ bonds ]
    1     2    1  0.47 2000
    2     3    1  0.47 2000
    3     4    1  0.47 2000
    4     5    1  0.47 2000
    5     6    1  0.47 2000
    5     7    1  0.47 2000
    [ system ]
    test system
    [ molecules ]
    testA 1
    """
    lines = textwrap.dedent(top_lines)
    lines = lines.splitlines()
    force_field = ForceField("test")
    topology = Topology(force_field)
    read_topology(lines=lines, topology=topology, cwdir="./")
    topology.preprocess()
    topology.volumes = {"GLY": 0.53, "GLU": 0.67, "ASP": 0.43}
    polyply.src.check_residue_equivalence.check_residue_equivalence(topology)
