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
import itertools
import pytest
import numpy as np
import networkx as nx
import vermouth
import polyply
from polyply.src.generate_templates import _expand_inital_coords
from polyply.src.linalg_functions import angle, dih
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
     1  2   1   0.49   100
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
     [ constraints ]
     1    2    1   0.240
     1    3    1   0.240
     2    3    1   0.240
     [ virtual_sitesn ]
     4    2    1   2   3
     """,
    # CH3-CH2-CH2 fragment from CHARMM36
     """
     [ moleculetype ]
     ; name  nrexcl
     test        3
     [ atoms ]
     ; nr    type    resnr   residu  atom    cgnr    charge  mass
     1  CTL3            1 MOA      C1          1    -0.270000    12.0110 ; qtot -0.270
     2  HAL3            1 MOA      H1A         2     0.090000     1.0080 ; qtot -0.180
     3  HAL3            1 MOA      H1B         3     0.090000     1.0080 ; qtot -0.090
     4  HAL3            1 MOA      H1C         4     0.090000     1.0080 ; qtot -0.000
     5  CTL2            1 MOA      C2          5    -0.180000    12.0110 ; qtot -0.180
     6  HAL2            1 MOA      H2A         6     0.090000     1.0080 ; qtot -0.090
     7  HAL2            1 MOA      H2B         7     0.090000     1.0080 ; qtot -0.000
     8  CTL2            1 MOA      C3          8    -0.180000    12.0110 ; qtot -0.180
     9  HAL2            1 MOA      H3A         9     0.090000     1.0080 ; qtot -0.090
     10 HAL2            1 MOA      H3B        10     0.090000     1.0080 ; qtot -0.000
     [ bonds ]
     ; ai    aj      funct   b0      Kb
     1     2     1   0.11110000    269449.60
     1     3     1   0.11110000    269449.60
     1     4     1   0.11110000    269449.60
     1     5     1   0.15280000    186188.00
     5     6     1   0.11110000    258571.20
     5     7     1   0.11110000    258571.20
     5     8     1   0.15300000    186188.00
     8     9     1   0.11110000    258571.20
     8    10     1   0.11110000    258571.20
     [ pairs ]
     ; ai    aj      funct   c6      c12
     2     6     1
     2     7     1
     2     8     1
     3     6     1
     3     7     1
     3     8     1
     4     6     1
     4     7     1
     4     8     1
     1     9     1
     1    10     1
     6     9     1
     6    10     1
     7     9     1
     7    10     1
     [ angles ]
     ; ai    aj      ak      funct   th0     cth     S0      Kub
     2     1     3     5    108.400000   297.064000   0.18020000      4518.72
     2     1     4     5    108.400000   297.064000   0.18020000      4518.72
     2     1     5     5    110.100000   289.532800   0.21790000     18853.10
     3     1     4     5    108.400000   297.064000   0.18020000      4518.72
     3     1     5     5    110.100000   289.532800   0.21790000     18853.10
     4     1     5     5    110.100000   289.532800   0.21790000     18853.10
     1     5     6     5    110.100000   289.532800   0.21790000     18853.10 ; HAL2 CTL2 CTL3
     1     5     7     5    110.100000   289.532800   0.21790000     18853.10 ; HAL2 CTL2 CTL3
     1     5     8     5    115.000000   485.344000   0.25610000      6694.40 ; CTL3 CTL2 CTL2
     6     5     7     5    109.000000   297.064000   0.18020000      4518.72 ; HAL2 CTL2 HAL2
     6     5     8     5    110.100000   221.752000   0.21790000     18853.10 ; HAL2 CTL2 CTL2
     7     5     8     5    110.100000   221.752000   0.21790000     18853.10 ; HAL2 CTL2 CTL2
     5     8     9     5    110.100000   221.752000   0.21790000     18853.10 ; HAL2 CTL2 CTL2
     5     8    10     5    110.100000   221.752000   0.21790000     18853.10 ; HAL2 CTL2 CTL2
     9     8    10     5    109.000000   297.064000   0.18020000      4518.72 ; HAL2 CTL2 HAL2
     [ dihedrals ]
     ; ai    aj      ak      al      funct   phi0    cp      mult
     2     1     5     6     9       0.000000     0.669440     3
     2     1     5     7     9       0.000000     0.669440     3
     2     1     5     8     9       0.000000     0.669440     3
     3     1     5     6     9       0.000000     0.669440     3
     3     1     5     7     9       0.000000     0.669440     3
     3     1     5     8     9       0.000000     0.669440     3
     4     1     5     6     9       0.000000     0.669440     3
     4     1     5     7     9       0.000000     0.669440     3
     4     1     5     8     9       0.000000     0.669440     3
     1     5     8     9     9       0.000000     0.794960     3
     1     5     8    10     9       0.000000     0.794960     3
     6     5     8     9     9       0.000000     0.794960     3
     6     5     8    10     9       0.000000     0.794960     3
     7     5     8     9     9       0.000000     0.794960     3
     7     5     8    10     9       0.000000     0.794960     3
     """,
    # test 1 PS monomer -> improper dihedral + flat ring system + 1 proper dih
    """
    [ moleculetype ]
    test 3

    [ atoms ]
     1 CH3 1 PS VC1  1     0.0 14.0
     2 CH2 1 PS VC2  2     0.1 13.0
     3 C   1 PS SC1  3    -0.1 12.0
     4 C   1 PS SC2  4 -0.1298 12.0
     5 HC  1 PS H1   5  0.1298  1.0
     6 C   1 PS C3   6 -0.1298 12.0
     7 HC  1 PS H2   7  0.1298  1.0
     8 C   1 PS C4   8 -0.1298 12.0
     9 HC  1 PS H3   9  0.1298  1.0
    10 C   1 PS C5  10 -0.1298 12.0
    11 HC  1 PS H4  11  0.1298  1.0
    12 C   1 PS SC3 12 -0.1298 12.0
    13 HC  1 PS H5  13  0.1298  1.0
    [ bonds ]
     1  2 2 1.5300000e-01  7.1500000e+06 ; C,CHn-CHn,C
     2  3 2 1.5300000e-01  7.1500000e+06 ; C,CHn-CHn,C
     5  4 2 1.0900000e-01  1.2300000e+07 ; C-HC
     3  4 2 1.3900000e-01  1.0800000e+07 ; C-C arom.
     7  6 2 1.0900000e-01  1.2300000e+07 ; C-HC
     4  6 2 1.3900000e-01  1.0800000e+07 ; C-C arom.
     9  8 2 1.0900000e-01  1.2300000e+07 ; C-HC
     6  8 2 1.3900000e-01  1.0800000e+07 ; C-C arom.
    11 10 2 1.0900000e-01  1.2300000e+07 ; C-HC
     8 10 2 1.3900000e-01  1.0800000e+07 ; C-C arom.
    13 12 2 1.0900000e-01  1.2300000e+07 ; C-HC
     3 12 2 1.3900000e-01  1.0800000e+07 ; C-C arom.
    10 12 2 1.3900000e-01  1.0800000e+07 ; C-C arom.
    [ angles ]
     3  2  1 2 1.1100000e+02  5.3000000e+02 ; CHn-CHn-C,CHn,OA,OE,NR,NT,NL
    12  3  2 2 1.2000000e+02  5.6000000e+02 ; 6-ring
     4  3  2 2 1.2000000e+02  5.6000000e+02 ; 6-ring
    13 12  3 2 1.1100000e+02  5.3000000e+02 ; HC-C-C
    10 12  3 2 1.2000000e+02  5.6000000e+02 ; 6-ring
     5  4  3 2 1.2000000e+02  5.0500000e+02 ; HC-C-C
     6  4  3 2 1.2000000e+02  5.6000000e+02 ; 6-ring
    12  3  4 2 1.2000000e+02  5.6000000e+02 ; 6-ring
     7  6  4 2 1.2000000e+02  5.0500000e+02 ; HC-C-C
     8  6  4 2 1.2000000e+02  5.6000000e+02 ; 6-ring
     6  4  5 2 1.2000000e+02  5.0500000e+02 ; HC-C-C
     9  8  6 2 1.2000000e+02  5.0500000e+02 ; HC-C-C
    10  8  6 2 1.2000000e+02  5.6000000e+02 ; 6-ring
     8  6  7 2 1.2000000e+02  5.0500000e+02 ; HC-C-C
    11 10  8 2 1.2000000e+02  5.0500000e+02 ; HC-C-C
    12 10  8 2 1.2000000e+02  5.6000000e+02 ; 6-ring
    10  8  9 2 1.2000000e+02  5.0500000e+02 ; HC-C-C
    13 12 10 2 1.2000000e+02  5.0500000e+02 ; HC-C-C
    12 10 11 2 1.2000000e+02  5.0500000e+02 ; HC-C-C

    [ dihedrals ]
     1  2  3 12 1 0.0000000e+00  1.0000000e+00  6 ; X-CHn-C,CR1-X

    [ dihedrals ]
     4  3 12 10 2  0.0000000e+00  1.6742312e+02 ; arom. planar center
     6  4  3 12 2  0.0000000e+00  1.6742312e+02 ; arom. planar center
     3 12 10  8 2  0.0000000e+00  1.6742312e+02 ; arom. planar center
     3  4  6  8 2  0.0000000e+00  1.6742312e+02 ; arom. planar center
     4  6  8 10 2  0.0000000e+00  1.6742312e+02 ; arom. planar center
     6  8 10 12 2  0.0000000e+00  1.6742312e+02 ; arom. planar center
     3  2  4 12 2  0.0000000e+00  1.6742312e+02 ; planar center
     4  5  6  3 2  0.0000000e+00  1.6742312e+02 ; planar center
     6  7  8  4 2  0.0000000e+00  1.6742312e+02 ; planar center
     8 10  6  9 2  0.0000000e+00  1.6742312e+02 ; planar center
    10 12  8 11 2  0.0000000e+00  1.6742312e+02 ; planar center
    12  3 13 10 2  0.0000000e+00  1.6742312e+02 ; planar center
    """
))
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
    for _iteration in range(0, 10):
        init_coords = _expand_inital_coords(block)
        success, coords = optimize_geometry(block, init_coords, ["bonds", "constraints", "angles"])
        success, coords = optimize_geometry(block, coords, ["bonds", "constraints", "angles", "dihedrals"])
        if success:
            break
    else:
        print(_iteration)

    # sanity check
    assert success

    # this part checks that the tolarances are actually obayed in the minimizer
    for bond in itertools.chain(block.interactions["bonds"], block.interactions["constraints"]):
        ref = float(bond.parameters[1])
        dist = np.linalg.norm(coords[bond.atoms[0]] - coords[bond.atoms[1]])
        assert np.isclose(dist, ref, atol=0.05)

    for inter in block.interactions["angles"]:
        ref = float(inter.parameters[1])
        ang = angle(coords[inter.atoms[0]], coords[inter.atoms[1]], coords[inter.atoms[2]])
        assert np.isclose(ang, ref, atol=5)

    # only improper dihedrals
    for inter in block.interactions["dihedrals"]:
        if inter.parameters[0] == "2":
            ref = float(inter.parameters[1])
            ang = dih(coords[inter.atoms[0]],
                      coords[inter.atoms[1]],
                      coords[inter.atoms[2]],
                      coords[inter.atoms[3]])
            assert np.isclose(ang, ref, atol=5)

    for virtual_site in block.interactions["virtual_sitesn"]:
        ref_coord = construct_vs("virtual_sitesn", virtual_site, coords)
        vs_coords = coords[virtual_site.atoms[0]]
        assert np.allclose(ref_coord, vs_coords)
