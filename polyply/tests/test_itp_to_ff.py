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
Integration tests for the itp_to_ff utility program.
"""
from pathlib import Path
import numpy as np
import pytest
from vermouth.molecule import Molecule, Interaction
from vermouth.forcefield import ForceField
from vermouth.gmx.itp_read import read_itp
import polyply
from polyply import itp_to_ff, gen_params
from polyply.src.graph_utils import find_one_ismags_match
from .test_ffoutput import (_read_force_field, equal_ffs)
from .test_lib_files import _interaction_equal 

def _mass_match(node1, node2):
    return node1['mass'] == node2['mass']

def _read_itp(itppath):
    with open(itppath, "r") as _file:
        lines = _file.readlines()
    force_field = ForceField("tmp")
    read_itp(lines, force_field)
    block = next(iter(force_field.blocks.values()))
    mol = block.to_molecule()
    mol.make_edges_from_interaction_type(type_="bonds")
    return mol

def itp_equal(ref_mol, new_mol):
    """
    Leightweight itp comparison.
    """
    # new_node: ref_node
    match = find_one_ismags_match(new_mol, ref_mol, _mass_match)
    for node in new_mol.nodes:
        # check if important attributes are the same
        #assert new_mol.nodes[node]['atype'] == ref_mol.nodes[match[node]]['atype']
        # charge
        assert np.isclose(new_mol.nodes[node]['charge'],
                          ref_mol.nodes[match[node]]['charge'],
                          atol=0.1)

    for inter_type in new_mol.interactions:
        assert len(new_mol.interactions[inter_type]) == len(ref_mol.interactions[inter_type])
        for inter in new_mol.interactions[inter_type]:
            new_atoms = [match[atom] for atom in inter.atoms]
            new_inter = Interaction(atoms=new_atoms,
                                    parameters=inter.parameters,
                                    meta=inter.meta)
            for other_inter in ref_mol.interactions[inter_type]:
                if _interaction_equal(inter, other_inter, inter_type):
                    break
            else:
                assert False
    return True

@pytest.mark.parametrize("case, smiles, resnames, charges", [
    ("PEO_OHter", ["[OH][CH2]", "[CH2]O[CH2]", "[CH2][OH]"], ["OH", "PEO", "OH"], [0, 0, 0]),
    ("PEG_PBE", ["[CH3]", "[CH2][CH][CH][CH2]", "[CH2]O[CH2]"], ["CH3", "PBE", "PEO"], [0, 0, 0]),
])
def test_itp_to_ff(tmp_path, case, smiles, resnames, charges):
    """
    Call itp-to-ff and check if it generates the same force-field
    as in the ref.ff file.
    """
    tmp_path = Path("/Users/fabian/ProgramDev/polyply_1.0/polyply/tests/test_data/itp_to_ff/PEG_PBE/tmp")
    tmp_file = Path(tmp_path) / "test.ff"
    inpath = Path(polyply.TEST_DATA) / "itp_to_ff" / case
    itp_to_ff(itppath=inpath/"in_itp.itp",
              fragment_smiles=smiles,
              resnames=resnames,
              charges=charges,
              term_prefix='ter',
              outpath=tmp_file,)
    # now generate an itp file with this ff-file
    tmp_itp = tmp_path / "new.itp"
    gen_params(inpath=[tmp_file],
               seq_file=inpath/"seq.txt",
               outpath=tmp_itp, name="new")
    # read the itp-file and return a molecule
    new_mol = _read_itp(tmp_itp)
    ref_mol = _read_itp(inpath/"ref.itp")
    # check if itps are the same
    assert itp_equal(ref_mol, new_mol)
