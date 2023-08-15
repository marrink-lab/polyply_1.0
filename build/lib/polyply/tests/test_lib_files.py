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
Runs more elaborate integration tests
"""

from collections import defaultdict
from pathlib import Path
import shlex
import subprocess
import sys
import numpy as np

import pytest
import vermouth
from vermouth.forcefield import ForceField

from polyply import TEST_DATA
from vermouth.tests.helper_functions import find_in_path

INTEGRATION_DATA = Path(TEST_DATA + '/library_tests')

PATTERN = '{path}/{library}/{polymer}/polyply'

POLYPLY = find_in_path(names=("polyply", ))

def assert_equal_blocks(block1, block2):
    """
    Asserts that two blocks are equal to gain the pytest rich comparisons,
    which is lost when doing `assert block1 == block2`
    """
    assert block1.name == block2.name
    assert block1.nrexcl == block2.nrexcl
    assert block1.force_field == block2.force_field  # Set to be equal

    for node in block1.nodes:
        # for the simulation only these two attributes matter
        # as we have 3rd party reference files we don't do more
        # checks
        for attr in ["atype", "charge"]:
            # if the reference itp has the attribute check it
            if attr in block1.nodes[node]:
                assert block1.nodes[node][attr] == block2.nodes[node][attr]

    edges1 = {frozenset(e[:2]): e[2] for e in block1.edges(data=True)}
    edges2 = {frozenset(e[:2]): e[2] for e in block2.edges(data=True)}

    for e, attrs in edges2.items():
        for k, v in attrs.items():
            if isinstance(v, float):
                attrs[k] = pytest.approx(v, abs=1e-3) # PDB precision is 1e-3

    assert edges1 == edges2

    for inter_type in ["bonds", "angles", "constraints", "exclusions", "pairs", "impropers", "dihedrals"]:
        ref_interactions = block1.interactions.get(inter_type, [])
        new_interactions = block2.interactions.get(inter_type, [])
        print(inter_type)
        assert len(ref_interactions) == len(new_interactions)

        ref_terms = defaultdict(list)
        for inter in ref_interactions:
            atoms = inter.atoms
            ref_terms[frozenset(atoms)].append(inter)

        new_terms = defaultdict(list)
        for inter_new in new_interactions:
            atoms = inter_new.atoms
            new_terms[frozenset(atoms)].append(inter_new)

        for atoms, ref_interactions in ref_terms.items():
            new_interactions = new_terms[atoms]
            for ref_inter in ref_interactions:
                print(ref_inter)
                for new_inter in new_interactions:
                    if _interaction_equal(ref_inter, new_inter, inter_type):
                        break
                else:
                    assert False

def compare_itp(filename1, filename2):
    """
    Asserts that two itps are functionally identical
    """
    dummy_ff = ForceField(name='dummy')
    with open(filename1) as fn1:
        vermouth.gmx.read_itp(fn1, dummy_ff)
    dummy_ff2 = ForceField(name='dummy')
    with open(filename2) as fn2:
        vermouth.gmx.read_itp(fn2, dummy_ff2)
    for block in dummy_ff2.blocks.values():
        block._force_field = dummy_ff
    assert set(dummy_ff.blocks.keys()) == set(dummy_ff2.blocks.keys())
    for name in dummy_ff.blocks:
        block1 = dummy_ff.blocks[name]
        block2 = dummy_ff2.blocks[name]
        assert_equal_blocks(block1, block2)


def compare_pdb(filename1, filename2):
    """
    Asserts that two pdbs are functionally identical
    """
    pdb1 = vermouth.pdb.read_pdb(filename1)
    pdb2 = vermouth.pdb.read_pdb(filename2)
    assert len(pdb1) == len(pdb2)
    for mol1, mol2 in zip(pdb1, pdb2):
        for mol in (mol1, mol2):
            for n_idx in mol:
                node = mol.nodes[n_idx]
                if 'position' in node and node['atomname'] in ('SCN', 'SCP'):
                    # Charge dummies get placed randomly, which complicated
                    # comparisons to no end.
                    # These will be caught by the distances in the edges instead.
                    del node['position']

        assert_equal_blocks(mol1, mol2)


COMPARERS = {'.itp': compare_itp,
             '.pdb': compare_pdb}


def _interaction_equal(interaction1, interaction2, inter_type):
    """
    Returns True if interaction1 == interaction2, ignoring rounding errors in
    interaction parameters.
    """
    p1 = list(map(str, interaction1.parameters))
    p2 = list(map(str, interaction2.parameters))
    a1 = list(interaction1.atoms)
    a2 = list(interaction2.atoms)


    if p1 != p2:
        return False

    if interaction1.meta != interaction2.meta:
        return False

    if inter_type in ["constraints", "bonds", "exclusions", "pairs"]:
        a1.sort()
        a2.sort()
        return a1 == a2

    elif inter_type in ["impropers", "dihedrals"]:
        if a1 == a2:
            return True
        a1.reverse()
        if a1 == a2:
            return True
        else:
            print(a1, a2)

    elif inter_type in ["angles"]:
        return a1[1] == a2[1] and frozenset([a1[0], a1[2]]) == frozenset([a2[0], a2[2]])

    return False

@pytest.mark.parametrize("library, polymer", [
     ['2016H66', 'PP'],
     ['2016H66', 'C12E4'],
     ['2016H66', 'PE'],
     ['2016H66', 'PVA'],
     ['2016H66', 'PMA'],
     ['2016H66', 'PS'],
     ['gromos53A6', 'P3HT'],
     ['oplsaaLigParGen', 'PEO'],
     ['martini3', 'PROT'],
     ['martini3', 'PEO'],
     ['martini3', 'PS'],
     ['martini3', 'PE'],
     ['martini3', 'DEX'],
     ['martini3', 'CEL'],
     ['martini3', 'P3HT'],
     ['martini3', 'PPE'],
     ['martini3', 'PTMA'],
     ['martini2', 'PEO'],
     ['martini2', 'PS'],
     ['martini2', 'PEL'],
     ['martini2', 'PEO_PE'],
     ['martini2', 'ssDNA'],
     ['parmbsc1', 'DNA'],
  # -> proteins?
])
def test_integration_protein(tmp_path, monkeypatch, library, polymer):
    """
    Runs tests on the library by executing the contents of the file `command` in
    the folder libname/polymer, and tests whether the contents of the produced
    files are the same as the reference files. The comparison of the files is
    governed by `COMPARERS`.

    Parameters
    ----------
    tmp_path
    library: str
    polymer: str
    """
    monkeypatch.chdir(tmp_path)
    data_path = Path(PATTERN.format(path=INTEGRATION_DATA, library=library, polymer=polymer))

    with open(str(data_path / 'command')) as cmd_file:
        command = cmd_file.read().strip()

    assert command  # Defensive
    command = shlex.split(command)
    result = [sys.executable]
    for token in command:
        if token.startswith('polyply'):
            result.append(str(POLYPLY))
        elif token.startswith('.'):
            result.append(str(data_path / token))
        else:
            result.append(token)

    command = result

    # read the citations that are expected
    # citations = []
    # with open(str(data_path/'citation')) as cite_file:
    #     for line in cite_file:
    #         citations.append(line.strip())

    proc = subprocess.run(command, cwd='.', timeout=60, check=False,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          universal_newlines=True)
    exit_code = proc.returncode
    if exit_code:
        #print(proc.stdout)
        #print(proc.stderr)
        assert not exit_code

    # check if strdout has citations in string
    #for citation in citations:
    #    assert citation in proc.stderr

    files = list(tmp_path.iterdir())

    assert files
    assert list(tmp_path.glob('*.itp')), files

    for new_file in tmp_path.iterdir():
        filename = new_file.name
        reference_file = data_path/filename
        assert reference_file.is_file()
        ext = new_file.suffix.lower()
        if ext in COMPARERS:
            COMPARERS[ext](str(reference_file), str(new_file))
