# Copyright 2022 University of Groningen
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
Test the fragment finder for itp_to_ff.
"""
import pytest
from pathlib import Path
import networkx as nx
from vermouth.molecule import Interaction
from polyply.src.molecule_utils import extract_links
from .test_apply_links import example_meta_molecule

@pytest.mark.parametrize('inters, expected',(
    # simple bond spanning two residues
    ({'bonds':[Interaction(atoms=(0, 1), parameters=['1', '0.33', '500'], meta={}),
               Interaction(atoms=(1, 2), parameters=['1', '0.33', '500'], meta={}),
               Interaction(atoms=(1, 4), parameters=['1', '0.30', '500'], meta={}),
               Interaction(atoms=(4, 5), parameters=['1', '0.35', '500'], meta={}),]},
     {'bonds': [Interaction(atoms=['BB1', '+BB'],
                            parameters=['1', '0.30', '500'],
                            meta={'version': 0, 'comment': 'link'}),
               ]},
    ),
    # double version dihedral spanning two residues
    ({'dihedrals':[Interaction(atoms=(0, 1, 4, 5),
                               parameters=['9', '120', '4', '1'],
                               meta={}),
                   Interaction(atoms=(0, 1, 4, 5),
                               parameters=['9', '120', '4', '2'],
                               meta={}),
                   Interaction(atoms=(0, 1, 2, 3),
                               parameters=['9', '120', '4', '2'],
                               meta={})]
     },
     {'dihedrals': [Interaction(atoms=['BB', 'BB1', '+BB', '+BB1'],
                                parameters=['9', '120', '4', '1'],
                                meta={'version': 0, 'comment': 'link'}),
                    Interaction(atoms=['BB', 'BB1', '+BB', '+BB1'],
                                parameters=['9', '120', '4', '2'],
                                meta={'version': 1, 'comment': 'link'}),]
     },
    ),
    # 1-5 pairs spanning 3 residues
    ({'pairs': [Interaction(atoms=(1, 9),
                            parameters=[1],
                            meta={})]},
    {'pairs': [Interaction(atoms=['BB1', '++BB'],
                           parameters=[1],
                           meta={'version': 0, 'comment': 'link'})]
    }),
))
def test_extract_links(example_meta_molecule, inters, expected):
    mol = example_meta_molecule.molecule
    mol.add_edges_from([(1, 4), (8, 9)])
    nx.set_node_attributes(mol, {0: "resA", 1: "resA", 2: "resA", 3: "resA",
                                 4: "resB", 5: "resB", 6: "resB", 7: "resB", 8: "resB",
                                 9: "resA", 10: "resA", 11: "resA", 12: "resA"}, "resname")
    nx.set_node_attributes(mol, {0: "BB", 1: "BB1", 2: "SC1", 3: "SC2",
                                 4: "BB", 5: "BB1", 6: "BB2", 7: "SC1", 8: "SC2",
                                 9: "BB", 10: "BB1", 11: "SC1", 12: "SC2"}, "atomname")
    mol.interactions.update(inters)
    link = extract_links(mol)[0]
    for inter_type in expected:
        assert expected[inter_type] == link.interactions[inter_type]


