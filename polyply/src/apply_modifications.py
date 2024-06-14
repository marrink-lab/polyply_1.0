# Copyright 2024 University of Groningen
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

from vermouth.log_helpers import StyleAdapter, get_logger
from vermouth.processors.annotate_mut_mod import parse_residue_spec
LOGGER = StyleAdapter(get_logger(__name__))
from .processor import Processor
import networkx as nx

def annotate_protein(meta_molecule):
    """
    make a resspec for a protein with correct N-ter and C-ter
    """

    resids = nx.get_node_attributes(meta_molecule.molecule, 'resid')
    last_resid = max([val for val in resids.values()])

    resnames = nx.get_node_attributes(meta_molecule.molecule, 'resname')
    last_resname = [val for val in resnames.values()][-1]

    protein_termini = [({'resid': 1, 'resname': resnames[0]}, 'N-ter'),
                       ({'resid': last_resid, 'resname': last_resname}, 'C-ter')
                       ]

    return protein_termini

def apply_terminal_mod(meta_molecule, modifications):
    """

    Apply a terminal modification.
    Note this assumes that the modification is additive, ie. no atoms are
    removed

    Parameters
    ----------
    meta_molecule: :class:`polyply.src.meta_molecule.MetaMolecule`
    modifications: list
        list of (resspec, modification) pairs to apply

    Returns
    ----------
    meta_molecule
    """

    if not modifications:
        return meta_molecule

    molecule = meta_molecule.molecule

    for target, desired_mod in modifications:

        target_resid = target['resid']

        mod_atoms = {}
        for mod_atom in molecule.force_field.modifications[desired_mod].atoms:
            if 'replace' in mod_atom:
                mod_atoms[mod_atom['atomname']] = mod_atom['replace']
            else:
                mod_atoms[mod_atom['atomname']] = {}


        anum_dict = {}
        for atomid, node in enumerate(meta_molecule.molecule.nodes):
            if meta_molecule.molecule.nodes[node]['resid'] == target_resid:
                # add the modification from the modifications dict
                aname = meta_molecule.molecule.nodes[node]['atomname']
                if aname in mod_atoms.keys():
                    anum_dict[aname] = atomid
                    for key, value in mod_atoms[aname].items():
                        meta_molecule.molecule.nodes[node][key] = value

        mod_interactions = molecule.force_field.modifications[desired_mod].interactions
        for i in mod_interactions:
            for j in molecule.force_field.modifications[desired_mod].interactions[i]:
                molecule.add_interaction(i,
                                         (anum_dict[j.atoms[0]],
                                          anum_dict[j.atoms[1]]),
                                         j.parameters,
                                         meta=j.meta)

    return meta_molecule

class ApplyModifications(Processor):
    def __init__(self, modifications=None, protter=False):
        if not modifications:
            modifications = []
        self.modifications = []
        for resspec, val in modifications:
            self.modifications.append((parse_residue_spec(resspec), val))
        self.protter = protter

    def run_molecule(self, meta_molecule):

        if self.protter:
            self.modifications = annotate_protein(meta_molecule)

        apply_terminal_mod(meta_molecule, self.modifications)

        return meta_molecule
