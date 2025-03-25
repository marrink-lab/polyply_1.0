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
import vermouth.molecule
from vermouth.log_helpers import StyleAdapter, get_logger
from vermouth.processors.annotate_mut_mod import parse_residue_spec
LOGGER = StyleAdapter(get_logger(__name__))
from .processor import Processor

protein_resnames = "GLY|ALA|CYS|VAL|LEU|ILE|MET|PRO|HYP|ASN|GLN|ASP|ASP0|GLU|GLU0|THR|SER|LYS|LYS0|ARG|ARG0|HIS|HISH|PHE|TYR|TRP"

def _patch_protein_termini(meta_molecule, ter_mods=['N-ter', 'C-ter']):
    """
    make a resspec for a protein with correct terminal modification
    """
    protein_termini = [({'resid': 1, 'resname': meta_molecule.nodes[0]['resname']}, ter_mods[0])]
    max_resid = meta_molecule.max_resid
    last_node = max_resid - 1
    last_resname = meta_molecule.nodes[last_node]['resname']
    if len(ter_mods) > 1:
        last_mod = ({'resid': max_resid, 'resname': last_resname}, ter_mods[1])
        protein_termini.append(last_mod)
    else:
        # if only one mod in ter_mods, apply the mod to both start and end residue
        LOGGER.warning("Only one terminal modification specified. "
                    f"Will apply {ter_mods[0]} to both {meta_molecule.nodes[0]['resname']}1 and {last_resname}{max_resid}")
        protein_termini.append(({'resid': max_resid, 'resname': last_resname}, ter_mods[0]))
    return protein_termini


def apply_mod(meta_molecule, modifications):
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

    molecule = meta_molecule.molecule

    if not molecule.force_field.modifications:
        LOGGER.info('No modifications present in forcefield, none will be applied')
        return meta_molecule

    for target, desired_mod in modifications:
        LOGGER.info(f"Applying {desired_mod} to {target['resname']}{target['resid']}")

        target_resid = target['resid']

        mod_atoms = {}
        for mod_atom in molecule.force_field.modifications[desired_mod].atoms:
            if 'replace' in mod_atom:
                mod_atoms[mod_atom['atomname']] = mod_atom['replace']
            else:
                mod_atoms[mod_atom['atomname']] = {}

        target_residue = meta_molecule.nodes[target_resid - 1]
        # takes care to skip all residues that come from an itp file
        if target_residue.get('from_itp'):
            LOGGER.info("meta_molecule has come from itp. Will not attempt to modify.")
            continue
        # checks that the resname is a protein resname as defined above
        if not vermouth.molecule.attributes_match(target_residue,
                                              {'resname': vermouth.molecule.Choice(protein_resnames.split("|"))}):
            LOGGER.info("The resname of your target residue is not recognised as a protein resname. "
                           "Will not attempt to modify.")
            continue

        anum_dict = {}
        # this gives you the correct node indices
        for node in target_residue['graph'].nodes:
            aname = molecule.nodes[node]['atomname']
            if aname in mod_atoms.keys():
                anum_dict[aname] = node + 1
                for key, value in mod_atoms[aname].items():
                    molecule.nodes[node][key] = value

        mod_interactions = molecule.force_field.modifications[desired_mod].interactions
        for i in mod_interactions:
            for j in molecule.force_field.modifications[desired_mod].interactions[i]:
                molecule.add_interaction(i,
                                         [anum_dict[k]-1 for k in j.atoms],
                                         j.parameters,
                                         meta=j.meta)

    return meta_molecule

def modifications_finalising(meta_molecule, modifications):
    """
    clarify modifications in case we have multiple modifications targeting the same residue
    """
    initial_target_mods = _patch_protein_termini(meta_molecule)
    # parse all additional modifications
    additional_modifications = []
    for resspec, val in modifications:
        additional_modifications.append((parse_residue_spec(resspec), val))

    # check if any of the additional modifications are in the target modifications
    default_targets = [spec[0] for spec in initial_target_mods]
    additional_targets = [spec[0] for spec in additional_modifications]
    to_remove = []
    for target in additional_targets:
        if target in default_targets:
            to_remove.append(default_targets[default_targets == target])
    # make sure the list is unique in case, eg. modify a terminal residue twice over
    unique_removals = list({v['resid']: v for v in to_remove}.values())
    # now filter the default modifications by the overwritten ones
    final_target_mods = []
    for spec in initial_target_mods:
        if spec[0] not in unique_removals:
            final_target_mods.append(spec)

    final_target_mods.extend(additional_modifications)
    # check whether the final modifications are the same as the original ones
    if modifications and (not set(modifications[0]) == set([i[1] for i in final_target_mods])):
        LOGGER.info("Default modifications overwritten. Check log for modifications applied.")

    return final_target_mods

class ApplyModifications(Processor):
    """
    This processor takes a class:`polyply.src.MetaMolecule` and
    based on modifications defined in the `force-field` attribute of the
    MetaMolecule applies them when appropriate.

    """
    def __init__(self, meta_molecule, modifications=[]):
        self.target_mods = modifications_finalising(meta_molecule, modifications)

    def run_molecule(self, meta_molecule):
        apply_mod(meta_molecule, self.target_mods)
        return meta_molecule
