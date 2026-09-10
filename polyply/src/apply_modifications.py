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
from vermouth.processors.sort_molecule_atoms import SortMoleculeAtoms
LOGGER = StyleAdapter(get_logger(__name__))
from .processor import Processor

protein_resnames = "GLY|ALA|CYS|VAL|LEU|ILE|MET|PRO|HYP|ASN|GLN|ASP|ASP0|GLU|GLU0|THR|SER|LYS|LYS0|ARG|ARG0|HIS|HISH|PHE|TYR|TRP"

#: attributes of a modification atom that describe the modification
#: itself and thus are not written to the molecule
MOD_ONLY_ATTRS = ('PTM_atom', 'replace', 'order')

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


def _add_ptm_atoms(molecule, residue, modification, mod_to_mol, resid, resname):
    """
    Add those atoms of `modification` that are only described by the
    modification (i.e. the PTM atoms) to `molecule` and to the graph of
    the `residue` they belong to. The correspondence of modification
    atoms to molecule atoms is updated in place.

    Parameters
    ----------
    molecule: :class:`vermouth.molecule.Molecule`
    residue: :class:`networkx.Graph`
        the graph of the residue the modification is applied to
    modification: :class:`vermouth.molecule.Modification`
    mod_to_mol: dict
        correspondence of the modification atoms to the molecule atoms;
        it must already contain the atoms described by the block
    resid: int
    resname: str

    Returns
    -------
    list
        the nodes that were added to the molecule
    """
    # the resid and resname are not part of the modification, because the
    # same modification can be applied to any residue; the same is true
    # for the charge group, which is taken from one of the atoms the
    # modification is attached to
    charge_group = None
    for mol_node in mod_to_mol.values():
        charge_group = molecule.nodes[mol_node].get('charge_group', None)
        if charge_group is not None:
            break

    new_nodes = []
    for mod_node, mod_attrs in modification.nodes(data=True):
        if not mod_attrs.get('PTM_atom', False):
            continue
        attrs = {attr: value for attr, value in mod_attrs.items()
                 if attr not in MOD_ONLY_ATTRS}
        attrs.update({'resid': resid, 'resname': resname})
        if charge_group is not None:
            attrs['charge_group'] = charge_group
        new_node = max(molecule.nodes) + 1
        molecule.add_node(new_node, **attrs)
        residue.add_node(new_node, **attrs)
        mod_to_mol[mod_node] = new_node
        new_nodes.append(new_node)
    return new_nodes

def apply_mod(meta_molecule, modifications):
    """
    Apply a modification to the residues of `meta_molecule`.

    A modification can replace attributes of atoms that are already
    described by the block of the residue, add the atoms that only the
    modification describes (i.e. the PTM atoms) together with the edges
    connecting them, and add interactions.

    Parameters
    ----------
    meta_molecule: :class:`polyply.src.meta_molecule.MetaMolecule`
    modifications: list
        list of (resspec, modification) pairs to apply

    Returns
    ----------
    meta_molecule

    Raises
    ------
    IOError
        if a modification is not defined in the force-field
    """

    molecule = meta_molecule.molecule

    if not molecule.force_field.modifications:
        LOGGER.info('No modifications present in forcefield, none will be applied')
        return meta_molecule

    added_atoms = []
    for target, desired_mod in modifications:
        LOGGER.info(f"Applying {desired_mod} to {target['resname']}{target['resid']}")

        if desired_mod not in molecule.force_field.modifications:
            msg = f"Modification {desired_mod} is not defined in the force-field."
            raise IOError(msg)
        modification = molecule.force_field.modifications[desired_mod]

        target_resid = target['resid']
        target_residue = meta_molecule.nodes[target_resid - 1]
        # takes care to skip all residues that come from an itp file
        if target_residue.get('from_itp'):
            LOGGER.info("meta_molecule has come from itp. Will not attempt to modify.")
            continue

        # the atoms of the modification that are already described by the
        # block are matched to the residue by atomname
        residue = target_residue['graph']
        res_atoms = {molecule.nodes[node]['atomname']: node for node in residue.nodes}
        mod_to_mol = {}
        missing = []
        for mod_node, mod_attrs in modification.nodes(data=True):
            if mod_attrs.get('PTM_atom', False):
                continue
            atomname = mod_attrs.get('atomname', mod_node)
            if atomname in res_atoms:
                mod_to_mol[mod_node] = res_atoms[atomname]
            else:
                missing.append(atomname)

        if missing:
            LOGGER.info(f"Cannot apply {desired_mod} to {target['resname']}{target_resid}, "
                        f"because the residue has no atoms {' '.join(missing)}.")
            continue

        # the modification can overwrite attributes of the atoms that are
        # described by the block
        for mod_node, mol_node in mod_to_mol.items():
            molecule.nodes[mol_node].update(modification.nodes[mod_node].get('replace', {}))

        added_atoms += _add_ptm_atoms(molecule,
                                      residue,
                                      modification,
                                      mod_to_mol,
                                      target_resid,
                                      target_residue['resname'])

        for mod_node, other_mod_node in modification.edges:
            mol_node = mod_to_mol[mod_node]
            other_mol_node = mod_to_mol[other_mod_node]
            molecule.add_edge(mol_node, other_mol_node)
            if mol_node in residue and other_mol_node in residue:
                residue.add_edge(mol_node, other_mol_node)

        for inter_type, interactions in modification.interactions.items():
            for interaction in interactions:
                molecule.add_or_replace_interaction(inter_type,
                                                    [mod_to_mol[atom] for atom in interaction.atoms],
                                                    interaction.parameters,
                                                    meta=interaction.meta)

    # the atoms of a modification are added at the end of the molecule,
    # so the molecule has to be sorted to keep the residues contiguous.
    # note that this reorders the complete molecule, including residues
    # that come from an itp file and are never modified themselves (for
    # example a protein a polymer is attached to). Those residues keep
    # their atoms in place as long as they come first in the sequence,
    # which other parts of polyply require anyway
    if added_atoms:
        SortMoleculeAtoms().run_molecule(molecule)

    return meta_molecule

def _annotated_modifications(meta_molecule):
    """
    Collect the modifications that links have annotated on the atoms of
    the molecule. A link can annotate a residue with a modification when
    it matches, for example at a terminus, by means of a replace
    statement setting the `annotated_modifications` attribute.

    Parameters
    ----------
    meta_molecule: :class:`polyply.src.meta_molecule.MetaMolecule`

    Returns
    -------
    list[tuple(dict, str)]
        the (resspec, modification) pairs found
    """
    targets = []
    for node, attrs in meta_molecule.molecule.nodes(data=True):
        for mod_name in attrs.get('annotated_modifications', []):
            target = ({'resid': attrs['resid'], 'resname': attrs['resname']}, mod_name)
            if target not in targets:
                targets.append(target)
    return targets

def modifications_finalising(meta_molecule, modifications):
    """
    clarify modifications in case we have multiple modifications targeting the same residue
    """
    # the terminal modifications are a default that the user has not asked
    # for, so they are only applied to protein residues and only if the
    # force-field defines them; modifications the user asks for explicitly
    # are applied to any residue
    known_mods = meta_molecule.force_field.modifications
    initial_target_mods = [(target, mod) for target, mod
                           in _patch_protein_termini(meta_molecule)
                           if target['resname'] in protein_resnames.split("|")
                           and mod in known_mods]
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

    # a link can annotate a residue with a modification, for example at a
    # terminus. Such an annotation is more specific than the default
    # terminal modifications, so it replaces them, but it is less
    # specific than what the user asks for explicitly
    for target, mod_name in _annotated_modifications(meta_molecule):
        specified = [spec for spec in final_target_mods
                     if spec[0]['resid'] == target['resid']]
        if any(spec in additional_modifications for spec in specified):
            continue
        for spec in specified:
            final_target_mods.remove(spec)
        final_target_mods.append((target, mod_name))

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
