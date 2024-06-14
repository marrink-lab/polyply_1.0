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
LOGGER = StyleAdapter(get_logger(__name__))

def apply_terminal_mod(meta_molecule, term, res):
    """

    Apply a terminal modification.
    Note this assumes that the modification is additive, ie. no atoms are
    removed

    Parameters
    ----------
    meta_molecule: :class:`polyply.src.meta_molecule.MetaMolecule`
    term: str
        name modification in forcefield to apply
    res: int
        resid of where to apply the modification

    Returns
    ----------
    meta_molecule
    """

    molecule = meta_molecule.molecule

    mod_atoms = {}
    for i in molecule.force_field.modifications[term].atoms:
        if 'replace' in i:
            mod_atoms[i['atomname']] = i['replace']
        else:
            mod_atoms[i['atomname']] = {}


    anum_dict = {}
    for atomid, node in enumerate(meta_molecule.molecule.nodes):
        if meta_molecule.molecule.nodes[node]['resid'] == res:
            # Ensure unique resnames for unique blocks, necessary for coordinate building
            # This does mean that the resnames are not automatically recognised as protein
            # ones by gromacs, but what's fun if not making your own index file?
            # Raise some info about it so people know
            resname = meta_molecule.molecule.nodes[node]['resname']
            meta_molecule.molecule.nodes[node]['resname'] = resname + term[0]
            LOGGER.info("Changing terminal resnames by their patch. Check your index file.")

            # add the modification from the modifications dict
            aname = meta_molecule.molecule.nodes[node]['atomname']
            if aname in mod_atoms.keys():
                anum_dict[aname] = atomid
                for key, value in mod_atoms[aname].items():
                    meta_molecule.molecule.nodes[node][key] = value

    mod_interactions = molecule.force_field.modifications[term].interactions
    for i in mod_interactions:
        for j in molecule.force_field.modifications[term].interactions[i]:
            molecule.add_interaction(i,
                                     (anum_dict[j.atoms[0]],
                                      anum_dict[j.atoms[1]]),
                                     j.parameters,
                                     meta=j.meta)
    return meta_molecule


def prot_termini(meta_molecule):

    max_res = max([meta_molecule.molecule.nodes[i]['resid'] for i in meta_molecule.molecule.nodes])
    modifications = [('N-ter', 1),
                     ('C-ter', max_res)]

    for mod, res in modifications:
        meta_molecule = apply_terminal_mod(meta_molecule, mod, res)

    return meta_molecule
