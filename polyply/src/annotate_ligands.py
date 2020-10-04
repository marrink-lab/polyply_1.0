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
from .processor import Processor

def parse_residue_spec(resspec):
    """
    Light version of:
    vermouth.processors.annotate_mud_mod.parse_residue_spec

    Parse a residue specification:
    <mol_name>#<mol_idx>-<resname>#<resid>

    Returns a dictionary with keys 'mol_name',
    'mol_idx', and 'resname' for the fields that are specified.
    Resid will be an int.

    Parameters
    ----------
    resspec: str
    Returns
    -------
    dict
    """
    molname = None
    resname = None
    mol_idx = None
    resid = None

    mol_name_idx, *res = resspec.split('-', 1)
    molname, *mol_idx = mol_name_idx.split('#', 1)

    if res:
        resname, *resid = res[0].split('#', 1)

    out = {}
    if mol_idx:
        mol_idx = int(mol_idx)
        out['molidx'] = mol_idx[0]
    if resname:
        out['resname'] = resname
    if molname:
        out['molname'] = molname
    if resid:
        out['resid'] = float(resid[0])

    return out


def _find_nodes(molecule, mol_attr):
    nodes = []
    attr = {}
    for key in ["resname", "resid"]:
        if key in mol_attr:
           attr[key] = mol_attr[key]

    for node in molecule.nodes:
        if all([molecule.nodes[node][attr] == value for
                attr, value in attr.items()]):

            nodes.append(node)

    return nodes


class AnnotateLigands(Processor):
    """
    Add Ligands to your system. This is mostly a workaround
    for placing ions in a clever manner.
    """

    def __init__(self, topology, ligands):
        """
        """
        self.ligands = ligands
        self.topology = topology

    def _connect_ligands_to_molecule(self, meta_molecule):
        """

        """

        for mol_spec, lig_spec in self.ligands:
            mol_attr = parse_residue_spec(mol_spec)
            lig_attr = parse_residue_spec(lig_spec)

            if "mol_idx" in mol_attr:
                molecules = [self.topology.molecules[mol_attr["mol_idx"]]]
            else:
                mol_name = mol_attr["molname"]
                mol_idxs = self.topology.mol_idx_by_name[mol_name]
                molecules = [self.topology.molecules[mol_idx] for mol_idx in mol_idxs]

            if "mol_idx" in lig_attr:
                ligands = [self.topology.molecules[lig_attr["mol_idx"]]]
                lig_idxs = [lig_attr["mol_idx"]]
            else:
                lig_name = lig_attr["molname"]
                lig_idxs = self.topology.mol_idx_by_name[lig_name]
                ligands = [self.topology.molecules[mol_idx] for mol_idx in lig_idxs]

            # sanity check
            if len(ligands) < len(molecules):
                raise IOError

            lig_count = 0
            for molecule in molecules:
                mol_nodes = _find_nodes(molecule, mol_attr)
                current = len(molecule.nodes) + 1

                for mol_node in mol_nodes:
                    ligand = ligands[lig_count]
                    lig_nodes = _find_nodes(ligand, lig_attr)

                    for lig_node in lig_nodes:
                        resname = ligand.nodes[lig_node]["resname"]

                        if lig_node in lig_nodes:
                            molecule.add_monomer(current, resname, [(mol_node, current)])
                        else:
                            molecule.add_monomer(current, resname, [])

                        molecule.nodes[current]["build"] = True
                        molecule.nodes[current]["ligated"] = (lig_idxs[lig_count],
                                                        lig_node)
                        current += 1

                    lig_count += 1

    def split_ligands(self):
        for molecule in self.topology.molecules:
            remove_nodes = []

            for node in molecule.nodes:
                if "ligated" in molecule.nodes[node]:
                    lig_idx, lig_node = molecule.nodes[node]["ligated"]
                    pos = molecule.nodes[node]["position"]
                    self.topology.molecules[lig_idx].nodes[lig_node]["position"] = pos
                    remove_nodes.append(node)

            for node in remove_nodes:
                molecule.remove_node(node)

    def run_molecule(self, meta_molecule):
        """
        Perform the random walk for a single molecule.
        """
        self._connect_ligands_to_molecule(meta_molecule)
        return meta_molecule
