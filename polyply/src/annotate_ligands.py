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
from collections import defaultdict
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

    error_msg = ('Your selection {} is invalid. Was not able to assign {}'
                 'with value {}')

    out = {}
    if mol_idx:

        try:
           mol_idx = int(mol_idx[0])
        except ValueError:
           raise IOError(error_msg.format(resspec, 'mol_idx', mol_idx))

        out['mol_idx'] = mol_idx

    if resname:
        out['resname'] = resname
    if molname:
        out['molname'] = molname
    if resid:
        try:
           out['resid'] = float(resid[0])
        except ValueError:
           raise IOError(error_msg.format(resspec, 'mol_idx', mol_idx))

    return out


def _find_nodes(molecule, mol_attr):
    """
    Find nodes with resname and resid
    attribute if they are defined in mol_attr.
    """
    attr = {}
    for key in ["resname", "resid"]:
        if key in mol_attr:
           attr[key] = mol_attr[key]

    for node in molecule.nodes:
        if all([molecule.nodes[node][key] == value for
                key, value in attr.items()]):

            yield node

class AnnotateLigands(Processor):
    """
    Given some molecules defined as Ligands,
    the processor adds them to the ligated molecules
    via an edge and indicates the node as ligated.
    This will cause it to be picked up by the build
    processor and place the ligands close. The processor
    also contains an extract method, which will readout
    the coordinates from the ligated molecules and puts
    them back such that the topology order is kept.
    """

    def __init__(self, topology, ligands):
        """
        Initalize the processor with a :class:`topology`
        and a list of `ligand definitions` which
        are used both in annotating and extracting
        the positions later. Format of ligand definitions
        needs to be in the format of:

        <mol_name>#<mol_idx>-<resname>#<resid>:<mol_name>#<mol_idx>-<resname>#<resid>

        Elements of this may be omitted. <resid> and <mol_idx> cannot
        contain - or # characters.
        """
        self.topology = topology
        self.ligand_defs = defaultdict(list)

        for mol_spec, lig_spec in ligands:
            mol_attr = parse_residue_spec(mol_spec)
            lig_attr = parse_residue_spec(lig_spec)

           # check which molecules are elligable for annotation
            if "mol_idx" in mol_attr:
                mol_idxs = [mol_attr["mol_idx"]]

                if "molname" in mol_attr:
                    mol_name = mol_attr["molname"]
                    allowed_idxs = self.topology.mol_idx_by_name[mol_name]
                    if any(idx not in allowed_idxs for idx in mol_idxs):
                       msg = ("Your molecule name {} does not"
                              " match your molecule index {}.")
                       raise IOError(msg.format(mol_name, mol_idxs))

            elif "molname" in mol_attr:
                mol_name = mol_attr["molname"]
                mol_idxs = self.topology.mol_idx_by_name[mol_name]
            # if neither molname nor index are provided all molecules
            # are eligible
            else:
                mol_idxs = range(0, len(self.topology.molecules))

            # check which ligands are elligable for annotation
            if "mol_idx" in lig_attr:
                lig_idxs = [lig_attr["mol_idx"]]
            elif "molname" in lig_attr:
                lig_name = lig_attr["molname"]
                lig_idxs = self.topology.mol_idx_by_name[lig_name]
            else:
                msg = ("Your ligand selection needs to conatin a molecule idx and/or "
                       "a molecule name. Currently the selection {} contains none.")
                raise IOError(msg.format(lig_spec))

            # make sure that each lig associated with a molecule
            # is a different molecule in the topology
            total = 0
            for mol_idx in mol_idxs:
                mol_nodes = _find_nodes(topology.molecules[mol_idx], mol_attr)
                for mol_node in mol_nodes:
                    self.ligand_defs[mol_idx].append((mol_node, lig_idxs[total], mol_attr, lig_attr))
                    total += 1

    def _connect_ligands_to_molecule(self, molecule, mol_idx):
        """
        Given a `molecule` check if any of the ligand
        specs correspond to that molecule and annotate the
        ligand if the specs match. A node is added to the
        matching molecule. This node as the additional
        attributes:

        ligated:   tuple(int, node_key)
            molecule index of the ligand, node_key of the ligands node
        build: True
            this node needs to be built
        """
        current = max(molecule.nodes) + 1
        for mol_node, lig_idx, mol_attr, lig_attr in self.ligand_defs[mol_idx]:
            ligand = self.topology.molecules[lig_idx]
            lig_nodes = _find_nodes(ligand, lig_attr)

            for lig_node in lig_nodes:
                resname = ligand.nodes[lig_node]["resname"]
                molecule.add_monomer(current, resname, [(mol_node, current)])

                molecule.nodes[current]["build"] = True
                molecule.nodes[current]["ligated"] = (lig_idx,
                                                      lig_node)
                current += 1

    def split_ligands(self):
        """
        Given the ligand specs extract ligand positions from the
        molecules and put them back in the inital topology molecules
        which discribe the ligands.
        """
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

    def run_molecule(self, meta_molecule, mol_idx):
        """
        Perform the random walk for a single molecule.
        """
        self._connect_ligands_to_molecule(meta_molecule, mol_idx)
        return meta_molecule

    def run_system(self, system):
        """
        Process `system`.
        Parameters
        ----------
        system: vermouth.system.System
            The system to process. Is modified in-place.
        """
        mols = []
        for mol_idx, molecule in enumerate(system.molecules):
            mols.append(self.run_molecule(molecule, mol_idx))
        system.molecules = mols
