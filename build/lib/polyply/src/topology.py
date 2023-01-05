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
Provides a class used to describe a gromacs topology and all assciated data.
"""
import os
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import numpy as np
import networkx as nx
from vermouth.system import System
from vermouth.forcefield import ForceField
from vermouth.gmx.gro import read_gro
from vermouth.pdb import read_pdb
from vermouth.molecule import Interaction
from .top_parser import read_topology
from .linalg_functions import center_of_geometry

COORD_PARSERS = {"pdb": read_pdb,
                 "gro": read_gro}

# small wrapper that is neccessiataed
# by the fact that gro and pdb readers
# return a molecule and a list respectively


def _coord_parser(path, extension):
    reader = COORD_PARSERS[extension]
    molecules = reader(path, exclude=())
    if extension == "pdb":
        molecule = molecules[0]
        for new_mol in molecules[1:]:
            molecule.merge_molecule(new_mol)
    else:
        molecule = molecules
    positions = np.array(list(nx.get_node_attributes(molecule, "position").values()))
    return positions


def replace_defined_interaction(interaction, defines):
    """
    Given an `interaction` check if parameters
    are defined in a list of defines and replace
    by the corresponding numeric value.

    Parameters
    -----------
    interaction: :tuple:`vermouth.molecule.Interaction`
    defines:  dict
      dictionary of type [define]:value

    Returns
    --------
    interaction
      interaction with replaced defines
    """
    new_parameters = []
    for parameter in interaction.parameters:
        if parameter in defines:
            values = defines[parameter]
            for new_param in values:
                new_parameters.append(new_param)
        else:
            new_parameters.append(parameter)

    interaction.parameters[:] = new_parameters[:]

    return interaction


def lorentz_berthelot_rule(sig_A, sig_B, eps_A, eps_B):
    """
    Lorentz-Berthelot rules for combining LJ paramters.

    Parameters
    -----------
    sig_A:  float
    sig_B:  float
        input sigma values
    eps_A:  float
    eps_B:  float
        input epsilon values

    Returns
    --------
    float
        sigma
    float
        epsilon
    """
    sig = (sig_A + sig_B)/2.0
    eps = (eps_A * eps_B)**0.5
    return sig, eps


def geometric_rule(C6_A, C6_B, C12_A, C12_B):
    """
    Geometric combination rule for combining
    LJ parameters.

    Parameters:
    -----------
    C6_A:  float
    C6_B:  float
        input C6 values
    C12_A:  float
    C12_B:  float
        input C12 values

    Returns:
    --------
    float
         C6
    float
         C12
    """
    C6 = (C6_A * C6_B)**0.5
    C12 = (C12_A * C12_B)**0.5
    return C6, C12


def _wildcard_dih(atoms, idxs):
    """
    Given atoms and idxs which define a
    pattern return a tuple consisting of
    a entry from atoms if the pattern is
    an int and else the value in idxs.
    """
    atom_key = []
    for idx in idxs:
        if type(idx) is int:
            atom_key.append(atoms[idx])
        else:
            atom_key.append(idx)

    return tuple(atom_key)

def match_dihedral_interaction_types(atoms, interaction_dict):
    """
    Dihedral interaction types within GROMACS can have wildcard
    pattern matching. The wildcard is indicated by an 'X' and
    grompp matches against the atoms and the reverse atoms. For
    example 'X C C C' would apply to an interaction with atoms
    'N C C C' as well as 'C C C N'. Also note that the most
    specific pattern takes precedence.

    Parameters
    ----------
    atoms: abc.iteratable
        list of atom-types
    interaction_dict: dict
        dict of interaction types

    Returns
    --------
    tuple
        a tuple of 4 atom indices, which are the matching key
        to the interaction dict.
    """
    patterns = [(0, 1, 2, 3),
                ('X', 1, 2, 3),
                (0, 'X', 2, 3),
                (0, 1, 'X', 3),
                ('X', 1, 2, 'X'),
                ('X', 'X', 2, 3),
                (0, 'X', 'X', 3),
                ('X', 1, 'X', 3),
                ('X', 'X', 'X', 3)]

    for pattern in patterns:
        key = _wildcard_dih(atoms, pattern)
        if key in interaction_dict:
            return key
        elif key[::-1] in interaction_dict:
            return key[::-1]

    return None


class Topology(System):
    """
    Ties together vermouth molecule definitions, and
    Gromacs topology information.

    Parameters
    ----------
    force_field: :class:`vermouth.forcefield.ForceField`
        A force field object.
    name: str, optional
        The name of the topology.

    Attributes
    ----------
    molecules: list[:class:`~vermouth.molecule.Molecule`]
        The molecules in the system.
    force_field: a :class:`vermouth.forcefield.ForceField`
    nonbond_params: dict
        A dictionary of all nonbonded parameters
    types: dict
        A dictionary of all typed parameter
    defines: list
        A list of everything that is defined
    """

    def __init__(self, force_field, name=None):
        super().__init__(force_field)
        self.name = name
        self.defaults = {}
        self.defines = {}
        self.description = []
        self.atom_types = {}
        self.types = defaultdict(lambda: defaultdict(list))
        self.nonbond_params = {}
        self.mol_idx_by_name = defaultdict(list)
        self.persistences = []
        self.distance_restraints = defaultdict(dict)
        self.volumes = {}

    def preprocess(self):
        """
        Apply all defaults, generate pairs, convert non-bonded
        units. It performs most of the conversion which otherwise
        is done by grompp.
        """
        self.gen_pairs()
        # we need to replace defines before doing bonded interactions
        self.replace_defines()
        self.gen_bonded_interactions()
        # only convert if we not already have sig-eps form
        if self.defaults['comb-rule'] == 1:
            self.convert_nonbond_to_sig_eps()

    def replace_defines(self):
        """
        Replace all interaction paramers with defined parameters.
        """
        # Note that a topology cannot define and generate links so
        # they don't need to be replaced or handled elsewhere
        for block in self.force_field.blocks.values():
            for interactions in block.interactions.values():
                for interaction in interactions:
                    new_interaction = replace_defined_interaction(interaction, self.defines)

    def gen_pairs(self):
        """
        If pairs default is set to yes the non-bond params are
        generated for all pairs according to the combination
        rules. Regardless of if pairs is set or not the self
        interactions from atomtypes are added to `self.nonbond_params`.
        Note that nonbond_params takes precedence over atomtypes and
        generated pairs.
        """
        comb_funcs = {1.0: lorentz_berthelot_rule,
                      2.0: geometric_rule,
                      3.0: lorentz_berthelot_rule}

        comb_rule = comb_funcs[self.defaults["comb-rule"]]

        if self.defaults["gen-pairs"] == "yes":
            for atom_type_A, atom_type_B in combinations(self.atom_types, r=2):
                if frozenset([atom_type_A, atom_type_B]) not in self.nonbond_params:
                    nb1_A = self.atom_types[atom_type_A]["nb1"]
                    nb2_A = self.atom_types[atom_type_A]["nb2"]
                    nb1_B = self.atom_types[atom_type_B]["nb1"]
                    nb2_B = self.atom_types[atom_type_B]["nb2"]
                    nb1, nb2 = comb_rule(nb1_A, nb1_B, nb2_A, nb2_B)
                    self.nonbond_params.update({frozenset([atom_type_A, atom_type_B]):
                                                {"nb1": nb1, "nb2": nb2}})

        for atom_type in self.atom_types:
            if frozenset([atom_type, atom_type]) not in self.nonbond_params:
                nb1 = self.atom_types[atom_type]["nb1"]
                nb2 = self.atom_types[atom_type]["nb2"]
                self.nonbond_params.update({frozenset([atom_type, atom_type]):
                                            {"nb1": nb1, "nb2": nb2}})

    def gen_bonded_interactions(self):
        """
        Check for each interaction, if the parameter list is empty. If it is
        check if the parameters are defined in the bonded types directive
        of the topology. This essentially does part of the step done by
        grompp.

        Updates
        -------
        self.molecules.interactions
        self.blocks.interactions

        Raises
        ------
        OSError
            no match for an interaction in the bonded types provided
        """
        # We loop over blocks not molecules, because there is only
        # one unique block per molecule, whereas there cab be a large number of
        # duplicate molecules in the topology.molecules list. As the interactions
        # in those molecules are still references to the blocks updating them here
        # will propagate into the molecules. This works except for one case where
        # a single dihedral directive is expanded into multiple ones. Updating the
        # block dict will not propagate into the molecules.
        for mol_name, block in self.force_field.blocks.items():
            additional_interactions = defaultdict(list)
            for inter_type, interactions in block.interactions.items():
                # these interactions have no types associated
                if inter_type in ["pairs", "exclusions", "virtual_sitesn",
                                  "virtual_sites2", "virtual_sites3", "virtual_sites4"]:
                    continue

                for interaction in interactions:
                    if len(interaction.parameters) == 1:
                        # Some force-fields - in GMX library only OPLS - use bond-type
                        # definitions. Each atomtype matches one bond-type, which
                        # in turn matches an expression in the bondedtypes section
                        if "_FF_OPLS" in self.defines or "_FF_OPLS_AA" in self.defines:
                            atoms = tuple(self.atom_types[block.nodes[node]["atype"]]["bond_type"]
                                          for node in interaction.atoms)
                        # Other force-fields like charmm and amber use the atomtype directly for
                        # matching the bondded types
                        else:
                            atoms = tuple(block.nodes[node]["atype"] for node in interaction.atoms)

                        # now we match the atom or bondtypes to the types defined in the topology
                        if atoms in self.types[inter_type]:
                            new_params = self.types[inter_type][atoms]
                        elif atoms[::-1] in self.types[inter_type]:
                            new_params  = self.types[inter_type][atoms[::-1]]
                        # dihedrals are more complicated because they are treated as symmetric and
                        # can have wild-cards
                        elif inter_type in "dihedrals":
                            match = match_dihedral_interaction_types(atoms, self.types[inter_type])
                            if match:
                                new_params = self.types[inter_type][match]
                            else:
                                msg = ("In section dihedrals interaction of atoms {} has no "
                                       "corresponding bonded type.")
                                atoms = " ".join(list(map(lambda x: str(x), interaction.atoms)))
                                raise OSError(msg.format(atoms))

                        else:
                            msg = ("In section {} interaction of atoms {} has no corresponding "
                                   "bonded type.")
                            atoms = " ".join(list(map(lambda x: str(x), interaction.atoms)))
                            raise OSError(msg.format(inter_type, atoms))

                        for idx, (new_param, meta) in enumerate(new_params):
                            if not meta:
                                meta = {}
                            # there is always at least one interaction in molecule, which
                            # needs to get the typed parameters
                            if idx == 0:
                                interaction.parameters[:] = new_param[:]
                                interaction.meta.update(meta)
                            # however, sometimes a single interaction term needs to be
                            # expanded (i.e. a single statment spwans multiple interactions)
                            # In that case we update the parameters of the first term and
                            # need to add the other interactions additionally
                            else:
                                new_interaction = Interaction(atoms=tuple(interaction.atoms),
                                                              parameters=new_param,
                                                              meta=meta)
                                additional_interactions[inter_type].append(new_interaction)


            # here we add the expanded interactions into the molecules
            for mol_idx in self.mol_idx_by_name[mol_name]:
                for inter_type, new_inters in additional_interactions.items():
                    self.molecules[mol_idx].molecule.interactions[inter_type] += new_inters

    def convert_nonbond_to_sig_eps(self):
        """
        Convert all nonbond_params to sigma epsilon form of the
        LJ potential. Note that this assumes the parameters
        are in A, B form.
        """
        for atom_pair in self.nonbond_params:
            nb1 = self.nonbond_params[atom_pair]["nb1"]
            nb2 = self.nonbond_params[atom_pair]["nb2"]

            if nb2 != 0:
                sig = (nb2/nb1)**(1.0/6.0)
            else:
                sig = 0

            if nb1 != 0:
                eps = nb1**2.0/(4*nb2)
            else:
                eps = 0

            self.nonbond_params.update({atom_pair: {"nb1": sig, "nb2": eps}})

    def add_positions_from_file(self, path, skip_res=[], resolution='mol'):
        """
        Add positions to molecules in topology from coordinate file.
        Depending on the resolution set they are either added to the
        meta_molecule or the molecules. With `skip_res` residues can
        be skipped by name.

        Note that molecule coordinates will also set the meta_molecule
        coordinates. Coordinates for a single residue must be complete.

        Currently .gro and .pdb file parsers are supported for the
        coordinate reading. See `_coord_parsers` for more information.

        Parameters
        ----------
        path: :class:`pathlib.Path`
            path to coordinate file
        skip_res: list[str]
            list of resnames to skip
        resolution: str
            choice of meta_mol or mol
        """
        path = Path(path)
        extension = path.suffix.casefold()[1:]
        positions = _coord_parser(path, extension)
        max_coords = len(positions)
        total = 0
        for meta_mol in self.molecules:
            for meta_node in meta_mol.nodes:
                resname = meta_mol.nodes[meta_node]["resname"]
                # the fragment graph nodes are not sorted so we sort them by index
                # as defined in the itp-file to capture cases, where the molecule
                # graph nodes are permuted with respect to the index
                idx_nodes = nx.get_node_attributes(meta_mol.nodes[meta_node]['graph'], "index")
                mol_nodes = sorted(idx_nodes, key=idx_nodes.get)
                # skip residue if resname is to be skipped or
                # if the no more coordinates are available
                # in that case we want to build the node and
                # backmap it
                if resname in skip_res or total >= max_coords:
                    meta_mol.nodes[meta_node]["build"] = True
                    meta_mol.nodes[meta_node]["backmap"] = True
                # here we only add meta_molecule coordiantes
                # in that case we only want to backmap
                elif resolution == 'meta_mol':
                    meta_mol.nodes[meta_node]["position"] = positions[total]
                    meta_mol.nodes[meta_node]["backmap"] = True
                    meta_mol.nodes[meta_node]["build"] = False
                    total += 1
                # here we set molecule coordinates in that case we neither
                # want to backmap nor build these nodes
                else:
                    start = total
                    for mol_node in mol_nodes:
                        # of the coordinates for a single residue are incomplete
                        # we raise an error because otherwise we would set them
                        # based on a non-complete residue
                        try:
                            meta_mol.molecule.nodes[mol_node]["position"] = positions[total]
                        except IndexError:
                            resid = meta_mol.nodes[meta_node]['resid']
                            mol_name = meta_mol.mol_name
                            msg = (f"Trying to add position to {resname}{resid} of "
                                    "molecule {mol_name}, but missing coordinates. "
                                    "Coordinates of residues must be complete or "
                                    "build from scratch. Partial reconstruction is "
                                    "not supported.")
                            raise IOError(msg) from IndexError
                        total += 1

                    meta_mol.nodes[meta_node]["position"] = center_of_geometry(positions[start:total])
                    meta_mol.nodes[meta_node]["build"] = False
                    meta_mol.nodes[meta_node]["backmap"] = False

    def convert_to_vermouth_system(self):
        system = System()
        system.molecules = []
        system.force_field = self.force_field

        for meta_mol in self.molecules:
            system.molecules.append(meta_mol.molecule)

        return system

    @classmethod
    def from_gmx_topfile(cls, path, name):
        """
        Read a gromacs topology file and return an topology object.

        Parameters
        ----------
        path:  str
           The name of the topology file
        name:  str
           The name of the system
        """
        with open(path, 'r') as _file:
            lines = _file.readlines()

        cwdir = os.path.dirname(path)
        force_field = ForceField(name)
        topology = cls(force_field=force_field, name=name)
        read_topology(lines=lines, topology=topology, cwdir=cwdir)
        return topology
