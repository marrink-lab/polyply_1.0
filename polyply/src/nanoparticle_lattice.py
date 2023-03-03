import collections
import copy
import itertools
import logging  # implement logging parts here

from pprint import pprint
from typing import Any, Optional
import textwrap

import networkx as nx
import numpy as np
import vermouth
import vermouth.forcefield
import vermouth.molecule
from pydantic import BaseModel, validator
from scipy.spatial import distance
from vermouth.gmx import gro  # gro library to output the file as a gro file

from polyply.src.generate_templates import (
    _expand_inital_coords,
    _relabel_interaction_atoms,
    find_atoms,
    replace_defined_interaction,
    # map_from_CoG,
    # compute_volume,
    # extract_block,
    # GenerateTemplates,
    # find_interaction_involving,
)
import polyply.src.meta_molecule
from polyply.src.load_library import load_library
from vermouth.gmx.itp import write_molecule_itp

from polyply.src.meta_molecule import MetaMolecule
from polyply.src.processor import Processor
from polyply.src.topology import Topology
from amber_nps import return_amber_nps_type  # this will be changed

"""
Internal notes:
--------------

for the UAModelSingleNP:

The program outputs LAMMPS data file "data.np" and CFG configuration file 
cfg which can be visualized with the atomeeye file 

The program reads the relaxed coordinates of a single gold-alkanethiol
nanoparticle. These coordinates are writte in the file 
singleNP.cfg. 

-- notes from singleNP: 

The program outputs LAMMPS data file "data.np" and CFG file. 

"""
# generate pydantic class for lennard jones parameters
# pair_coeff       1 1 morse  10.956 1.5830 3.0242 # These paramaeters are identical as with the other orientations and visa versa


class NanoparticleCoordinates(Processor):
    """
    uses the polyply processor to read the molecule object, assign the
    posoitonal coordinates and then utilizes networkx to assign the position attribute
    to each atomic node
    Parameters
    ----------
    Returns
    -------
    """

    def run_molecule(self, meta_molecule):

        # later we can add more complex protocols
        init_coords = _expand_inital_coords(meta_molecule)
        # this line adds the position to the molecule
        nx.set_node_attributes(meta_molecule, init_coords, "position")
        return meta_molecule


class GoldNanoparticleSingle(Processor, BaseModel):
    """
    this class is also embedded with the pydantic model which gives an error in case the
    error

    this is probably rather new so I'm happy to change this back to the original base
    class like structure as with in the other modules

    Parameters
    ----------


    Returns
    -------

    """

    # orientation: Any
    lattice_constant: float = 4.0782
    box_size: int = 100
    matrix: Any = np.array([[box_size, 0, 0], [0, box_size, 0], [0, 0, box_size]])
    n_atom_thiol: int = 9
    n_phi: int = 8
    n_theta: int = 17
    n_thiol: int = n_phi * n_theta
    gold_layer: int = 6
    n_gold: float = (
        (10 * (gold_layer**3) + 11 * gold_layer) / 3 - 5 * (gold_layer**2) - 1
    )
    # matrix store gold atom coordinates
    pos_gold: Any = np.zeros((int(n_gold), 3))  # TODO
    center: list[float] = [0.5, 0.5, 0.5]  # TODO
    phi: float = (np.sqrt(5) + 1) / 2
    V: Any  # this should be changed from any to something more appropriate
    E: Any  # ditto
    F: Any  # ditto
    # blocks and nanoparticle specific parts to work with polyply
    force_field: Any  # this needs a custom type
    top: Any  # this needs a custom typey
    NP_block: Any

    class Config:
        """ """

        arbitrary_types_allowed = True

    @validator("box_size")
    def box_size_must_be_positive(cls, v):
        """
        ensure we dont have a weird box size
        """
        if v <= 0:  # if the box size is smaller than or equal to 0
            raise ValueError("We cannot have a box size that is smaller than 0 or 0")
        return v

    @validator("matrix")
    def dimension_of_matrix(cls, v):
        """
        return the matrix model
        """
        pass
        # if v.shape != (3, 3):
        #    raise ValueError("")
        # return v

    @validator("n_atom_thiol")
    def n_atom_thiol_check(cls, v):
        """
        n_atom_thiol must be an integer
        """
        assert (
            type(v) == int
        ), "need an integer value for the number of atoms for thiols"
        return v

    def _set_ff(self) -> None:
        """ """
        self.NP_block = self.force_field[self.np_name]
        self.ligand_block = self.force_field.blocks[self.ligand_name]

    def _vertices_edges(self) -> None:
        """
        iniiate np array values
        """
        self.V = np.zeros((12, 3))
        self.E = np.zeros((2, 3, 30))
        self.F = np.zeros((3, 3, 20))

        self.V[0, :] = [1, 0, self.phi]
        self.V[1, :] = [-1, 0, self.phi]
        self.V[2, :] = [-1, self.phi, 1]
        self.V[3, :] = [self.phi, -1, 0]
        self.V[4, :] = [self.phi, 1, 0]
        self.V[5, :] = [0, self.phi, 1]
        # -ve persions of the phi values
        self.V[6:, :] = -self.V[:6, :]

        self.E[0, :, 0] = [1, 0, self.phi]
        self.E[0, :, 1] = [1, 0, self.phi]
        self.E[0, :, 2] = [1, 0, self.phi]
        self.E[0, :, 3] = [1, 0, self.phi]
        self.E[0, :, 4] = [1, 0, self.phi]
        self.E[0, :, 5] = [0, self.phi, 1]
        self.E[0, :, 6] = [0, self.phi, 1]
        self.E[0, :, 7] = [0, self.phi, 1]
        self.E[0, :, 8] = [0, self.phi, 1]
        self.E[0, :, 9] = [self.phi, 1, 0]
        self.E[0, :, 10] = [self.phi, 1, 0]
        self.E[0, :, 11] = [self.phi, 1, 0]
        self.E[0, :, 13] = [1, 0, -self.phi]
        self.E[0, :, 14] = [0, self.phi, -1]
        self.E[1, :, 0] = [-1, 0, self.phi]
        self.E[1, :, 1] = [0, -self.phi, 1]
        self.E[1, :, 2] = [self.phi, -1, 0]
        self.E[1, :, 3] = [self.phi, 1, 0]
        self.E[1, :, 4] = [0, self.phi, 1]
        self.E[1, :, 5] = [self.phi, 1, 0]
        self.E[1, :, 6] = [0, self.phi, -1]
        self.E[1, :, 7] = [-self.phi, 1, 0]
        self.E[1, :, 8] = [-1, 0, self.phi]
        self.E[1, :, 9] = [self.phi, -1, 0]
        self.E[1, :, 10] = [1, 0, -self.phi]
        self.E[1, :, 11] = [0, self.phi, -1]
        self.E[1, :, 12] = [1, 0, -self.phi]
        self.E[1, :, 13] = [0, self.phi, -1]
        self.E[1, :, 14] = [-self.phi, 1, 0]
        self.E[:, :, 15:] = -self.E[:, :, :15]

        self.F[1, :, 0] = [1, 0, self.phi]
        self.F[1, :, 0] = [-1, 0, self.phi]
        self.F[2, :, 0] = [0, -self.phi, 1]
        self.F[0, :, 1] = [1, 0, self.phi]
        self.F[1, :, 1] = [0, -self.phi, 1]
        self.F[2, :, 1] = [self.phi, -1, 0]
        self.F[0, :, 2] = [1, 0, self.phi]
        self.F[1, :, 2] = [self.phi, -1, 0]
        self.F[2, :, 2] = [self.phi, 1, 0]
        self.F[0, :, 3] = [1, 0, self.phi]
        self.F[1, :, 3] = [self.phi, 1, 0]
        self.F[2, :, 3] = [0, self.phi, 1]
        self.F[0, :, 4] = [1, 0, self.phi]
        self.F[1, :, 4] = [0, self.phi, 1]
        self.F[2, :, 4] = [-1, 0, self.phi]
        self.F[0, :, 5] = [0, self.phi, 1]
        self.F[1, :, 5] = [self.phi, 1, 0]
        self.F[2, :, 5] = [0, self.phi, -1]
        self.F[0, :, 6] = [0, self.phi, 1]
        self.F[1, :, 6] = [0, self.phi, -1]
        self.F[2, :, 6] = [-self.phi, 1, 0]
        self.F[0, :, 7] = [0, self.phi, 1]
        self.F[1, :, 7] = [-self.phi, 1, 0]
        self.F[2, :, 7] = [-1, 0, self.phi]
        self.F[0, :, 8] = [self.phi, 1, 0]
        self.F[1, :, 8] = [self.phi, -1, 0]
        self.F[2, :, 8] = [1, 0, -self.phi]
        self.F[0, :, 9] = [self.phi, 1, 0]
        self.F[1, :, 9] = [1, 0, -self.phi]
        self.F[2, :, 9] = [0, self.phi, -1]
        self.F[:, :, 10:] = -self.F[:, :, :10]

        scaling = self.lattice_constant / np.sqrt(2 * (1 + self.phi * self.phi))
        self.V = self.V * scaling
        self.E = self.E * scaling
        self.F = self.F * scaling

    def _generate_positions(self) -> None:
        """
        Need some general notes on how the core was generated.
        """
        self._vertices_edges()  # need brief description of this function does
        self.pos_gold[0, :] = self.center

        # coordinates of the second innermost layer atoms
        for i in range(12):
            self.pos_gold[i + 1, 0] = self.V[i, 0] / self.box_size + self.center[0]
            self.pos_gold[i + 1, 1] = self.V[i, 1] / self.box_size + self.center[1]
            self.pos_gold[i + 1, 2] = self.V[i, 2] / self.box_size + self.center[2]

        # coordinates of the third innermost layer atoms
        for i in range(12):
            self.pos_gold[i + 13, 0] = self.V[i, 0] * 2 / self.box_size + self.center[0]
            self.pos_gold[i + 13, 1] = self.V[i, 1] * 2 / self.box_size + self.center[1]
            self.pos_gold[i + 13, 2] = self.V[i, 2] * 2 / self.box_size + self.center[2]

        # coordinates of the fourth innermost layer atoms
        for i in range(30):
            self.pos_gold[i + 25, 0] = (
                self.E[0, 0, i] + self.E[1, 0, i]
            ) / self.box_size + self.center[0]
            self.pos_gold[i + 25, 1] = (
                self.E[0, 1, i] + self.E[1, 1, i]
            ) / self.box_size + self.center[1]
            self.pos_gold[i + 25, 2] = (
                self.E[0, 2, i] + self.E[1, 2, i]
            ) / self.box_size + self.center[2]

        for N in range(3, self.gold_layer):
            n_atom_to_now = int(
                ((10 * (N - 1) ** 3 + 11 * (N - 1)) / 3 - 5 * ((N - 1) ** 2) - 1)
            )

            for i in range(12):
                self.pos_gold[n_atom_to_now + i][0] = (
                    self.V[i][0] * (N - 1) / self.box_size + self.center[0]
                )
                self.pos_gold[n_atom_to_now + i][1] = (
                    self.V[i][1] * (N - 1) / self.box_size + self.center[1]
                )
                self.pos_gold[n_atom_to_now + i][2] = (
                    self.V[i][2] * (N - 1) / self.box_size + self.center[2]
                )
            n_atom_to_now = n_atom_to_now + 12

            for i in range(30):
                for j in range(N - 2):
                    self.pos_gold[n_atom_to_now + (i - 1) * (N - 2) + j][0] = (
                        self.E[0][0][i] * (N - 1)
                        + (self.E[1][0][i] - self.E[0][0][i]) * j
                    ) / self.box_size + self.center[0]
                    self.pos_gold[n_atom_to_now + (i - 1) * (N - 2) + j][1] = (
                        self.E[0][1][i] * (N - 1)
                        + (self.E[1][1][i] - self.E[0][1][i]) * j
                    ) / self.box_size + self.center[1]
                    self.pos_gold[n_atom_to_now + (i - 1) * (N - 2) + j][2] = (
                        self.E[0][2][i] * (N - 1)
                        + (self.E[1][2][i] - self.E[0][2][i]) * j
                    ) / self.box_size + self.center[2]

            n_atom_to_now = n_atom_to_now + 30 * (N - 2)

            for i in range(20):
                for m in range(N - 3):
                    for n in range(N - m - 2):
                        self.pos_gold[
                            n_atom_to_now + (2 * N - m - 4) * (m - 1) // 2 + n, 0
                        ] = (
                            self.F[0, 0, i] * (N - 1)
                            + (self.F[1, 0, i] * (N - 1) - self.F[0, 0, i] * (N - 1))
                            * m
                            / (N - 1)
                            + (self.F[2, 0, i] * (N - 1) - self.F[0, 0, i] * (N - 1))
                            * n
                            / (N - 1)
                        ) / self.box_size + self.center[
                            0
                        ]
                        self.pos_gold[
                            n_atom_to_now + (2 * N - m - 4) * (m - 1) // 2 + n, 1
                        ] = (
                            self.F[0, 1, i] * (N - 1)
                            + (self.F[1, 1, i] * (N - 1) - self.F[0, 1, i] * (N - 1))
                            * m
                            / (N - 1)
                            + (self.F[2, 1, i] * (N - 1) - self.F[0, 1, i] * (N - 1))
                            * n
                            / (N - 1)
                        ) / self.box_size + self.center[
                            1
                        ]
                        self.pos_gold[
                            n_atom_to_now + (2 * N - m - 4) * (m - 1) // 2 + n, 2
                        ] = (
                            self.F[0, 2, i] * (N - 1)
                            + (self.F[1, 2, i] * (N - 1) - self.F[0, 2, i] * (N - 1))
                            * m
                            / (N - 1)
                            + (self.F[2, 2, i] * (N - 1) - self.F[0, 2, i] * (N - 1))
                            * n
                            / (N - 1)
                        ) / self.box_size + self.center[
                            2
                        ]
                n_atom_to_now += (N - 2) * (N - 3) // 2

    def create_gro(
        self, outfile: str, velocities=False, box="10.0, 10.0, 10.0"
    ) -> None:
        """
        plot out the lattice in a 3D structure - generating the coordinates part is easy. Now
        the hard part is generating the gro file
        this is somewhat working ... nowhere near what I want - need to review the core generation code I made above

        Parameters
        ----------

        Returns
        -------

        """
        velocity_fmt = ""
        self._generate_positions()  # generate positions for the gold lattice core
        core_numpy_coords = self.pos_gold
        coords = [[j for j in i] for i in core_numpy_coords]
        gold_str = [f"1AU  AU  {i + 1}" for i in range(0, len(coords))]

        # just need to figure out GRO CONTENT and COORINDATES
        velocities = [[x + 1 for x in line] for line in coords]
        with open(str(outfile), "w") as output:
            output.write("NP\n")
            output.write(str(len(coords)) + "\n")
            for atom, coords, vels in zip(gold_str, coords, velocities):
                # print(atom, coords, vels)
                output.write(
                    ("{}{:8.3f}{:8.3f}{:8.3f}" + velocity_fmt + "\n").format(
                        atom, *itertools.chain(coords, vels)
                    )
                )

            output.write(box)
            output.write("\n")

    def _identify_attachment_index(self):
        """

        Find index of atoms that corresponds to the atom on the ligand
        which you wish to bond with the ligand on the nanoparticle core

        Parameters
        ----------
        Returns
        -------

        """
        self.attachment_index = None
        for index, entry in enumerate(self.ligand_block.atoms):
            if entry["atomname"] == self.ligand_anchor:
                attachment_index = index
        return attachment_index


class gold_models(Processor):
    """
    we have a number of gold nanoparticle cores based on the amber forcefield
    that is avaliable as published here:

    https://pubs.acs.org/doi/abs/10.1021/acs.jctc.5b01053

    analyzing and creating the class depending on the
    number of atoms of gold we want
    """

    def __init__(self):
        """
        Parameters
        ----------
        Returns
        -------
        """
        # I will put a working sample here for now
        self.sample = return_amber_nps_type(
            "au144_OPLS_bonded"
        )  # call the appropriate amber code type

    def _extract_block_atom(self, molecule: str, atomname: str, defines: dict) -> None:
        """
        Given a `vermouth.molecule` and a `resname`
        extract the information of a block from the
        molecule definition and replace all defines
        if any are found.

        this has been adopted from the generate_templates
        function as in that scenario it cannot distinguish
        between different parts of a single resname. Hence,
        this would extract the core of a single type (for
        example, AU here) and convert that into a core

        Parameters
        ----------
        molecule:  :class:vermouth.molecule.Molecule
        atomname:   str
        defines:   dict
        dict of type define: value

        Returns
        -------
        :class:vermouth.molecule.Block
        """
        nodes = find_atoms(molecule, "atype", atomname)
        resid = molecule.nodes[nodes[0]]["resid"]
        block = vermouth.molecule.Block()

        # select all nodes with the same first resid and
        # make sure the block node labels are atomnames
        # also build a correspondance dict between node
        # label in the molecule and in the block for
        # relabeling the interactions
        mapping = {}
        for node in nodes:
            attr_dict = molecule.nodes[node]
            if attr_dict["resid"] == resid:
                block.add_node(attr_dict["atomname"], **attr_dict)
                mapping[node] = attr_dict["atomname"]

        for inter_type in molecule.interactions:
            for interaction in molecule.interactions[inter_type]:
                if all(atom in mapping for atom in interaction.atoms):
                    interaction = replace_defined_interaction(interaction, defines)
                    interaction = _relabel_interaction_atoms(interaction, mapping)
                    block.interactions[inter_type].append(interaction)

        for inter_type in [
            "bonds",
            "constraints",
            "virtual_sitesn",
            "virtual_sites2",
            "virtual_sites3",
            "virtual_sites4",
        ]:
            block.make_edges_from_interaction_type(inter_type)
        return block

    def core_generate_coordinates(self) -> None:
        """
        now I need to ensure that this function is

        using vermouth to strip the original amber nanoparticle itps
        and read in as residue that can be read

        Parameters
        ----------
        Returns
        -------
        """
        name = "test"
        self.ff = vermouth.forcefield.ForceField(name=name)
        vermouth.gmx.itp_read.read_itp(self.sample, self.ff)

        core_molecule = self.ff.blocks["NP2"].to_molecule()
        newblock = self._extract_block_atom(core_molecule, "AU", {})
        newblock.nrexcl = 3
        np_molecule = newblock.to_molecule()
        # np_molecule = NanoparticleCoordinates().run_molecule(newblock.to_molecule())
        for node in np_molecule.nodes:
            np_molecule.nodes[node]["resname"] = "TEST"
        graph = MetaMolecule._block_graph_to_res_graph(np_molecule)
        meta_mol = MetaMolecule(graph, force_field=self.ff, mol_name="test")
        np_molecule.meta["moltype"] = "TEST"
        meta_mol.molecule = np_molecule

        self.newmolecule = np_molecule
        self.np_top = Topology(name=name, force_field=self.ff)
        self.np_top.molecules = [meta_mol]
        # nx.set_node_attributes(meta_molecule, init_coords, "position")

    def write_itp(self) -> None:
        """

        Parameters
        ----------

        Returns
        -------

        """
        with open("np.itp", "w") as outfile:
            write_molecule_itp(self.newmolecule, outfile)

    def create_gro(self, write_path: str) -> None:
        """
        ideally, we generate the coordinates with this function
        and then store it within a 'coordinates' object. The form
        of which I am not certain yet.

        Parameters
        ----------

        Returns
        -------
        """
        self.top.box = np.array([3.0, 3.0, 3.0])
        system = self.np_top.convert_to_vermouth_system()
        vermouth.gmx.gro.write_gro(
            system,
            write_path,
            precision=7,
            title="gold nanoparticle core",
            box=self.top.box,
            defer_writing=False,
        )

    def load_models(self) -> None:
        """ """
        pass


# main code executable
if __name__ == "__main__":
    sampleNPCore = GoldNanoparticleSingle()
    sampleNPCore._generate_positions()
    sampleNPCore.create_gro(
        "/home/sang/Desktop/example_gold.gro"
    )  # this works but not fully

    # generate the core of the opls force field work
    gold_model = gold_models()
    gold_model.core_generate_coordinates()
