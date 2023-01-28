"""
testing debugging with dap-mode 
"""
import sys
import numpy as np
from scipy.spatial import distance
import networkx as nx
import vermouth

# import textwrap
# from pathlib import Path

import polyply.src.meta_molecule
from polyply.src.topology import Topology
from polyply.src.processor import Processor
from polyply.src.generate_templates import GenerateTemplates

# from polyply import TEST_DATA


from polyply.src.meta_molecule import MetaMolecule
import polyply.src.map_to_molecule
import polyply.src.polyply_parser

# from polyply.src.generate_templates import _expand_initial_coords
from polyply.src.load_library import load_library
from polyply.src.minimizer import optimize_geometry
from polyply.src.apply_links import MatchError, ApplyLinks
from polyply.src.linalg_functions import center_of_geometry

from polyply.src.generate_templates import (
    find_atoms,
    _expand_inital_coords,
    _relabel_interaction_atoms,
    compute_volume,
    map_from_CoG,
    GenerateTemplates,
    find_interaction_involving,
)

from polyply.src.linalg_functions import center_of_geometry
from vermouth.molecule import Interaction, Molecule, Block
from vermouth.gmx.itp import write_molecule_itp
import vermouth.forcefield
import vermouth.molecule
from vermouth.gmx.itp_read import read_itp
from vermouth.molecule import Interaction

# setting up pydantic checking just in case - TODO
from typing import List, Optional
from pydantic import BaseModel


class NanoparticleCoordinates(Processor):
    """
    uses the polyply processor to read the molecule object, assign the
    posoitonal coordinates and then utilizes networkx to assign the position attribute
    to each atomic node
    """

    def run_molecule(self, meta_molecule):
        # later we can add more complex protocols
        init_coords = _expand_inital_coords(meta_molecule)
        # this line adds the position to the molecule
        nx.set_node_attributes(meta_molecule, init_coords, "position")
        return meta_molecule


class NanoparticleLigandCoordinates(Processor):
    """ """

    def run_molecule(self, meta_molecule):
        # later we can add more complex protocols
        pos_dict = nx.spring_layout(meta_molecule, dim=3)
        nx.set_node_attributes(meta_molecule, pos_dict, "position")
        return meta_molecule


class NanoparticleGenerator(Processor):
    """
    initial setup of the class for nanoparticle compatible polyply generation

    we take the core itp file (in this case scenario, we tried using a C60 core from the montielli
    with the BCM ligand to create the PBCM nanoparticle with this approach.
    """

    def __init__(self, force_field, ligand_name, ligands_N, ligand_anchor, pattern):
        self.ligand_name = "BCM"
        self.np_name = "C60"
        self.ligands_N = ligands_N
        self.force_field = force_field
        # self.force_field = load_library(self.ff_name, ["oplsaaLigParGen"], None)
        self.top = Topology(self.force_field, name="test")
        self.pattern = pattern
        self.ligand_anchor = ligand_anchor
        # initialize inherited class
        # super().__init__(*args, **kwargs)

    def _set_ff(self):
        """
        take the relevant blocks names from the ff data directory to
        read to create the functionalized NP
        """
        self.NP_block = self.force_field.blocks[self.np_name]
        self.ligand_block = self.force_field.blocks[self.ligand_name]

    def _identify_indices_for_core_attachment(self):
        """
        based on the coordinates of the NP core provided,

        this function could just return a list - not sure it has to be a class
        attribute
        """
        # assign molecule object
        np_molecule = NanoparticleCoordinates().run_molecule(
            self.NP_block.to_molecule()
        )
        # init_coords = expand_initial_coords(self.NP_block)
        # In the case of a spherical core, find the radius of the core
        core_numpy_coords = np.asarray(
            list((nx.get_node_attributes(np_molecule, "position").values()))
        )
        # find the center of geometry of the core
        CoG = center_of_geometry(core_numpy_coords)
        # compute the radius
        radius = distance.euclidean(CoG, core_numpy_coords[0])
        # Find indices in the case of Janus and Striped particles
        length = radius * 2
        minimum_threshold, maximum_threshold = min(core_numpy_coords[:, 2]), max(
            core_numpy_coords[:, 2]
        )
        # coordinates we wish to return
        # the logic of this part takes from the original NPMDPackage code
        if self.pattern == None:
            core_values = {}
            for index, entry in enumerate(core_numpy_coords):
                core_values[index] = entry
            self.core_indices = [core_values]

        elif self.pattern == "Striped":
            core_striped_values = {}
            core_ceiling_values = {}
            threshold = length / 3  # divide nanoparticle region into 3
            for index, entry in enumerate(core_numpy_coords):
                if (
                    entry[2] > minimum_threshold + threshold
                    and entry[2] < maximum_threshold - threshold
                ):
                    core_striped_values[index] = entry
                else:
                    core_ceiling_values[index] = entry
            self.core_indices = [core_striped_values, core_ceiling_values]

        elif self.pattern == "Janus":
            core_top_values = {}
            core_bot_values = {}
            threshold = length / 2  # divide nanoparticle region into 2
            for index, entry in enumerate(core_numpy_coords):
                if entry[2] > minimum_threshold + threshold:
                    core_top_values[index] = entry
                else:
                    core_bot_values[index] = entry
            self.core_indices = [core_top_values, core_bot_values]

    def _identify_attachment_index(self):
        """
        Find index of atoms that corresponds to the atom on the ligand
        which you wish to bond with the ligand on the nanoparticle core
        """
        self.attachment_index = None
        for index, entry in enumerate(self.ligand_block.atoms):
            if entry["atomname"] == self.ligand_anchor:
                attachment_index = index
        return attachment_index

    def run(self, core_mol):
        """
        we have two blocks representing the blocks of the NP core and
        the ligand block. Hence, we wish to add links to the blocks
        first add connection between the blocks
        """
        self._set_ff()
        self.np_molecule = self.NP_block.to_molecule()
        ligand_len = len(list(self.ligand_block.atoms))
        # compute the max radius of the
        core_size = len(list(self.NP_block))
        # self.attachment_index = 5
        self._identify_indices_for_core_attachment()  # generate indices we want to find for core attachment
        attachment_index = self._identify_attachment_index()
        # tag on surface of the core to attach with ligands
        core_index_relevant = self.core_indices[
            0
        ]  # first entry of the identified core indices
        # core_index_relevant_II = self.core_indices[
        #    1
        # ]  # second entry of the identified core indices
        attachment_list = []
        # find the indices that are equivalent to the atom
        # on the ligand we wish to attach onto the NP
        # based on the ligands_N, and then merge onto the
        # NP core the main ligand blocks ligand_N number of times

        for index in range(0, self.ligands_N):
            ligand_index = index * ligand_len  # compute starting index for each ligand
            print(ligand_index, core_size)
            scaled_ligand_index = ligand_index + core_size
            attachment_list.append(scaled_ligand_index)
            # merge the relevant molecule
            self.np_molecule.merge_molecule(self.ligand_block)

        # create list of interactions for first batch of indices we have on the
        # core surface
        for index, index2 in enumerate(
            list(core_index_relevant.keys())[: self.ligands_N]
        ):
            interaction = vermouth.molecule.Interaction(
                atoms=(index2, attachment_list[index] + attachment_index),
                parameters=["1", "0.0033", "50000"],
                meta={},
            )
            # append onto bonds the interactions between equivalent indices
            self.np_molecule.interactions["bonds"].append(interaction)

        # create list for interactions for second batch of indices we have on the core surfaces
        # for index, index2 in enumerate(
        #    list(core_index_relevant_II.keys())[: self.ligands_N]
        # ):
        #    interaction = vermouth.molecule.Interaction(
        #        atoms=(index2, atom_max + (index * length) + attachment_index),
        #        parameters=["1", "0.0033", "50000"],
        #        meta={},
        #    )
        #    # append onto bonds the interactions between equivalent indices
        #    self.np_molecule.interactions["bonds"].append(interaction)

        for node in self.np_molecule.nodes:
            # change resname to moltype
            self.np_molecule.nodes[node]["resname"] = "TEST"

        self.graph = MetaMolecule._block_graph_to_res_graph(
            self.np_molecule
        )  # generate residue graph

        # generate meta molecule from the residue graph with a new molecule name
        self.meta_mol = MetaMolecule(
            self.graph, force_field=self.force_field, mol_name="test"
        )

        self.np_molecule.meta["moltype"] = "TEST"
        # reassign the molecule with the np_molecule we have defined new interactions with
        self.meta_mol.molecule = self.np_molecule
        return self.meta_mol

    def create_itp(self, write_path):
        """
        create itp of the final concatenated ligand-functionalized nanoparticle file
        """
        with open(str(write_path), "w") as outfile:
            write_molecule_itp(self.np_molecule, outfile)

    def create_gro(self, write_path):
        """
        create gro file from the system
        """
        self.top.box = np.array([3.0, 3.0, 3.0])
        system = self.top.convert_to_vermouth_system()
        vermouth.gmx.gro.write_gro(
            system,
            write_path,
            precision=7,
            title="these are random positions",
            box=self.top.box,
            defer_writing=False,
        )


def gen_np(
    library,
    out_path,
    inpath,
    core_name,
    ligand_names,
    ligand_anchor,
    pattern="random",
    name="nanoparticle",
    box=np.array([10.0, 10.0, 10]),
):
    """

    function that can create a flexible nanoparticle

    How about we define a function here that is called gen_np analogously
    to the gen_coords or gen_params. This function should have the
    arguments that are tunable to build the NP.

    """
    # 1. load library as described in the second comment
    force_field = load_library(name, library, inpath)
    # 2. select the core block and generate coordinates of the core
    core_mol = force_field.blocks[core_name].to_molecule()
    core_mol = NanoparticleCoordinates().run_molecule(
        core_mol
    )  # not entirely certain what this part does..

    # 3. initiate a class NPAttachLigandsGraphs, that generates only the itp files
    full_np_metamol = NanoparticleGenerator(
        force_field, ligand_names, 1, ligand_anchor, None
    ).run(core_mol)
    # probably a good idea to let this function return a meta_molecule
    # once we have the metamolecule we can instantiate a topology object to use later

    np_top = Topology(name=name, force_field=force_field)
    np_top.molecules = [full_np_metamol]
    # 4. Generate coordinates
    # 5. Write output like you've done before
    # first the itp file
    with open(str(out_path) + "/" + "NP.itp", "w") as outfile:
        write_molecule_itp(np_top.molecules[0].molecule, outfile)

    # then the coordinate file
    # command = " ".join(sys.argv)
    system = np_top.convert_to_vermouth_system()
    NanoparticleLigandCoordinates().run_system(system)

    vermouth.gmx.write_gro(
        system,
        str(out_path) + "/" + "NP.gro",
        precision=7,
        title="new",
        box=box,
        defer_writing=False,
    )


if __name__ == "__main__":
    # example input with gen_np to generate the fullerene/BCM nanoparticle
    gen_np(["oplsaaLigParGen"], "/home/sang/Desktop", None, "C60", "BCM", "C07")

    # previous version of the code - will keep as a reference point
    # for how we called it before

    # building fullerene PBCM complex - need necessary itp paths
    # PATH = "/home/sang/Desktop/git/polyply_1.0/polyply/data/nanoparticle_test/PCBM"
    # NP_itp = PATH + "/" + "C60.itp"
    # BCM_itp = PATH + "/" + "BCM.itp"
    # In this case, as the C60 core is relatively small compared to the ligand size, the
    # system will not work
    # BCM_class = NanoparticleGenerator(NP_itp, BCM_itp, "test", 1, "Janus", "C07")
    # BCM_class.set_NP_ff("C60")
    # BCM_class.set_ligands_ff("BCM")
    # BCM_class._convert_to_vermouth_molecule()
    # BCM_class.create_top()
    # BCM_class.create_itp(PATH + "/" + "NP.itp")
    # BCM_class.create_gro(PATH + "/" + "NP.gro")
