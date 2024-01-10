"""
You can make the code cleaner and more organized by following some best
practises in python coding. Here are some suggestions to improve the code


1. Use meaningful variable names: Choose variable names that clearly describe their
   purpose. This makes the code more readable and self-explanatory

2. Add docstrings and comments: Include docstrings for functions and comments for complex
   sections of code to explain their purpose and functionality

3. Organize imports: Import statements should be at the top of your file, and group them
   based on their source (standard library, third-party libraries, and local modules)

4. Avoid redundant code: There are some lines of code that seem to be repeated. Ensure you avoid
   avoid redundancy by eliminating duplicate code.

5. Follow pep 8 conventions, adhere to pep 8 style guide for python code to maintain
   consistency and readability

"""
import numpy as np
import networkx as nx
from scipy.spatial import ConvexHull, distance

import vermouth
import vermouth.forcefield
import vermouth.molecule
from vermouth.gmx import gro  # gro library to output the file as a gro file
from vermouth.gmx.itp import write_molecule_itp

from polyply.src.generate_templates import (
    _expand_inital_coords,
    _relabel_interaction_atoms,
    extract_block,
    find_atoms,
    replace_defined_interaction,
)
from polyply.src.linalg_functions import center_of_geometry
from polyply.src.meta_molecule import MetaMolecule
from polyply.src.processor import Processor
from polyply.src.topology import Topology

from nanoparticle_lattice import (
    NanoparticleModels,
    generate_artificial_core,
    NanoparticleCoordinates,
    PositionChangeCore,
    PositionChangeLigand,
)
from amber_nps import return_amber_nps_type  # this will be changed


class ArtificialNanoparticleModels(NanoparticleModels):
    """
    Uses inherited methods from the original class to build its own model of nanoparticles

    We have defined a new _initiate_nanoparticle_coordinates to overwrite the method
    that has already been defined in the inherited method
    """

    def __init__(
        self,
        gro_file,
        sample,
        np_component,
        np_atype,
        ligand_path,
        ligands,
        ligand_N,
        pattern,
        ligand_anchor_atoms,
        ligand_tail_atoms,
        nrexcl,
        ff_name="test",
        length=1.5,
        original_coordinates=None,
        identify_surface=False,
        core_option=None,
    ):
        self.gro_file = gro_file
        self.sample = sample
        self.np_component = np_component
        self.np_atype = np_atype
        self.ligand_path = ligand_path
        self.ligands = ligands
        self.ligand_N = ligand_N
        self.pattern = pattern
        self.ligand_anchor_atoms = ligand_anchor_atoms
        self.ligand_tail_atoms = ligand_tail_atoms
        self.nrexcl = nrexcl
        self.ff = vermouth.forcefield.ForceField(name=ff_name)
        self.length = length
        self.identify_surface = identify_surface
        self.original_coordinates = original_coordinates
        self.core_option = core_option

    def _core_generate_artificial_coordinates(self) -> None:
        """
        internal method for generating the artificial core and registering the
        itp with in the force field
        """
        generate_artificial_core("output.gro", 100, 3.0, self.ff, self.np_atype)
        self.np_block = self.ff.blocks[self.np_component]
        self.np_molecule_new = self.ff.blocks[self.np_component].to_molecule()
        self.core_len, self.core = len(self.np_molecule_new), len(self.np_molecule_new)
        NanoparticleCoordinates().run_molecule(self.np_molecule_new)
        self.core_center = center_of_geometry(
            np.asarray(
                list(
                    (nx.get_node_attributes(self.np_molecule_new, "position").values())
                )
            )
        )

    def _initiate_nanoparticle_coordinates(self) -> None:
        """
        Initialize nanoparticle coordinates
        """
        self.graph = MetaMolecule._block_graph_to_res_graph(
            self.np_molecule_new
        )  # generate residue graph
        # generate meta molecule fro the residue graph with a new molecule name
        self.meta_mol = MetaMolecule(
            self.graph, force_field=self.ff, mol_name="random_name"
        )
        self.np_molecule_new.meta["moltype"] = "TEST"

        # reassign the molecule with the np_molecule we have defined new interactions with
        NanoparticleCoordinates().run_molecule(self.np_molecule_new)
        PositionChangeCore(self.gro_file, "MOL", self.np_atype).run_molecule(
            self.np_molecule_new
        )
        for resname in self.ligand_block_specs.keys():
            PositionChangeLigand(
                ligand_block_specs=self.ligand_block_specs,
                core_len=self.core_len,
                original_coordinates=self.original_coordinates,
                length=self.length,
                core_center=self.core_center,
                resname=resname,
            ).run_molecule(self.np_molecule_new)
        # prepare meta molecule
        self.meta_mol.molecule = self.np_molecule_new


if __name__ == "__main__":
    artificial_martini_model = ArtificialNanoparticleModels(
        "TEST.gro",
        return_amber_nps_type("artificial"),
        "TEST",  # np_component
        "CA",  # np_atype
        ligand_path="/home/sang/Desktop/Papers_NP/Personal_papers/polyply_paper/Martini3-small-molecules/models/itps/cog-mono",  # ligand_path
        ligands=["PHEN_cog.itp", "PHEN_cog.itp"],  # ligands
        ligand_N=[20, 20],  # ligands_N
        pattern="Janus",  # Pattern
        ligand_anchor_atoms=["SN6", "SN6"],  # anchor_atoms
        ligand_tail_atoms=["TC5", "TC5"],  # tail_atoms
        nrexcl=3,  # nrexcl
        ff_name="test",  # ff_test
        original_coordinates={
            "PHEN": gro.read_gro(
                "/home/sang/Desktop/Papers_NP/Personal_papers/polyply_paper/Martini3-small-molecules/models/gros/PHEN.gro"
            ),
            "PHEN": gro.read_gro(
                "/home/sang/Desktop/Papers_NP/Personal_papers/polyply_paper/Martini3-small-molecules/models/gros/PHEN.gro"
            ),
            # "1MIMI_cog": "/home/sang/Desktop/Papers_NP/Personal_papers/polyply_paper/Martini3-small-molecules/models/gros/1MIMI.gro",
            # "PHEN": gro.read_gro(
            #    "/home/sang/Desktop/Papers_NP/Personal_papers/polyply_paper/Martini3-small-molecules/models/gros/PHEN.gro"
            # ),
        },  # original_coordinates
    )
    artificial_martini_model._core_generate_artificial_coordinates()
    artificial_martini_model._identify_indices_for_core_attachment()
    artificial_martini_model._ligand_generate_coordinates()
    artificial_martini_model._add_block_indices()
    artificial_martini_model._generate_ligand_np_interactions()
    artificial_martini_model._generate_bonds()
    artificial_martini_model._initiate_nanoparticle_coordinates()
    artificial_martini_model.create_gro("ART.gro")
    artificial_martini_model.write_itp("ART.itp")
