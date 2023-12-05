import numpy as np
import vermouth
import vermouth.forcefield
import vermouth.molecule
from polyply.src.generate_templates import (
    _expand_inital_coords,
    _relabel_interaction_atoms,
    extract_block,
    find_atoms,
    replace_defined_interaction,
)
from polyply.src.linalg_functions import center_of_geometry

# import polyply.src.meta_molecule
# from polyply.src.load_library import load_library
from polyply.src.meta_molecule import MetaMolecule
from polyply.src.processor import Processor
from polyply.src.topology import Topology

from scipy.spatial import ConvexHull, distance
from vermouth.gmx import gro  # gro library to output the file as a gro file
from vermouth.gmx.itp import write_molecule_itp
from nanoparticle_lattice import (
    NanoparticleModels,
    generate_artificial_core,
    NanoparticleCoordinates,
)
from amber_nps import return_amber_nps_type  # this will be changed


class ArtificialNanoparticleModels(NanoparticleModels):
    """
    Uses inherited methods from the original class to build its own model of nanoparticles
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
        length=4.5,
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
        self.NP_block = self.ff.blocks[self.np_component]
        self.np_molecule_new = self.ff.blocks[self.np_component].to_molecule()
        NanoparticleCoordinates().run_molecule(self.np_molecule_new)
        self.core_len, self.core = len(self.np_molecule_new), len(self.np_molecule_new)
        # Generate the positions for the core and assign as position elements
        NanoparticleCoordinates().run_molecule(self.np_molecule_new)
        self.core_center = center_of_geometry(
            np.asarray(
                list(
                    (nx.get_node_attributes(self.np_molecule_new, "position").values())
                )
            )
        )


if __name__ == "__main__":
    Artificial_martini_model = ArtificialNanoparticleModels(
        "TEST.gro",
        return_amber_nps_type("artificial"),
        "TEST",  # np_component
        "P5",  # np_atype
        ligand_path="/home/sang/Desktop/Papers_NP/Personal_papers/polyply_paper/Martini3-small-molecules/models/itps/cog-mono",  # ligand_path
        ligands=["PHEN_cog.itp", "PHEN_cog.itp"],  # ligands
        ligand_N=[20, 20],  # ligands_N
        pattern="Striped",  # Pattern
        ligand_anchor_atoms=["SN6", "SN6"],  # anchor_atoms
        ligand_tail_atoms=["TC5", "TC5"],  # tail_atoms
        nrexcl=3,  # nrexcl
        ff_name="test",  # ff_test
        original_coordinates={
            "PHEN": gro.read_gro(
                "/home/sang/Desktop/Papers_NP/Personal_papers/polyply_paper/Martini3-small-molecules/models/gros/PHEN.gro"
            ),
            # "1MIMI_cog": "/home/sang/Desktop/Papers_NP/Personal_papers/polyply_paper/Martini3-small-molecules/models/gros/1MIMI.gro",
            "PHEN": gro.read_gro(
                "/home/sang/Desktop/Papers_NP/Personal_papers/polyply_paper/Martini3-small-molecules/models/gros/PHEN.gro"
            ),
        },  # original_coordinates
    )
    Artificial_martini_model._core_generate_artificial_coordinates()
    Artificial_martini_model._identify_indices_for_core_attachment()
    Artificial_martini_model._ligand_generate_coordinates()
    Artificial_martini_model._add_block_indices()
    Artificial_martini_model._generate_ligand_np_interactions()
    Artificial_martini_model._generate_bonds()
    Artificial_martini_model._initiate_nanoparticle_coordinates()
    Artificial_martini_model.create_gro("ART.GRO")
    Artificial_martini_model.write_itp("ART.ITP")
