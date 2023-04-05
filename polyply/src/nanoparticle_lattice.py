import collections
import copy
import itertools
import logging  # implement logging parts here
import random

from pprint import pprint
from typing import Any, Literal
from scipy.spatial import distance

import networkx as nx
import numpy as np
import vermouth
import vermouth.forcefield
import vermouth.molecule
from vermouth.gmx import gro  # gro library to output the file as a gro file
from vermouth.gmx.itp import write_molecule_itp

from pydantic import BaseModel, validator
from polyply.src.generate_templates import (
    _expand_inital_coords,
    _relabel_interaction_atoms,
    find_atoms,
    replace_defined_interaction,
    # map_from_CoG,
    # compute_volume,
    extract_block,
    # GenerateTemplates,
    # find_interaction_involving,
)
import polyply.src.meta_molecule
from polyply.src.load_library import load_library

from polyply.src.meta_molecule import MetaMolecule
from polyply.src.processor import Processor
from polyply.src.topology import Topology
from polyply.src.linalg_functions import center_of_geometry

from amber_nps import return_amber_nps_type  # this will be changed

# logging configuration

logging.basicConfig(level=logging.DEBUG)

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
    positional coordinates and then utilizes networkx to assign the position attribute
    to each atomic node

    Parameters
    ----------
    Returns
    -------
    """

    def run_molecule(self, meta_molecule) -> None:
        # later we can add more complex protocols
        # pos_dict = nx.spring_layout(top.molecules[0].molecule, dim=3)
        init_coords = _expand_inital_coords(meta_molecule)
        # this line adds the position to the molecule
        nx.set_node_attributes(meta_molecule, init_coords, "position")


def return_np(
    first_atom_index_pos: np.ndarray,
    last_atom_index_pos: np.ndarray,
    core_center: np.ndarray,
    core_index: np.ndarray,
    core_coordinates: np.ndarray,
    length: float = 1.0,
):
    """ """

    def rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        Find the rotation matrix that aligns vec1 to vec2
        Args:
        vec1:
            A 3d "source" vector
        vec2:
            A 3d "destination" vector
        Returns:
        rotation_matrix:
            A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        Raises:
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
            vec2 / np.linalg.norm(vec2)
        ).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
        return rotation_matrix

    # get the vector representing the ligand in space
    ligand_alignment_vector = first_atom_index_pos - last_atom_index_pos
    logging.info(f"Calling the rotation matrix from vector")
    transformation_vector = rotation_matrix_from_vectors(vecLigand, vec2)
    unit_vector = np.linalg.norm(LigandAlignmentVector)
    vector_ligand = ligand_alignment_vector.tolist()

    # TransformationVector = rotation_matrix_from_vectors(vecLigand, vec2)

    for i, j in enumerate(Ligand.positions):
        vector = (j - FirstAtomGroup.positions)[0]
        vector[0] = LigandAlignmentVector[0] - vector[0]
        vector[1] = LigandAlignmentVector[1] - vector[1]
        vector[2] = LigandAlignmentVector[2] - vector[2]
        if vector[0] == -math.inf:
            pass
        if vector[0] == 0.0:
            pass
        else:
            TransformationList.append([vector, Ligand.atoms[i].type])

    unitVector = np.linalg.norm(LigandAlignmentVector)
    vecLigand = LigandAlignmentVector.tolist()

    # Loop over the sphere and find the
    for index in range(0, len(SphereList)):
        vec2 = SphereList[index]
        # Find the transformationvector for the ligand vector to vec2, which is the position of the point on sphere
        TransformationVector = rotation_matrix_from_vectors(vecLigand, vec2)
        # Rotate the vector
        vec1_rot = TransformationVector.dot(
            vecLigand
        )  # Rotate the vector to match the surface point on the sphere
        # Get the absolute length of the unit vector
        unitVectorabs = np.linalg.norm(LigandAlignmentVector)
        # Change the rotation vector in unit vector, then multiply by the absolute
        # length of the sphere
        vecMultiplier = vec1_rot / unitVectorabs * (np.linalg.norm(np.array(vec2))) + (
            vec1_rot / unitVectorabs * Length
        )
        # Find the difference in length

        Sphere.append(vec2)

        # Translate the vector further out
        for trans in TransformationList:
            LigandAtomcoordinate = TransformationVector.dot(trans[0])
            LigandAtomcoordinate[0] = LigandAtomcoordinate[0] + vecMultiplier[0]
            LigandAtomcoordinate[1] = LigandAtomcoordinate[1] + vecMultiplier[1]
            LigandAtomcoordinate[2] = LigandAtomcoordinate[2] + vecMultiplier[2]
            LigandList.append(
                LigandAtomcoordinate.tolist()
            )  # Append coordinates of the
            NameList.append(trans[1])  # Append the names of the atoms


class PositionChange(Processor):
    """
    Parameters
    ----------

    Returns
    -------
    """

    def __init__(self, ligand_block_specs, core_len, *args, **kwargs):
        self.ligand_block_specs = ligand_block_specs
        self.core_len = core_len
        super().__init__(*args, **kwargs)

    def run_molecule(self, meta_molecule):
        # first we find a starting node by looping over the molecule nodes
        # finding any one with a degree that is larger than 1
        # shift coordinates - not sure how to fully do this as we have a generator
        ligands = list(self.ligand_block_specs.keys())
        shift_array_I = [
            self.ligand_block_specs[ligands[0]]["indices"][key]
            for key in list(self.ligand_block_specs[ligands[0]]["indices"].keys())
        ]

        shift_array_II = [
            self.ligand_block_specs[ligands[1]]["indices"][key]
            for key in list(self.ligand_block_specs[ligands[1]]["indices"].keys())
        ]
        unique_res = []
        for index, node in enumerate(list(meta_molecule.nodes)):
            unique_res.append(meta_molecule.nodes[node]["resid"])
        unique_res = list(set(unique_res))

        for resid in unique_res:
            shift_value = np.array(
                [
                    random.uniform(-2.0, 2.0),
                    random.uniform(-2.0, 2.0),
                    random.uniform(-2.0, 2.0),
                ]
            )
            for index, node in enumerate(list(meta_molecule.nodes)):
                if meta_molecule.nodes[node]["resid"] == resid:
                    meta_molecule.nodes[node]["position"] += shift_value
        #
        # for index, node in enumerate(
        #    list(meta_molecule.nodes)[
        #        self.core_len : self.core_len + len(shift_array_I) + 1
        #    ]
        # ):
        #    print(index, meta_molecule.nodes[node], node)
        #    shift_value = np.array([10, 10, 10])  # shift_array_I[index]
        #    meta_molecule.nodes[node]["position"] += shift_value
        #
        # for index, node in enumerate(
        #    list(meta_molecule.nodes)[len(shift_array_I) : len(shift_array_II) + 1]
        # ):
        #    print(index, meta_molecule.nodes[node], node)
        #    shift_value = np.array([10, 10, 10])
        #    # shift_value = shift_array_II[index]
        #    meta_molecule.nodes[node]["position"] += shift_value


class GoldNanoparticleSingle:
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

    def __init__(self):
        # orientation: Any
        self.lattice_constant: float = 4.0782
        self.box_size: int = 100
        self.matrix: Any = np.array(
            [[self.box_size, 0, 0], [0, self.box_size, 0], [0, 0, self.box_size]]
        )
        self.n_atom_thiol: int = 9
        self.n_phi: int = 8
        self.n_theta: int = 17
        self.n_thiol: int = self.n_phi * self.n_theta
        self.gold_layer: int = 6
        self.n_gold: float = (
            (10 * (self.gold_layer**3) + 11 * self.gold_layer) / 3
            - 5 * (self.gold_layer**2)
            - 1
        )
        # matrix store gold atom coordinates
        self.pos_gold: Any = np.zeros((int(self.n_gold), 3))  # TODO
        self.center: list[float] = [0.5, 0.5, 0.5]  # TODO
        self.phi: float = (np.sqrt(5) + 1) / 2
        self.V: Any  # this should be changed from any to something more appropriate
        self.E: Any  # ditto
        self.F: Any  # ditto
        # blocks and nanoparticle specific parts to work with polyply
        self.force_field: Any  # this needs a custom type
        self.top: Any  # this needs a custom typey
        self.NP_block: Any

    def _set_ff(self) -> None:
        """ """
        logging.info("defining the nanoparticle and ligand block")
        self.NP_block = self.force_field[self.np_name]
        self.ligand_block = self.force_field.blocks[self.ligand_name]

    def _vertices_edges(self) -> None:
        """
        iniiate np array values
        """
        logging.info("defining V E and F")
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
        self, outfile: str, velocities: bool = False, box: str = "5.0, 5.0, 5.0"
    ) -> None:
        """
        Plot out the lattice in a 3D structure - generating the coordinates part
        is easy. Now
        the hard part is generating the gro file
        this is somewhat working ... nowhere near what I want - need to review the core generation code I made above

        Parameters
        ----------
        outfile: str
        velocities: bool
        box: str

        Returns
        -------
        None

        """
        logging.info("Creating the main gromacs file")
        velocity_fmt = ""
        self._generate_positions()  # generate positions for the gold lattice core
        logging.info("put log here")
        core_numpy_coords = self.pos_gold
        coords = [[j for j in i] for i in core_numpy_coords]
        gold_str = [f"1AU  AU  {i + 1}" for i in range(0, len(coords))]
        # just need to figure out GRO CONTENT and COORINDATES
        velocities = [[x + 1 for x in line] for line in coords]
        with open(str(outfile), "w") as output:
            output.write("NP\n")
            output.write(str(len(coords)) + "\n")
            for atom, coords, vels in zip(gold_str, coords, velocities):
                output.write(
                    ("{}{:8.3f}{:8.3f}{:8.3f}" + velocity_fmt + "\n").format(
                        atom, *itertools.chain(coords, vels)
                    )
                )

            output.write(box)
            output.write("\n")


class gold_models(Processor):
    """
    We have a number of gold nanoparticle cores based on the amber forcefield
    that is avaliable as published here:

    https://pubs.acs.org/doi/abs/10.1021/acs.jctc.5b01053

    analyzing and creating the class depending on the
    number of atoms of gold we want

    Parameters
    ----------
    outfile: str
    velocities: bool
    box: str

    Returns
    -------
    None
    """

    def __init__(self):
        """
        Main class for making the gold nanoparticles.

        Parameters
        ----------
        sample:  str
        ligand_path:  str
        ligand_N: List[int]
        pattern: List[str]
        ligand_anchor_atoms:

        """
        self.sample: str = return_amber_nps_type(
            "au144_OPLS_bonded"
        )  # call the appropriate amber code type
        self.ligand_path: str = "/home/sang/Desktop/git/polyply_1.0/polyply/tests/test_data/np_test_files/AMBER_AU/ligand"
        self.ligands = ["UNK_12B037/UNK_12B037.itp", "UNK_DA2640/UNK_DA2640.itp"]
        self.ligand_N = [10, 10]
        self.pattern: Literal["Janus", "Striped"] = "Striped"
        self.ligand_anchor_atoms: list[str] = ["S00", "S07"]
        self.ff = vermouth.forcefield.ForceField(name="test")

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
        if not self.ff:
            raise ValueError("Need to initiate force field first!")

        nodes = find_atoms(molecule, "atype", atomname)
        resid = molecule.nodes[nodes[0]]["resid"]
        block = vermouth.molecule.Block(force_field=self.ff)
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
        None: None

        Returns
        -------
        :None
        """
        vermouth.gmx.itp_read.read_itp(self.sample, self.ff)
        self.NP_block = self.ff.blocks["NP2"]
        core_molecule = self.ff.blocks["NP2"].to_molecule()  # make metamolcule object
        core_block = self._extract_block_atom(
            core_molecule, "AU", {}
        )  # get only the gold atoms from the itp
        self.core_block = core_block
        self.core_len, self.core = len(core_block), len(core_block)

        core_block.nrexcl = 3
        self.np_molecule_new = core_block.to_molecule()

    def _identify_lattice_surface(self) -> None:
        """
        Identify atoms that we know are bonded to sulfur anchors,
        so that we can use the indices for replacing and redesigning

        Parameters
        ----------
        None: None

        Returns
        -------

        """
        assert "NP2" in self.ff.blocks.keys(), "we have not stored NP information"
        surface_atom_dict = {}
        # create dictionary
        for atom in self.ff.blocks["NP2"].atoms:
            surface_atom_dict[atom["index"]] = atom["atype"]

        # find the AU atoms that have a bond with the S atoms
        self.surface_atoms = []
        for interaction_node in self.ff.blocks["NP2"].interactions["bonds"]:
            interaction = [
                surface_atom_dict[
                    interaction_node.atoms[0] + 1
                ],  # add one to ensure we have a non zero index start
                surface_atom_dict[interaction_node.atoms[1] + 1],
            ]
            if "S" in interaction and "AU" in interaction:
                self.surface_atoms.append(interaction_node.atoms[1] + 1)

        # get only the unique atoms
        # self.surface_atoms = list(set(self.surface_atoms))
        self.surface_atoms = [1]

    def _identify_indices_for_core_attachment(self):
        """
        based on the coordinates of the NP core provided,
        this function could just return a list - not sure it has to be a class
        attribute

        Parameters
        ----------
        None: None

        Returns
        -------

        """
        # assign molecule object

        # init_coords = expand_initial_coords(self.NP_block)
        NanoparticleCoordinates().run_molecule(self.np_molecule_new)
        # In the case of a spherical core, find the radius of the core
        keys = [atom[0] for atom in self.np_molecule_new.atoms]
        core_numpy_coords = np.asarray(
            list((nx.get_node_attributes(self.np_molecule_new, "position").values()))
        )
        logging.info(
            f"The filtered NP core has {len(core_numpy_coords)} number of atoms"
        )
        # Find the center of geometry of the core
        center_np = center_of_geometry(core_numpy_coords)
        # Compute the radius
        radius = distance.euclidean(center_np, core_numpy_coords[0])
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

        # return the Striped pattern
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
        # return the Janus pattern
        elif self.pattern == "Janus":
            core_top_values = {}
            core_bot_values = {}
            threshold = length / 2  # divide nanoparticle region into 2
            for index, entry in enumerate(core_numpy_coords):
                if entry[2] > minimum_threshold + threshold:
                    core_top_values[index] = entry
                else:
                    core_bot_values[index] = entry

            self.core_indices: list[dict[int, int]] = [core_top_values, core_bot_values]

        # identify the surface gold atoms
        self._identify_lattice_surface()
        # filter out from the Janus and non-Janus to
        for core_index in range(0, len(self.core_indices)):
            self.core_indices[core_index] = {
                x: self.core_indices[core_index][x]
                for x in self.surface_atoms
                if x in list(self.core_indices[core_index])
            }

    def _identify_attachment_index(self, ligand_block, anchor_atom):
        """
        Find index of atoms that corresponds to the atom on the ligand
        which you wish to bond with the ligand on the nanoparticle core

        Parameters
        ----------
        None: None

        Returns
        -------

        """
        attachment_index = None
        for index, entry in enumerate(ligand_block.atoms):
            if entry["atomname"] == anchor_atom:
                attachment_index = index
        return attachment_index

    def _ligand_generate_coordinates(self) -> None:
        """
        Read the ligand itp files to the force field and ensure we can read the blocks

        Parameters
        ----------
        None: None

        Returns
        -------

        """
        # if any(
        #    x > min([len(entry) for entry in self.core_indices]) for x in self.ligand_N
        # ):
        #    raise ValueError(f"{x} should not be larger than and of the core indices!")

        # ensure we have a ff initialized
        self.ligand_block_specs = {}
        for ligand in self.ligands:
            with open(self.ligand_path + "/" + ligand, "r") as f:
                ligand_file = f.read()
                ligand_file = ligand_file.split("\n")
                # register the itp file
                vermouth.gmx.itp_read.read_itp(ligand_file, self.ff)

                # assert len(self.ff.blocks.keys()) == len(self.ligand_N)
        for index, block_name in enumerate(self.ff.blocks.keys()):
            if block_name != "NP2":
                NanoparticleCoordinates().run_molecule(self.ff.blocks[block_name])
                self.ligand_block_specs[block_name] = {
                    "name": block_name,
                    "length": len(
                        list(self.ff.blocks[block_name].atoms)
                    ),  # store length of the ligand
                    "N": self.ligand_N[index - 1],  # store number of ligands - TODO
                    "anchor_index": self._identify_attachment_index(
                        self.ff.blocks[block_name],
                        self.ligand_anchor_atoms[index - 1],
                    ),  # store the nth atom of the ligand that correponds to anchor
                    "indices": self.core_indices[index - 1],  # store the
                }

    def _add_block_indices(self) -> None:
        """
        Parameters
        ----------
        None: None

        Returns
        -------
        """
        core_size = len(list(self.NP_block))
        scaling_factor = 0
        for key in self.ligand_block_specs.keys():
            self.ligand_block_specs[key]["scaled_ligand_index"] = [
                (index * self.ligand_block_specs[key]["length"] + core_size)
                + scaling_factor
                for index in range(0, self.ligand_block_specs[key]["N"])
            ]
            scaling_factor += (
                self.ligand_block_specs[key]["N"]
                * self.ligand_block_specs[key]["length"]
            )

    def _generate_ligand_np_interactions(self) -> None:
        """
        Parameters
        ----------
        None: None

        Returns
        -------

        """
        core_size = self.core
        for key in self.ligand_block_specs.keys():
            attachment_list = {}
            for index in range(
                0, len(list(self.ligand_block_specs[key]["indices"].keys())) + 1
            ):  # loop over the number of ligands we want to create
                self.np_molecule_new.merge_molecule(self.ff.blocks[key])
                attachment_list[index] = [
                    core_size + 1,
                    core_size + 1 + self.ligand_block_specs[key]["length"],
                ]
                core_size += self.ligand_block_specs[key]["length"]
            self.ligand_block_specs[key]["shift_index"] = attachment_list

    def _generate_bonds(self) -> None:
        """


        Parameters
        ----------
        None: None

        Returns
        -------
        """
        # get random N elements from the list
        for key in self.ligand_block_specs.keys():
            logging.info(f"bonds for {key}")
            for index, entry in enumerate(
                list(self.ligand_block_specs[key]["indices"].keys())
            ):
                base_anchor = (
                    self.core_len
                    + ((index + 1) * self.ligand_block_specs[key]["length"])
                    + self.ligand_block_specs[key]["anchor_index"]
                )
                interaction = vermouth.molecule.Interaction(
                    atoms=(
                        entry,
                        base_anchor,
                    ),
                    parameters=["1", "0.0033", "5000"],
                    meta={},
                )
                logging.info(f"generating bonds between {entry} and {base_anchor}")
                self.np_molecule_new.interactions["bonds"].append(interaction)

            logging.info(
                f"core length is {self.core_len}, number of atoms with ligands added is {self.ligand_block_specs[key]['length'] * self.ligand_block_specs[key]['N']} "
            )

            self.core_len += self.ligand_block_specs[key]["length"] * len(
                list(self.ligand_block_specs[key]["indices"])
            )
            logging.info(f"core length is now {self.core_len}")

    def _run(self):
        """ """
        for node in self.np_molecule_new.nodes:
            # change resname to moltype
            self.np_molecule_new.nodes[node]["resname"] = "TEST"
        self.graph = MetaMolecule._block_graph_to_res_graph(
            self.np_molecule_new
        )  # generate residue graph
        # generate meta molecule fro the residue graph with a new molecule name
        self.meta_mol = MetaMolecule(self.graph, force_field=self.ff, mol_name="sdfs")
        self.np_molecule_new.meta["moltype"] = "TEST"
        # reassign the molecule with the np_molecule we have defined new interactions with
        NanoparticleCoordinates().run_molecule(self.np_molecule_new)
        # shift the positions of the ligands so that they are initiated on the surface of the NP
        PositionChange(
            core_len=self.core_len, ligand_block_specs=self.ligand_block_specs
        ).run_molecule(self.np_molecule_new)
        # prepare meta molecule
        self.meta_mol.molecule = self.np_molecule_new
        # set the topology object
        self.np_top = Topology(name="nanoparticle", force_field=self.ff)
        self.np_top.molecules = [self.meta_mol]
        # assign size of the box
        self.np_top.box = np.array([10.0, 10.0, 10.0])

    def generate_dicts(self) -> None:
        """
        TODO
        """
        self._identify_indices_for_core_attachment()
        self._ligand_generate_coordinates()
        self._add_block_indices()
        self._generate_ligand_np_interactions()
        self._generate_bonds()
        self._run()

    def create_gro(self, write_path: str) -> None:
        """
        ideally, we generate the coordinates with this function
        and then store it within a 'coordinates' object. The form
        of which I am not certain yet.

        Parameters
        ----------
        write_path : str

        Returns
        -------
        None

        """
        system = self.np_top.convert_to_vermouth_system()
        gro.write_gro(
            system,
            write_path,
            precision=7,
            title="gold nanoparticle core",
            box=self.np_top.box,
            defer_writing=False,
        )

    def write_itp(self) -> None:
        """
        Parameters
        ----------
        None: None

        Returns
        -------
        None
        """
        self.np_top = Topology(name="nanoparticle", force_field=self.ff)
        self.np_top.molecules = [self.meta_mol]
        self.np_top.box = np.array([30.0, 30.0, 30.0])
        with open("np.itp", "w") as outfile:
            write_molecule_itp(self.np_top.molecules[0].molecule, outfile)


# main code executable
if __name__ == "__main__":
    # sampleNPCore = GoldNanoparticleSingle()
    # sampleNPCore._generate_positions()
    # sampleNPCore.create_gro(
    #    "/home/sang/Desktop/example_gold.gro"
    # )  # this works but not fully

    # generate the core of the opls force field work
    gold_model = gold_models()
    gold_model.core_generate_coordinates()
    gold_model.generate_dicts()
    gold_model.create_gro("new.gro")
    gold_model.write_itp()
