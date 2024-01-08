import itertools
import logging
import math
import io
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, IO

import networkx as nx
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
from pydantic import BaseModel, validator
from scipy.spatial import distance
from vermouth.gmx import gro  # gro library to output the file as a gro file
from vermouth.gmx.itp import write_molecule_itp

# Nanoparticle types
from amber_nps import return_amber_nps_type  # this will be changed
from cg_nps import return_cg_nps_type  # this needs to be changed as well

# logging configuration
# logging.basicConfig(level=#logging.INFO)


class CentralCoreGenerator:
    """
    Generation of the structure of the center core of the NP.
    Initially, we will be using the Fibanocci sphere generator
    to create the artificial sphere.

    Parameters:
    ----------

    Returns:
    -------
    """

    def __init__(
        self,
        filename: str,
        points: int,
        R: int,
        center: int,
        ff: str,
        np_component: str,
        moleculename: str,
    ):
        self.ff = ff
        self.R = R
        self.center = center
        self.output = open(filename + ".itp", "w")
        self.np_component = np_component
        self.moleculename = moleculename

    def _nanoparticle_base_fibonacci_sphere(self, num_points: int = 100) -> None:
        """
        Generate evenly distributed points on a sphere using the Fibonacci lattice algorithm.

        Parameters:
        ----------

        radius (float): The radius of the sphere.
        num_points (int): The number of points to generate.

        Returns:
        --------

        np.ndarray: An array of shape (num_points, 3) containing (x, y, z) coordinates.
        """
        radius = self.R
        points = []
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle in radians

        for i in range(num_points):
            y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
            radius_at_y = math.sqrt(1 - y * y)
            theta = phi * i  # golden angle increment
            x = math.cos(theta) * radius_at_y
            z = math.sin(theta) * radius_at_y
            points.append((x, y, z))

        self.points = np.array(points)

    def _create_np_atom_type_dict(self, mass: float = 1.0, charge: float = 0.0) -> None:
        """
        Store atom information to use with NP core creation
        """
        self._nanoparticle_base_fibonacci_sphere()
        mass = 1.0
        atom_type = self.np_component
        charge = 1.0
        self.atoms = {}
        for index, coordinate in enumerate(self.points):
            self.atoms[index] = (coordinate, mass, atom_type, charge)

    def _distance_array(
        self, specific_point: np.ndarray, points: np.ndarray
    ) -> list[float]:
        """
        compute distance between single point on the NP core with others
        on the NP core
        """
        output = []
        for point in points:
            dist = math.dist(specific_point, point)
            output.append(dist)
        return output

    def _process_atoms(self, default_dist=0.7) -> None:
        """
        Generate bonded distance within the network of contraints to be created for the
        core that is allowed.
        """
        self.np_array = []

        def assign_constraint(
            restraint_unique_list: list[Tuple[list[float], float]],
            restraint_val: float = 5000.0,
        ) -> List[Tuple[int, int, int, float]]:
            """
            generate line for constraint
            """
            output_restraints = []
            for restraints in restraint_unique_list:
                constraint = (
                    restraints[0],
                    restraints[1],
                    1,
                    restraint_val,
                )
                output_restraints.append(constraint)
            return output_restraints

        def check_and_add(sorted_input, bond_distance, np_array):
            """
            record the sorted inputs as tuples and append unique restraint
            entries that we would like to use
            """
            index_distance_input = tuple(sorted_input)
            if index_distance_input not in np_array:
                np_array.append(index_distance_input)

        for np_index, atom in enumerate(self.points):
            # compute the distance between the atom and the rest of the atoms within the core
            dist_array = self._distance_array(atom, self.points)
            for np_index_2, entry in enumerate(dist_array):
                if np_index == np_index_2:
                    continue  # skip if looking at the same index
                if (
                    entry / 10 >= default_dist
                ):  # for martini bonds, we cannot have bonds longer than 0.7 nm
                    continue  # skip if bond length is more than 0.7
                if np_index >= 1:
                    sorted_input = sorted([np_index, np_index_2])
                    # check that we don't have repeated entries for restraints
                    # and once check has passed,
                    check_and_add(sorted_input, entry / 10, self.np_array)

        self.output_constraint = assign_constraint(self.np_array)

    def _write_gro_file(
        self,
        output_filename: str = "output.gro",
        atom_names: str = None,
    ) -> IO:
        """
        generate gro file as the reference input for polyply
        """
        num_atoms = len(self.points)
        header = "Generated by Python\n{:5d}\n".format(num_atoms)

        with open(output_filename, "w") as gro_file:
            gro_file.write(header)
            for i in range(num_atoms):
                atom_name = atom_names[i] if atom_names else f"A{i+1}"
                x, y, z = self.points[i]
                gro_file.write(
                    "{:5d}{:<5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}\n".format(
                        1, "MOL", atom_name, i + 1, x, y, z
                    )
                )
            gro_file.write("   0.00000   0.00000   0.00000\n")

    def _write_itp_file(self) -> IO:
        """
        generate itp file for polyply to read and modify later
        """
        # Write the ITP file
        moleculename = "TEST"
        atom_name = "MOL"
        self._create_np_atom_type_dict()
        self._process_atoms()

        with open("TEST.itp", "w") as itp_file:
            itp_file.write("[ moleculetype ]\n")
            itp_file.write("; Name            nrexcl\n")
            itp_file.write(f"{moleculename}   3\n\n")
            itp_file.write("[ atoms ]\n")
            itp_file.write(
                ";   nr       type  resnr residue  atom   cgnr     charge       mass  typeB    chargeB      massB\n"
            )

            for atom in self.atoms.keys():
                itp_file.write(
                    f"   {atom+1}    {atom_name}    1     {moleculename}    {atom_name}   1    {self.atoms[atom][3]:.6f}    {12.0:.6f}\n"
                )
            itp_file.write("\n")
            # have to write the restraints here
            itp_file.write("[ constraints ]\n")
            for entry in self.output_constraint:
                itp_file.write(f"{entry[0]+1} {entry[1]+1} {entry[2]} {entry[3]}\n")
            itp_file.write("\n")

    def _generate_itp_string(self) -> str:
        """
        Generate an ITP string for polyply to use and modify later.
        """
        # Initialize an empty string to store the ITP content
        itp_content = ""

        # Define the parameters
        moleculename = self.moleculename
        atom_name = "CA"
        self._create_np_atom_type_dict()
        self._process_atoms()

        # Append the content to the itp_content string
        itp_content += "[ moleculetype ]\n"
        itp_content += "; Name            nrexcl\n"
        itp_content += f"{moleculename}   3\n\n"
        itp_content += "[ atoms ]\n"
        itp_content += ";   nr       type  resnr residue  atom   cgnr     charge       mass  typeB    chargeB      massB\n"

        for atom in self.atoms.keys():
            itp_content += f"   {atom+1}    {atom_name}    1     {moleculename}    {atom_name}   1    {self.atoms[atom][3]:.6f}    {12.0:.6f}\n"

        itp_content += "\n[ constraints ]\n"
        for entry in self.output_constraint:
            itp_content += f"{entry[0]+1} {entry[1]+1} {entry[2]} {entry[3]}\n"

        itp_content += "\n"
        return itp_content

    def _generate_gro_string(
        self,
        atom_names: str = None,
    ) -> str:
        """
        Generate GRO file content as a string for use as the reference input for polyply.
        """
        num_atoms = len(self.points)
        header = "Generated by Python\n{:5d}\n".format(num_atoms)

        gro_content = io.StringIO()
        gro_content.write(header)

        for i in range(num_atoms):
            atom_name = atom_names[i] if atom_names else f"A{i+1}"
            x, y, z = self.points[i]
            gro_content.write(
                "{:5d}{:<5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}\n".format(
                    1, "MOL", atom_name, i + 1, x, y, z
                )
            )

        gro_content.write("   0.00000   0.00000   0.00000\n")

        # Get the content of the StringIO object as a string
        gro_string = gro_content.getvalue()

        return gro_string


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
        # #logging.info("defining the nanoparticle and ligand block")
        self.NP_block = self.force_field[self.np_name]
        self.ligand_block = self.force_field.blocks[self.ligand_name]

    def _vertices_edges(self) -> None:
        """
        iniiate np array values
        """
        # #logging.info("defining V E and F")
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
        Generate the gro structure for the artificial NP.

        Parameters
        ----------
        outfile: str
        velocities: bool
        box: str

        Returns
        -------
        None

        """
        # logging.info("Creating the main gromacs file")
        velocity_fmt = ""
        self._generate_positions()  # generate positions for the gold lattice core
        # ##logging.info("put log here")
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


if __name__ == "__main__":
    nanoparticle_sample = CentralCoreGenerator(
        "output.pdb", 100, 3.0, 0.0, "ff", "P5", "TEST"
    )
    nanoparticle_sample._nanoparticle_base_fibonacci_sphere()
    nanoparticle_sample._write_gro_file()
    nanoparticle_sample._write_itp_file()
    gro_string = nanoparticle_sample._generate_gro_string()
    itp_string = nanoparticle_sample._generate_itp_string()
