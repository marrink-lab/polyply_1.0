import io
import itertools
import logging  # implement logging parts here
from typing import Any, Dict, List, Literal, Tuple

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

from polyply.src.meta_molecule import MetaMolecule
from polyply.src.processor import Processor
from polyply.src.topology import Topology
from scipy.spatial import ConvexHull, distance
from vermouth.gmx import gro  # gro library to output the file as a gro file
from vermouth.gmx.itp import write_molecule_itp

from amber_nps import return_amber_nps_type  # this will be changed
from cg_nps import return_cg_nps_type  # this needs to be changed as well
from nanoparticle_generic import CentralCoreGenerator

# logging configuration
logging.basicConfig(level=logging.INFO)


def generate_artificial_core(
    output: str,
    number_of_atoms: int,
    radius: float,
    ff: vermouth.forcefield.ForceField,
    constitute: str,
    nanoparticle_name: str = "TEST",
) -> None:  # Tuple[str, str]:
    """
    use the imported class to generate a core of R size
    and write it to the main vermouth object that has been called.

    At the moment, the generation of gromacs file will need to be separately implemented by reading in the
    gromacs file that is generated from the nanoparticle_lattice.
    """
    nanoparticle_core_object = CentralCoreGenerator(
        output, number_of_atoms, radius, 0.0, ff, constitute, nanoparticle_name
    )
    nanoparticle_core_object._nanoparticle_base_fibonacci_sphere()
    logging.info("Writing the itp for the artificial core")
    itp_output = nanoparticle_core_object._generate_itp_string().split("\n")
    vermouth.gmx.itp_read.read_itp(itp_output, ff)
    logging.info(
        f"Writing the gro file for the artificial core - generating {nanoparticle_name}.gro"
    )
    nanoparticle_core_object._write_gro_file(output_filename=f"{nanoparticle_name}.gro")
    nanoparticle_core_object._write_itp_file()


def rotation_matrix_from_vectors(
    vec_a: np.ndarray, vec_b: np.ndarray, direction: str = "ccw"
) -> np.ndarray:
    """
    Find the rotation matrix that aligns vec_a to vec_b
    Args:
    vec_a:
        A 3D "source" vector
    vec_b:
        A 3D "destination" vector
    direction:
        'cw' (clockwise) or 'ccw' (counterclockwise) to specify the rotation direction
    Returns:
    rotation_matrix:
        A transformation matrix (3x3) which when applied to vec_a, aligns it with vec_b.
    """
    a, b = (vec_a / np.linalg.norm(vec_a)).reshape(3), (
        vec_b / np.linalg.norm(vec_b)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    if direction == "cw":
        rotation_matrix = np.eye(3) - kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    elif direction == "ccw":
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    else:
        raise ValueError("Invalid rotation direction. Use 'cw' or 'ccw'.")

    return rotation_matrix


def create_np_pattern(
    pattern: Literal[None, "Striped", "Janus"],
    core_numpy_coords: np.ndarray,
    length: int,
    minimum_threshold: float,
    maximum_threshold: float,
) -> List[np.ndarray]:
    """
    Output pattern for decorating the nanoparticle core.
    """
    # identify only the surface atoms
    if pattern == None:  # when we have no patterns on surfaces
        core_values = {}
        for index, entry in enumerate(core_numpy_coords):
            core_values[index] = entry
        core_indices = [core_values]

    elif pattern == "Striped":  # when we have a striped patterns
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
        core_indices = [core_striped_values, core_ceiling_values]

    elif pattern == "Janus":  # when we have a janus patterns
        core_top_values = {}
        core_bot_values = {}
        threshold = length / 2  # divide nanoparticle region into 2
        for index, entry in enumerate(core_numpy_coords):
            if entry[2] > minimum_threshold + threshold:
                core_top_values[index] = entry
            else:
                core_bot_values[index] = entry

        core_indices = [core_top_values, core_bot_values]

    return core_indices


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


class PositionChangeCore(Processor):
    """
    Reestablishing the core positions from the
    gro file
    """

    def __init__(self, gro_path: str, atom: str, atom_type: str):
        self.gro_path = gro_path
        self.np_core_atom = atom
        self.np_core_atom_type = atom_type

    def run_molecule(self, meta_molecule) -> None:
        """
        Adjust the core components as necessary
        """
        # Load the GROMACS (gro) file
        gro_loaded = gro.read_gro(self.gro_path)
        # Extract core incremental positions from the loaded GROMACS file
        core_incremental_positions = list(gro_loaded.nodes.values())
        # Extract original atomic coordinates for the specified 'resname'
        original_atomic_coordinates = [
            atom[1]["position"]
            for atom in gro_loaded.atoms
            if atom[1]["resname"] == self.np_core_atom
        ]

        # Update 'position' in 'meta_molecule' based on 'original_atomic_coordinates'
        for index, node in enumerate(
            list(meta_molecule.nodes)[: len(original_atomic_coordinates)]
        ):
            if meta_molecule.nodes[index + 1]["atype"] == self.np_core_atom_type:
                meta_molecule.nodes[index + 1][
                    "position"
                ] = original_atomic_coordinates[index]


class PositionChangeLigand(Processor):
    """
    Looks through the meta molecule assigned, and finds the core
    atom to which the ligand is attached. One this has been established,
    we modify the coordinates of the ligands to ensure that

    Parameters
    ----------

    Returns
    -------
    """

    def __init__(
        self,
        ligand_block_specs: Dict[str, Any],
        core_len: int,
        resname: str,
        original_coordinates: Dict[str, str],
        length: float,
        core_center: np.ndarray,
        option: str = "ligand",
        *args,
        **kwargs,
    ):
        self.ligand_block_specs = ligand_block_specs
        self.core_len = core_len
        self.resname = resname
        self.original_coordinates = original_coordinates
        self.length = length
        self.option = option
        self.core_center = core_center
        super().__init__(*args, **kwargs)

    def run_molecule(self, meta_molecule) -> None:
        # first we find a starting node by looping over the molecule nodes
        # finding any one with a degree that is larger than 1
        # shift coordinates - not sure how to fully do this as we have a generator

        rotation_matrix_dictionary = {}
        absolute_vectors = {}
        ligand_head_tail_pos = {}

        # Get the ligand alignment vectors for the gro file, to readjust the ligand coordinates
        # once we rotate it properly around the nanoparticle core
        ligand_positions = self.original_coordinates[self.resname]

        ligand_incremental_positions = [
            ligand_positions.nodes[node] for node in list(ligand_positions.nodes)
        ]

        zeroth_position = ligand_incremental_positions[0]["position"]

        # Compute the ligand representing the direction pointing from the
        # center of the nanoparticle core to the core atom where the
        # ligand is attached.

        core_ligand = [
            self.ligand_block_specs[self.resname]["indices"][key]
            for key in list(self.ligand_block_specs[self.resname]["indices"].keys())[
                : self.ligand_block_specs[self.resname]["N"]
            ]
        ]

        # reassign the coordinates to match the original gromacs coordinates
        for resid_index, resid in enumerate(
            self.ligand_block_specs[self.resname]["resids"]
        ):
            # For each of the nodes.. do something? Need to get to the bottom of this
            for index, node in enumerate(list(meta_molecule.nodes)):
                if meta_molecule.nodes[node]["resid"] == resid:
                    # According to index, change the value
                    for pos_index, lig in enumerate(ligand_incremental_positions):
                        if meta_molecule.nodes[node]["atomname"] == lig["atomname"]:
                            meta_molecule.nodes[node]["position"] = np.array(
                                lig["position"]
                            ) - np.array(zeroth_position)

        # Loop over the residue ids for the ligands
        for resid_index, resid in enumerate(
            self.ligand_block_specs[self.resname]["resids"]
        ):
            lig_vec = self.ligand_block_specs[self.resname]["indices"][
                list(self.ligand_block_specs[self.resname]["indices"].keys())[
                    resid_index
                ]
            ]
            # Assign ligand_head tail positional list to store the new position of
            # ligand that has its coordinates changed as with above
            ligand_head_tail_pos[resid] = []
            for index, node in enumerate(list(meta_molecule.nodes)):
                if meta_molecule.nodes[node]["resid"] == resid and (
                    meta_molecule.nodes[node]["index"]
                    == self.ligand_block_specs[self.resname]["anchor_index"] + 1
                    or meta_molecule.nodes[node]["index"]
                    == self.ligand_block_specs[self.resname]["tail_index"] + 1
                ):
                    node_position = meta_molecule.nodes[node]["position"]
                    ligand_head_tail_pos[resid].append(node_position)

            # Get the ligand directional vector
            ligand_directional_vector = (
                ligand_head_tail_pos[resid][1] - ligand_head_tail_pos[resid][0]
            )

            rot_mat = rotation_matrix_from_vectors(
                ligand_directional_vector, self.core_center - lig_vec
            )
            logging.info(
                f"The rotation matrix we have for residue {resid} is {rot_mat}"
            )
            # store the transformational matrix for that particular ligand
            rotation_matrix_dictionary[resid] = rot_mat

            absolute_vectors[resid] = [
                np.linalg.norm(ligand_directional_vector),
                np.linalg.norm(core_ligand[resid_index]),
            ]

        # Looping over the ligand resids

        for resid_index, resid in enumerate(
            self.ligand_block_specs[self.resname]["resids"]
        ):
            lig_vec = self.ligand_block_specs[self.resname]["indices"][
                list(self.ligand_block_specs[self.resname]["indices"].keys())[
                    resid_index
                ]
            ]

            for index, node in enumerate(list(meta_molecule.nodes)):
                # Pick out the residue ids for the ligand of interest
                if meta_molecule.nodes[node]["resid"] == resid:
                    rotated_value = rotation_matrix_dictionary[resid].dot(
                        meta_molecule.nodes[node]["position"]
                    )
                    logging.info(
                        f"Newly rotated ligand is {rotated_value} for residue {resid}"
                    )

                    modifier = (
                        rotated_value
                        / absolute_vectors[resid][0]
                        * np.linalg.norm(lig_vec)
                        + rotation_matrix_dictionary[resid].dot(
                            ligand_directional_vector
                        )
                        / absolute_vectors[resid][0]
                        * self.length
                    )
                    logging.info("Need to put in some logging notifiers here")

                    # Modify x, y, and z cartesian coordinates
                    for i in range(3):
                        meta_molecule.nodes[node]["position"][i] = (
                            rotated_value[i] + modifier[i]
                        )


class NanoparticleModels(Processor):
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
        """
        Main class for making the ligand-functionalized nanoparticles.

        Parameters
        ----------
        sample:  str
        ligand_path:  str
        ligand_N: List[int]
        pattern: List[str]
        ligand_anchor_atoms:

        """
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
        Now I need to ensure that this function is
        using vermouth to strip the original amber nanoparticle itps
        and read in as residue that can be read

        Only extract the core atoms we care about. Here, we may have a gold
        nanoparticle with ligands attached where the atom type for the core
        may just be Au, so here we need to filter out based on that criteria. If
        not, then do not need to worry about the ligand atom types that may be
        potentially attached onto the nanoparticle.

        Parameters
        ----------
        None: None

        Returns
        -------
        :None
        """

        if self.core_option == "Artificial":
            # generate the fibanocci core - reading in the itp and writing the output.gro to be used later
            self.core_generate_artificial_coordinates()
        else:  # otherwise, we are reading in an exsiti gncore and strippign it
            vermouth.gmx.itp_read.read_itp(
                self.sample, self.ff
            )  # Read the original itp file for the core
            self.np_block = self.ff.blocks[self.np_component]
            core_molecule = self.ff.blocks[
                self.np_component
            ].to_molecule()  # Make metamolcule object from the core block

            core_block = self._extract_block_atom(
                core_molecule, self.np_atype, {}
            )  # Get only the gold atoms from the itp

            self.core_block = core_block
            self.core_len, self.core = len(core_block), len(core_block)
            core_block.nrexcl = self.nrexcl
            self.np_molecule_new = core_block.to_molecule()
            # Generate the positions for the core and assign as position elements
            NanoparticleCoordinates().run_molecule(self.np_molecule_new)
            self.core_center = center_of_geometry(
                np.asarray(
                    list(
                        (
                            nx.get_node_attributes(
                                self.np_molecule_new, "position"
                            ).values()
                        )
                    )
                )
            )

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
        assert (
            self.np_component in self.ff.blocks.keys()
        ), "We have not stored NP information"
        surface_atom_dict = {}

        # Create dictionary
        for atom in self.ff.blocks[self.np_component].atoms:
            surface_atom_dict[atom["index"]] = atom["atype"]

        # Find the AU atoms that have a bond with the S atoms
        self.surface_atoms = []

        for interaction_node in self.ff.blocks[self.np_component].interactions["bonds"]:
            logging.info(f"Printing {interaction_node}")
            interaction = [
                surface_atom_dict[
                    interaction_node.atoms[0] + 1
                ],  # add one to ensure we have a non zero index start
                surface_atom_dict[interaction_node.atoms[1] + 1],
            ]
            if "S" in interaction and "AU" in interaction:
                self.surface_atoms.append(interaction_node.atoms[1] + 1)

        # get only the unique atoms
        self.surface_atoms = list(set(self.surface_atoms))

    def _identify_indices_for_core_attachment(self) -> None:
        """
        Based on the coordinates of the NP core provided,
        this function could just return a list - not sure it has to be a class
        attribute

        In the case of a spherical core, find the radius of the core

        The identify_surface function is only called when we have a ligand-functionalized
        gold nanoparticle that we would like to strip of the ligand molecules,
        and then identify the surface ligands from that. Otherwise, we assume
        self.identify_surface as being false, meaning that we have a straightforward
        input of a core

        Parameters
        ----------
        None: None

        Returns
        -------
        None

        """
        # assign molecule object
        # init_coords = expand_initial_coords(self.np_block)
        NanoparticleCoordinates().run_molecule(
            self.np_molecule_new
        )  # this is repetitive so we don't need this
        core_numpy_coords = np.asarray(
            list((nx.get_node_attributes(self.np_molecule_new, "position").values()))
        )  # get the cartesian positions of the main core
        self.core_center = center_of_geometry(core_numpy_coords)
        logging.info(
            f"The filtered NP core has {len(core_numpy_coords)} number of atoms"
        )
        # Find the center of geometry of the core
        center_np = center_of_geometry(core_numpy_coords)
        # Compute the radius
        radius_list = []
        for coord in core_numpy_coords:
            radius = distance.euclidean(center_np, coord)
            radius_list.append(radius)

        # Get the maximum recorded radius and find the max length based on that
        radius = max(radius_list)
        length = radius * 2
        self.max_radius = radius
        # Find indices in the case of Janus and Striped particles
        minimum_threshold, maximum_threshold = min(core_numpy_coords[:, 2]), max(
            core_numpy_coords[:, 2]
        )
        # the logic of this part takes from the original NPMDPackage code
        self.core_indices = create_np_pattern(
            self.pattern,
            core_numpy_coords,
            length,
            minimum_threshold,
            maximum_threshold,
        )

        # Can definitely add more potential patterns here
        if (
            self.identify_surface
        ):  # filter out the surface indices based on whether we think they are on the surface
            # identify the surface gold atoms
            self._identify_lattice_surface()
            # filter out from the Janus and non-Janus to
            for core_index in range(0, len(self.core_indices)):

                self.core_indices[core_index] = {
                    x: self.core_indices[core_index][x]
                    for x in self.surface_atoms
                    if x in list(self.core_indices[core_index])
                }

    def _identify_attachment_index(
        self, ligand_block, anchor_atom: str, type_ff: str = "MARTINI"
    ) -> int:
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
        for index, entry in enumerate(list(ligand_block.atoms)):
            if entry["atype"] == anchor_atom:
                attachment_index = index
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
        # ensure we have a ff initialized
        self.ligand_block_specs = {}
        for (
            ligand
        ) in self.ligands:  # open the itp file and read in the ligand file as necessary
            with open(self.ligand_path + "/" + ligand, "r") as f:
                ligand_file = f.read()
                ligand_file = ligand_file.split("\n")
                # register the itp file
                vermouth.gmx.itp_read.read_itp(ligand_file, self.ff)
                # assert len(self.ff.blocks.keys()) == len(self.ligand_N)

        for index, block_name in enumerate(self.ff.blocks.keys()):
            logging.info(f"{index}, {block_name}")
            if block_name != self.np_component:  # If the block is not that of the core
                # reset the resid
                # self.ff.blocks.key
                ligand_index = (
                    index - 1
                )  # As the first index represents the core, the starting index will always be
                # 1, hence we need to subtract to ensure that the indexing is correct

                NanoparticleCoordinates().run_molecule(
                    self.ff.blocks[block_name]
                )  # generate the position of the atoms for the ligands
                self.ligand_block_specs[block_name] = {
                    "name": block_name,  # store the ligand block name
                    "length": len(
                        list(self.ff.blocks[block_name].atoms)
                    ),  # store length of the ligand
                    "N": self.ligand_N[
                        # ligand_index
                        ligand_index
                    ],  # store number of ligands of that type to attach onto the core.
                    "anchor_index": self._identify_attachment_index(
                        self.ff.blocks[block_name],
                        self.ligand_anchor_atoms[ligand_index],
                    ),  # store the nth atom of the ligand that correponds to anchor.
                    "tail_index": self._identify_attachment_index(
                        self.ff.blocks[block_name],
                        self.ligand_tail_atoms[ligand_index],
                    ),  # store the nth atom of the ligand that correponds to anchor.
                    "indices": self.core_indices[
                        ligand_index
                    ],  # store the core indices to which we want to attach the ligands (of that specific type) to.
                    "core": self.core_center,  # center of mass for the core.
                }

    def _add_block_indices(self) -> None:
        """
        Scale the indexes so that when we create the ligand functionalized NP,
        the ligand component to it is built with the right indices. This requires scaling
        the index with the other ligand components and/or the core component

        Parameters
        ----------
        None: None

        Returns
        -------
        """
        core_size = len(
            list(self.np_block)
        )  # get the length of the core of the nanoparticle we are trying to build
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
        Looping over the core coordinates we picked out for functionalization
        also ensure that we are only attaching N number of ligands as specified
        when first building the class.
        """
        core_size = self.core  # get the core size
        resid_index = 2

        for key in self.ligand_block_specs.keys():
            attachment_list = {}
            resids = []
            adjusted_N_indices = list(self.ligand_block_specs[key]["indices"].keys())[
                : self.ligand_block_specs[key]["N"]
            ]
            for index, core_atom in enumerate(adjusted_N_indices, 1):
                # loop over the number of ligands we want to create
                # Get the core size and the final index of the ligand + core size
                attachment_list[index] = [
                    core_size,
                    core_size + self.ligand_block_specs[key]["length"],
                ]
                core_size += self.ligand_block_specs[key][
                    "length"
                ]  # Now that we have appened above, we need to increase
                # the core_size so that the index can be adjusted for the next appended ligand

                self.np_molecule_new.merge_molecule(
                    self.ff.blocks[key]
                )  # append to block
                resids.append(resid_index)  # why is this 2?
                resid_index += 1

            self.ligand_block_specs[key]["shift_index"] = attachment_list
            self.ligand_block_specs[key]["resids"] = resids

    def _generate_bonds(self) -> None:
        """
        Parameters
        ----------
        None: None

        Returns
        -------
        """
        core_size = self.core
        # get random N elements from the list
        for key in self.ligand_block_specs.keys():
            logging.info(f"bonds for {key}")
            adjusted_N_indices = list(self.ligand_block_specs[key]["indices"].keys())[
                : self.ligand_block_specs[key]["N"]
            ]
            logging.info(f"adjusted {adjusted_N_indices}")
            for index, entry in enumerate(adjusted_N_indices):
                base_anchor = (
                    core_size
                    + (index * self.ligand_block_specs[key]["length"])
                    + self.ligand_block_specs[key]["anchor_index"]
                )
                interaction = vermouth.molecule.Interaction(
                    atoms=(
                        entry,
                        base_anchor,
                    ),
                    # bonding parameters to add in for the head of the ligands to the nanoparticle core
                    parameters=[
                        "1",
                        "0.0013",
                        "500000.00",
                    ],
                    meta={},
                )
                logging.info(f"generating bonds between {entry} and {base_anchor}")

                self.np_molecule_new.interactions["bonds"].append(interaction)

            logging.info(
                f"core length is {self.core_len}, number of atoms with ligands added is {self.ligand_block_specs[key]['length'] * self.ligand_block_specs[key]['N']} "
            )
            core_size += (
                self.ligand_block_specs[key]["length"]
                * self.ligand_block_specs[key]["N"]
            )
            logging.info(f"core length is now {self.core_len}")

    def _initiate_nanoparticle_coordinates(self) -> None:
        """ """

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
        PositionChangeCore(self.gro_file, self.sample, self.np_atype).run_molecule(
            self.np_molecule_new
        )

        # shift the positions of the ligands so that they are initiated on the surface of the NP        self.np_component = np_component

        for resname in self.ligand_block_specs.keys():
            PositionChangeLigand(
                ligand_block_specs=self.ligand_block_specs,
                core_len=self.core_len,
                resname=resname,
                original_coordinates=self.original_coordinates,
                length=self.length,
                core_center=self.core_center,
            ).run_molecule(self.np_molecule_new)
        # prepare meta molecule
        self.meta_mol.molecule = self.np_molecule_new

    def create_gro(self, write_path: str) -> None:
        """
        Generate the gro file from the nanoparticle we have created.
        Ideally, we generate the coordinates with this function
        and then store it within a 'coordinates' object. The form
        of which I am not certain yet.
        """
        self.np_top = Topology(name="nanoparticle", force_field=self.ff)
        self.np_top.molecules = [self.meta_mol]
        self.np_top.box = np.array([10.0, 10.0, 10.0])
        system = self.np_top.convert_to_vermouth_system()
        gro.write_gro(
            system,
            write_path,
            precision=7,
            title="test_nanoparticle",
            box=self.np_top.box,
            defer_writing=False,
        )

    def write_itp(self, file_name: str) -> None:
        """
        Generate the itp file for the nanoparticle
        """
        # self.np_top = Topology(name="nanoparticle", force_field=self.ff)
        with open(f"{file_name}", "w") as outfile:
            write_molecule_itp(self.np_top.molecules[0].molecule, outfile)


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


# main code executable
if __name__ == "__main__":
    # The gold nanoparticle - generate the core of the opls force field work
    AUNP_model = NanoparticleModels(
        "/home/sang/Desktop/git/polyply_1.0/polyply/tests/test_data/np_test_files/AMBER_AU/au144.gro",
        return_amber_nps_type("au144_OPLS_bonded"),
        "NP2",
        "AU",
        "/home/sang/Desktop/git/polyply_1.0/polyply/tests/test_data/np_test_files/AMBER_AU/ligand",
        ["UNK_DA2640/UNK_DA2640.itp", "UNK_12B037/UNK_12B037.itp"],
        # ["UNK_DA2640/UNK_DA2640.itp"],
        [20, 20],
        "Janus",
        ["S07", "S00"],
        ["C08", "C07"],
        3,
        "test",
        original_coordinates={
            "DA": gro.read_gro(
                "/home/sang/Desktop/git/polyply_1.0/polyply/tests/test_data/np_test_files/AMBER_AU/ligand/UNK_DA2640/UNK_DA2640.gro"
            ),
            "12B": gro.read_gro(
                "/home/sang/Desktop/git/polyply_1.0/polyply/tests/test_data/np_test_files/AMBER_AU/ligand/UNK_12B037/UNK_12B037.gro"
            ),
        },
        identify_surface=False,
    )

    AUNP_model.core_generate_coordinates()
    AUNP_model._identify_indices_for_core_attachment()
    AUNP_model._ligand_generate_coordinates()
    AUNP_model._add_block_indices()  # Not sure whether we need this now ...
    AUNP_model._generate_ligand_np_interactions()
    AUNP_model._generate_bonds()
    AUNP_model._initiate_nanoparticle_coordinates()  # doesn't quite work yet.
    # Generating output files
    AUNP_model.create_gro("gold.gro")
    AUNP_model.write_itp("gold.itp")

    # PCBM nanoparticle (Coarse-grained) - constructing the PCBM
    PCBM_ligand_gro = "/home/sang/Desktop/git/polyply_1.0/polyply/tests/test_data/np_test_files/PCBM_CG/PCBM_ligand.gro"
    ## Creating the PCBM model
    PCBM_model = NanoparticleModels(
        "/home/sang/Desktop/git/polyply_1.0/polyply/tests/test_data/np_test_files/PCBM_CG/F16.gro",
        return_cg_nps_type("F16"),
        "F16",
        "CNP",
        "/home/sang/Desktop/git/polyply_1.0/polyply/tests/test_data/np_test_files/PCBM_CG/",
        ["PCBM_ligand.itp"],
        [10],
        "Striped",
        ["C4"],
        ["N1"],
        1,
        ff_name="test",
        original_coordinates={
            "PCBM": gro.read_gro(PCBM_ligand_gro),
        },
        identify_surface=False,
    )
    # Generate PCBM
    PCBM_model.core_generate_coordinates()
    PCBM_model._identify_indices_for_core_attachment()
    PCBM_model._ligand_generate_coordinates()
    PCBM_model._add_block_indices()  # Not sure whether we need this now ...
    PCBM_model._generate_ligand_np_interactions()
    PCBM_model._generate_bonds()
    PCBM_model._initiate_nanoparticle_coordinates()  # doesn't quite work yet.
    # Generating output files
    PCBM_model.create_gro("PCBM.gro")
    PCBM_model.write_itp("PCBM.itp")
