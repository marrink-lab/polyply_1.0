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
High level API for the polyply coordinate generator
"""
import sys
import numpy as np
import networkx as nx
import vermouth.forcefield
from vermouth.file_writer import DeferredFileWriter
from vermouth.log_helpers import StyleAdapter, get_logger
from .generate_templates import GenerateTemplates
from .backmap import Backmap
from .topology import Topology
from .build_system import BuildSystem
from .annotate_ligands import AnnotateLigands, parse_residue_spec, _find_nodes
from .build_file_parser import read_build_file
from .check_residue_equivalence import check_residue_equivalence

LOGGER = StyleAdapter(get_logger(__name__))

def find_starting_node_from_spec(topology, start_nodes):
    """
    Given a definition of one or multiple nodes in
    `start_nodes` find the corresponding molecule index
    and node key in `topology`. The format of start nodes
    is equivalent to that for annotating the residue and
    is described in detail in the function
    :meth:`polyply.src.annotate_ligands.annotate_molecules.parse_residue_spec`.

    Parameters:
    -----------
    topology: :class:`polyply.src.topology.Topology`
    start_nodes: list[str]
        a list of residue-specs

    Returns:
    --------
    dict
        a dict of molecule_idx as key and node key as value
    """
    start_dict = {mol_idx: None for mol_idx in range(len(topology.molecules))}
    for start in start_nodes:
        res_spec = parse_residue_spec(start)
        if 'mol_idx' in res_spec:
            mol_idx = res_spec['mol_idx']
            node = list(_find_nodes(topology.molecules[mol_idx], res_spec))[0]
            start_dict[mol_idx] = node
            topology.molecules[mol_idx].root = node
        else:
            for idx, molecule in enumerate(topology.molecules):
                if molecule.mol_name == res_spec['molname']:
                    node = list(_find_nodes(molecule, res_spec))[0]
                    start_dict[idx] = node
    return start_dict

def _initialize_cylces(topology, cycles, tolerance):
    for mol_name in cycles:
        for mol_idx in topology.mol_idx_by_name[mol_name]:
            # initalize as dfs tree
            molecule = topology.molecules[mol_idx]
            cycles = nx.cycle_basis(molecule)
            if len(cycles) > 1:
                raise IOError("More than one cycle is not allowed.")
            molecule.dfs=True
            nodes = (list(molecule.search_tree.edges)[0][0],
                     list(molecule.search_tree.edges)[-1][1])
            topology.distance_restraints[(mol_name, mol_idx)][nodes] = (0.0, tolerance)

def _check_molecules(molecules):
    """
    Helper method which raises an IOError
    if any molecule in `molecules` is
    disconnected.
    """
    # check if molecules are all connected
    for molecule in molecules:
        if not nx.is_connected(molecule):
            msg = ('\n Molecule {} consistes of two disconnected parts. '
                   'Make sure all atoms/particles in a molecule are '
                   'connected by bonds, constraints or virual-sites')
            raise IOError(msg.format(molecule.name))

def gen_coords(toppath,
               outpath,
               name,
               coordpath=None,
               coordpath_meta=None,
               build=None,
               build_res=[],
               ignore=[],
               cycles=[],
               cycle_tol=0.0,
               split=[],
               ligands=[],
               grid_spacing=0.2,
               grid=None,
               maxiter=800,
               start=[],
               density=None,
               box=None,
               maxiter_random=100,
               step_fudge=1.0,
               max_force=5*10**4.0,
               nrewind=5,
               bfudge=0.4):
    """
    Subprogram for coordinate generation which implements the default
    polyply workflow for structure generation. In general, a topology
    file is read from which all molecules are extracted. Subsequently for each
    residue in the system a template is built. Afterwards a random walk
    is performed for each residue based on a volume estimated from
    the templates. Once the random walk residue coordinates are generated, they
    are backmapped.

    Parameters
    ----------
    toppath: :class:pathlib.Path
        Path to topology file
    outpath: :class:pathlib.Path
        Path to coordinate file
    name: str
        Name of the molecule
    build: :class:pathlib.Path
        Path to build file
    build_res: list[str]
        List of resnames to build
    ignore: list[str]
        List of molecule names to ignore
    cycles: list[str]
        List of cyclic molecule names
    cycle_tol: float
        Tolerance in nm for cycles to be closed
    split: list[str]
        Split single residue into more residues. The syntax is
        <resname>:<new_resname>-<atom1>,<atom2><etc>:<new_resname>
        Note that all atoms of the old residues need to be in at
        most one new residue and must all be accounted for.
    ligands: list[str]
        Specify ligands of molecules which should be placed close
        to a specific molecule. The format is as follows:
        <mol_name>#<mol_idx>-<resname>#<resid>:<mol_name>#<mol_idx>-<resname>#<resid>.
        Elements of the specification can be omitted as required.
        Note mol_name and resname cannot contain the characters
        # and -.
    grid_spacing: float
        Grid spacing in nm
    grid: str
        Path to file with grid-points
    maxiter: int
        Maximum number of tries to generate coordinates for a molecule.
        The default is 800.
    start: list[str]
        Specify which residue to build first. The syntax is
        <mol_name>#<mol_idx>-<resname>#<resid>. Parts of this
        specification may be omitted.
        Note mol_name and resname cannot contain the characters #
        and -.
    density: float
        Density of the system (kg/m3)
    box: np.ndarray(1,3)
        Rectangular box x, y, z in nm
    maxiter_random: int
        Maximum number of tries to place a residue with
        the random walk module.
    step_fudge: float
        Scale the step length in random walk by this fudge factor.
        The default is 1.
    max_force: float
        Maximum force under which placement of a residue is accepted.
        The default is 5x10^4 kJ/(mol*nm).
    nrewind: int
        Number of residues to trace back when random walk step fails in first
        attempt. The default is 5.
    bfudge: float
        Fudge factor by which to scale the coordinates of the residues
        during the backmapping step. 1 will result in to-scale coordinates
        but will likely generate overlaps.
    """

    # Read in the topology
    LOGGER.info("reading topology",  type="step")
    topology = Topology.from_gmx_topfile(name=name, path=toppath)

    LOGGER.info("processing topology",  type="step")
    topology.preprocess()
    _check_molecules(topology.molecules)

    if split:
        LOGGER.info("splitting residues",  type="step")
        for molecule in topology.molecules:
            molecule.split_residue(split)

    # read in coordinates if there are any
    if coordpath:
        LOGGER.info("loading molecule coordinates",  type="step")
        topology.add_positions_from_file(coordpath,
                                         skip_res=build_res,
                                         resolution='mol')

    # read in meta-molecule coordinates of there are any
    if coordpath_meta:
        LOGGER.info("loading meta_molecule coordinates",  type="step")
        topology.add_positions_from_file(coordpath_meta,
                                         skip_res=build_res,
                                         resolution='meta_mol')
        last_node = len(topology.molecules[-1].nodes) - 1
        if topology.molecules[-1].nodes[last_node]["build"]:
            msg = ("You provided metamolecule coordiantes. However, "
                   "there were not enough coordinates for all metamolecule "
                   "residues. Polyply will built the missing residues.")
            LOGGER.warning(msg)

    # load in built file
    if build:
        LOGGER.info("reading build file",  type="step")
        with open(build) as build_file:
            lines = build_file.readlines()
            read_build_file(lines, topology.molecules, topology)

    # collect all starting points for the molecules
    start_dict = find_starting_node_from_spec(topology, start)

    # handle grid input
    if grid:
        LOGGER.info("loading grid",  type="step")
        grid = np.loadtxt(grid)

    # do a sanity check
    LOGGER.info("checking residue integrity",  type="step")
    check_residue_equivalence(topology)
    # Build polymer structure
    LOGGER.info("generating templates",  type="step")
    GenerateTemplates(topology=topology, max_opt=10).run_system(topology)
    LOGGER.info("annotating ligands",  type="step")
    ligand_annotator = AnnotateLigands(topology, ligands)
    ligand_annotator.run_system(topology)
    LOGGER.info("generating system coordinates",  type="step")
    _initialize_cylces(topology, cycles, cycle_tol)
    BuildSystem(topology,
                start_dict=start_dict,
                density=density,
                max_force=max_force,
                grid_spacing=grid_spacing,
                maxiter=maxiter,
                box=box,
                step_fudge=step_fudge,
                ignore=ignore,
                grid=grid,
                cycles=cycles,
                nrewind=nrewind).run_system(topology.molecules)
    ligand_annotator.split_ligands()
    LOGGER.info("backmapping to target resolution",  type="step")
    Backmap(fudge_coords=bfudge).run_system(topology)
    # Write output
    LOGGER.info("writing output",  type="step")
    command = ' '.join(sys.argv)
    system = topology.convert_to_vermouth_system()
    vermouth.gmx.gro.write_gro(system, outpath, precision=7,
                               title=command, box=topology.box)
    DeferredFileWriter().write()
