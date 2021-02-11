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
from collections import defaultdict
from functools import partial
import multiprocessing
import numpy as np
import networkx as nx
from tqdm import tqdm
import vermouth.forcefield
from vermouth.file_writer import DeferredFileWriter
from vermouth.log_helpers import StyleAdapter, get_logger
from .generate_templates import GenerateTemplates
from .backmap import Backmap
from .topology import Topology
from .build_system import BuildSystem
from .annotate_ligands import AnnotateLigands, parse_residue_spec, _find_nodes
from .build_file_parser import read_build_file

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
        else:
            for idx, molecule in enumerate(topology.molecules):
                if molecule.mol_name == res_spec['molname']:
                    node = list(_find_nodes(molecule, res_spec))[0]
                    start_dict[idx] = node
    return start_dict

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

def gen_coords(args):
    # Read in the topology
    LOGGER.info("reading topology",  type="step")
    topology = Topology.from_gmx_topfile(name=args.name, path=args.toppath)

    LOGGER.info("processing topology",  type="step")
    topology.preprocess()
    _check_molecules(topology.molecules)

    if args.split:
       LOGGER.info("splitting residues",  type="step")
       for molecule in topology.molecules:
           molecule.split_residue(args.split)

    # read in coordinates if there are any
    if args.coordpath:
        LOGGER.info("loading coordinates",  type="step")
        topology.add_positions_from_file(args.coordpath, args.build_res)

    # load in built file
    if args.build:
        LOGGER.info("reading build file",  type="step")
        with open(args.build) as build_file:
            lines = build_file.readlines()
            read_build_file(lines, topology.molecules)

    # collect all starting points for the molecules
    start_dict = find_starting_node_from_spec(topology, args.start)

    # handle grid input
    if args.grid:
        LOGGER.info("loading grid",  type="step")
        args.grid = np.loadtxt(args.grid)

    # Build polymer structure
    LOGGER.info("generating templates",  type="step")
    GenerateTemplates(topology=topology, max_opt=10).run_system(topology)
    LOGGER.info("annotating ligands",  type="step")
    AnnotateLigands(topology, args.ligands).run_system(topology)
    LOGGER.info("generating system coordinates",  type="step")
    BuildSystem(topology,
                start_dict=start_dict,
                density=args.density,
                max_force=args.max_force,
                grid_spacing=args.grid_spacing,
                maxiter=args.maxiter,
                box=args.box,
                step_fudge=args.step_fudge,
                ignore=args.ignore,
                grid=args.grid,
                nrewind=args.nrewind).run_system(topology.molecules)
    AnnotateLigands(topology, args.ligands).split_ligands()
    LOGGER.info("backmapping to target resolution",  type="step")
    Backmap(fudge_coords=args.bfudge).run_system(topology)
    # Write output
    LOGGER.info("writing output",  type="step")
    command = ' '.join(sys.argv)
    system = topology.convert_to_vermouth_system()
    vermouth.gmx.gro.write_gro(system, args.outpath, precision=7,
                               title=command, box=topology.box)
    DeferredFileWriter().write()
