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
from collections import defaultdict
import numpy as np
import networkx as nx
import vermouth.forcefield
from vermouth.file_writer import DeferredFileWriter
from .generate_templates import GenerateTemplates
from .backmap import Backmap
from .topology import Topology
from .build_system import BuildSystem
from .annotate_ligands import AnnotateLigands, parse_residue_spec, _find_nodes
from .build_file_parser import read_build_file

def split_residues(molecules, split):
    for mol in molecules:
        max_resid = len(mol.nodes)
        for split_string in split:
            max_resid = mol.split_residue(split_string, max_resid)

def gen_coords(args):
    # Read in the topology
    topology = Topology.from_gmx_topfile(name=args.name, path=args.toppath)
    topology.preprocess()

    # check if molecules are all connected
    for molecule in topology.molecules:
        if not nx.is_connected(molecule):
            msg = ('\n Molecule {} consistes of two disconnected parts. '
                   'Make sure all atoms/particles in a molecule are '
                   'connected by bonds, constraints or virual-sites')
            raise IOError(msg.format(molecule.name))

    if args.split:
       split_residues(topology.molecules, args.split)

    # read in coordinates if there are any
    if args.coordpath:
        topology.add_positions_from_file(args.coordpath, args.build_res)
    else:
        for molecule in topology.molecules:
            for node in molecule.nodes:
                molecule.nodes[node]["build"] = True

    for molecule in topology.molecules:
        for node in molecule.nodes:
            if not molecule.nodes[node]["build"]:
                molecule.nodes[node]["build"] = False

    # load in built file
    if args.build:
        with open(args.build) as build_file:
            lines = build_file.readlines()
            read_build_file(lines, topology.molecules)

    # deal with box-input
    if len(args.box) != 0:
        box = np.array(args.box)
    else:
        box = []

    # handle grid input
    if args.grid:
        grid = np.loadtxt(args.grid)
    else:
        grid = None

    # handle starting point input
    start_dict = {mol_idx:None for mol_idx, _ in enumerate(topology.molecules)}
    for start in args.start:
        res_spec = parse_residue_spec(start)
        if 'mol_idx' in res_spec:
            mol_idx = res_spec['mol_idx']
            node = _find_nodes(topology.molecules[mol_idx], res_spec)[0]
            start_dict[mol_idx] = node
        else:
            for idx, molecule in enumerate(topology.molecules):
                if molecule.mol_name == res_spec['molname']:
                    node = _find_nodes(molecule, res_spec)[0]
                    start_dict[idx] = node

    # Build polymer structure
    GenerateTemplates(topology=topology, max_opt=10).run_system(topology)

    AnnotateLigands(topology, args.ligands).run_system(topology)

    BuildSystem(topology,
                start_dict=start_dict,
                density=args.density,
                max_force=args.max_force,
                grid_spacing=args.grid_spacing,
                maxiter=args.maxiter,
                maxiter_random=args.maxiter_random,
                box=box,
                step_fudge=args.step_fudge,
                push=args.push,
                ignore=args.ignore,
                grid=grid).run_system(topology.molecules)

    AnnotateLigands(topology, args.ligands).split_ligands()

    Backmap().run_system(topology)

    # Write output
    system = topology.convert_to_vermouth_system()
    vermouth.gmx.gro.write_gro(system, args.outpath, precision=7,
                               title='polyply structure', box=topology.box)
    DeferredFileWriter().write()
