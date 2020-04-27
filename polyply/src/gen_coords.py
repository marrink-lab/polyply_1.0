#!/usr/bin/env python3
"""
High level API for the polyply coordinate generator
"""

from pathlib import Path
import networkx as nx
import vermouth.forcefield
from vermouth.gmx.gro import read_gro
from polyply import (MetaMolecule, DATA_PATH)
from polyply.src.generate_templates import GenerateTemplates
from polyply.src.random_walk import RandomWalk
from polyply.src.backmap import Backmap
from .gen_itp import read_ff_from_file
from .minimizer import optimize_geometry
from .topology import Topology

def read_system(path, system, ignore_resnames=(), ignh=None, modelidx=None):
    """
    Read a system from a PDB or GRO file.
    This function guesses the file type based on the file extension.
    The resulting system does not have a force field and may not have edges.
    """
    file_extension = path.suffix.upper()[1:]  # We do not keep the dot
    if file_extension in ['PDB', 'ENT']:
        vermouth.PDBInput(str(path), exclude=ignore_resnames, ignh=ignh,
                          modelidx=modelidx).run_system(system)
    elif file_extension in ['GRO']:
        vermouth.GROInput(str(path), exclude=ignore_resnames,
                          ignh=ignh).run_system(system)
    else:
        raise ValueError('Unknown file extension "{}".'.format(file_extension))
    return system


def gen_coords(args):

    # Import the force field definitions
    known_force_fields = vermouth.forcefield.find_force_fields(
        Path(DATA_PATH) / 'force_fields'
    )

    if args.lib:
        force_field = known_force_fields[args.lib]
    else:
        force_field = vermouth.forcefield.ForceField(name=args.name)

    if args.inpath:
        read_ff_from_file(args.inpath, force_field)

    # Read in the topology
    topology = Topology.from_gmx_topfile(name=args.name, path=args.toppath)
    if args.coordpath:
       topology.add_positions_from_gro(args.coordpath)

    print(len(topology.molecules[0].nodes))
    print(nx.is_connected(topology.molecules[0]))
    print([ i for i in nx.dfs_edges(topology.molecules[0], source=0)])

    # Build polymer structure
    GenerateTemplates().run_system(topology)
    RandomWalk().run_system(topology)
    Backmap().run_system(topology)
    #energy_minimize().run_system(topology)

    system = topology.convert_to_vermouth_system()
    # Write output
    vermouth.gmx.gro.write_gro(system, args.outpath, precision=7,
              title='polyply structure', box=(10, 10, 10))
