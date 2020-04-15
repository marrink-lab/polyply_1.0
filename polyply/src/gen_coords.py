#!/usr/bin/env python3
"""
High level API for the polyply coordinate generator
"""

from pathlib import Path
import vermouth.forcefield
from vermouth.gmx.gro import read_gro
from polyply import (MetaMolecule, DATA_PATH)
from polyply.src.generate_templates import GenerateTemplates
from polyply.src.random_walk import RandomWalk
from polyply.src.backmap import Backmap
from .gen_itp import read_ff_from_file

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


def assign_vdw_radii(molecule, radii):
    for node in molecule.nodes:
        atom_type = molecule.nodes[node]["atomname"]
        for key in radii:
            if key in atom_type:
               molecule.nodes[node]["vdwradius"] = radii[key]


def assign_coordinates(molecule, coord_molecule, ignore=None):
    for node in molecule.nodes:
        point = coord_molecule[node]
        molecule.nodes[node]["position"] = point
        molecule.nodes[node]["build"] = False


def read_vdw_file(path):
    vdwradii = {}
    with open(path, 'r') as _file:
        lines = _file.readlines()
        for line in lines:
            atom, rad = line.split()[:2]
            vdwradii[atom] = float(rad)

    return vdwradii


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

    vdw_radii = read_vdw_file(args.vdw_path)

    # Assing vdw-raddi
    for _, block in force_field.blocks.items():
        assign_vdw_radii(block, vdw_radii)


    # Generate a meta-molecule from an itp file
    meta_molecule = MetaMolecule.from_itp(force_field, args.itppath, args.name)
    assign_vdw_radii(meta_molecule.molecule, vdw_radii)

    # Import coordinates if there are any
    if args.coordpath:
        coord_molecule = read_gro(
            args.coord_file, exclude=('SOL',), ignh=False)
        assign_coordinates(meta_molecule.molecule, coord_molecule)

    # Build polymer structure
    GenerateTemplates().run_molecule(meta_molecule)
    RandomWalk().run_molecule(meta_molecule)
    Backmap().run_molecule(meta_molecule)

    # Write output
    system= vermouth.System()
    system.molecules = [meta_molecule.molecule]
    system.force_field = force_field

    vermouth.gmx.gro.write_gro(system, args.outpath, precision=7,
              title='polyply structure', box=(10, 10, 10))
