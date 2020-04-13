"""
High level API for the polyply coordinate generator
"""
import argparse
from pathlib import Path
import vermouth
import vermouth.forcefield
import polyply
import polyply.src.parsers
from polyply import (DATA_PATH, MetaMolecule, ApplyLinks, Monomer, MapToMolecule)

def gen_coords(args):

    known_force_fields = vermouth.forcefield.find_force_fields(
        Path(DATA_PATH) / 'force_fields'
    )

    known_mappings = read_mapping_directory(Path(DATA_PATH) / 'mappings',
                                            known_force_fields)

    if args.lib:
        force_field = known_force_fields[args.lib]
    else:
        force_field = vermouth.forcefield.ForceField(name=args.name)

    if args.inpath:
        read_ff_from_file(args.inpath, force_field)

    read_mapping_directory("./", known_force_fields)


    meta_molecule = MetaMolecule.from_itp(args.inpath, force_field, args.name)
    meta_molecule.molecule = meta_molecule.force_field[args.name].to_molecule()
    AssignVolume.run_molecule(meta_molecule, vdwradii)
    RandomWalk.run_molecule(meta_molecule)


