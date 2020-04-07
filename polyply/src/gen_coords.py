"""
High level API for the polyply itp generator
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

    if args.lib:
        force_field = known_force_fields[args.lib]
    else:
        force_field = vermouth.forcefield.ForceField(name=args.name)

    if args.inpath:
        read_ff_from_file(args.inpath, force_field)

    meta_molecule = MetaMolecule.from_itp(args.inpath, force_field, args.name)
    meta_molecule = MapToMolecule().run_molecule(meta_molecule)
    meta_molecule = ApplyLinks().run_molecule(meta_molecule)

    with open('{}'.format(args.outpath), 'w') as outpath:
        vermouth.gmx.itp.write_molecule_itp(meta_molecule.molecule, outpath,
                                            moltype=args.name, header=["polyply-itp"])
