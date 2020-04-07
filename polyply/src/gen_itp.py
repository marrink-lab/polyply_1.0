#!/usr/bin/env python3

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

def read_ff_from_file(paths, force_field):
    """
    read the itp and ff files for the defintion of blocks and links.
    """
    for path in paths:
        file_extension = path.suffix.upper()[1:]
        if file_extension == "FF":
            with open(path, 'r') as file_:
                lines = file_.readlines()
                vermouth.ffinput.read_ff(lines, force_field=force_field)
        elif file_extension == "ITP":
            with open(path, 'r') as file_:
                lines = file_.readlines()
                polyply.src.parsers.read_polyply(lines, force_field=force_field)
        else:
            raise IOError("Unkown file extension {}.".format(file_extension))
    return force_field

def split_seq_string(sequence):
    """
    Split a string defnintion for a linear sequence into monomer
    blocks and raise errors if the squence is not valid.
    """
    raw_monomers = sequence.split()
    monomers = []
    for monomer in raw_monomers:
        resname, n_blocks = monomer.split(":")
        n_blocks = int(n_blocks)
        if not isinstance(resname, str):
            raise ValueError("Resname {} is not a string.".format(resname))
        monomers.append(Monomer(resname=resname, n_blocks=n_blocks))
    return monomers

def generate_meta_molecule(raw_graph, force_field, name):
    """
    generate the meta molecule from intial graph input
    that comes from files or sequence definition.
    """
    try:
        file_extension = raw_graph.split(".")[1]
        if  file_extension == "json":
            meta_mol = MetaMolecule.from_json(json_file=raw_graph,
                                              force_field=force_field,
                                              mol_name=name)
   #    elif file_extension == "itp":
   #        meta_mol = MetaMolecule.from_itp(raw_graph, force_field, name)
    except IndexError:
        try:
            monomers = split_seq_string(raw_graph)
            meta_mol = MetaMolecule.from_monomer_seq_linear(monomers=monomers,
                                                            force_field=force_field,
                                                            mol_name=name)
        except ValueError:
            raise IOError("Unkown file fromat or sequence string {}.".format(raw_graph))

    return meta_mol

def gen_itp(args):

    known_force_fields = vermouth.forcefield.find_force_fields(
        Path(DATA_PATH) / 'force_fields'
    )

    if args.lib:
        force_field = known_force_fields[args.lib]
    else:
        force_field = vermouth.forcefield.ForceField(name=args.name)

    if args.inpath:
        read_ff_from_file(args.inpath, force_field)

    meta_molecule = generate_meta_molecule(args.raw_graph, force_field, args.name)
    meta_molecule = MapToMolecule().run_molecule(meta_molecule)
    meta_molecule = ApplyLinks().run_molecule(meta_molecule)

    with open('{}'.format(args.outpath), 'w') as outpath:
        vermouth.gmx.itp.write_molecule_itp(meta_molecule.molecule, outpath,
                                            moltype=args.name, header=["polyply-itp"])
