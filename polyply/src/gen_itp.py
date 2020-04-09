#!/usr/bin/env python3

"""
High level API for the polyply itp generator
"""
from pathlib import Path
import vermouth
import vermouth.forcefield
import polyply
import polyply.src.parsers
from polyply import (DATA_PATH, MetaMolecule, ApplyLinks, Monomer, MapToMolecule)

def read_ff_from_file(paths, force_field):
    """
    read the input files for the defintion of blocks and links.

    Parameters
    ----------
    paths: list
           List of vaild file paths
    force_field: class:`vermouth.force_field.ForceField`

    Returns
    -------
    force_field: class:`vermouth.force_field.ForceField`
       updated forcefield

    """
    line_parsers = {"ff": vermouth.ffinput.read_ff,
                    "itp": polyply.src.parsers.read_polyply,
                    "rtp":  vermouth.gmx.rtp.read_rtp}

    def wrapper(parser, path, force_field):
        with open(path, 'r') as file_:
             lines = file_.readlines()
             parser(lines, force_field=force_field)

    for path in paths:
        file_extension = path.suffix.casefold()[1:]
        try:
           parser = line_parsers[file_extension]
           wrapper(parser, path, force_field)
        except KeyError:
            raise IOError("Cannot parse file with extension {}.".format(file_extension))

    return force_field

def split_seq_string(sequence):
    """
    Split a string defnintion for a linear sequence into monomer
    blocks and raise errors if the squence is not valid.

    Parameters
    -----------
    sequence: str
            string of residues format name:number

    Returns:
    ----------
    list
       list of `polyply.Monomers`
    """
    raw_monomers = sequence.split()
    monomers = []
    for monomer in raw_monomers:
        resname, n_blocks = monomer.split(":")
        n_blocks = int(n_blocks)
        monomers.append(Monomer(resname=resname, n_blocks=n_blocks))
    return monomers

def gen_itp(args):

    known_force_fields = vermouth.forcefield.find_force_fields(
        Path(DATA_PATH) / 'force_fields'
    )

    # Import of Itp and FF files
    if args.lib:
        force_field = known_force_fields[args.lib]
    else:
        force_field = vermouth.forcefield.ForceField(name=args.name)

    if args.inpath:
        read_ff_from_file(args.inpath, force_field)

    # Generate the MetaMolecule
    if args.seq:
       monomers = split_seq_string(args.seq)
       meta_molecule = MetaMolecule.from_monomer_seq_linear(monomers=monomers,
                                                            force_field=force_field,
                                                            mol_name=args.name)
    elif args.seq_file:
       extension = args.seq_file.suffix.casefold()[1:]
       print(extension)
       if extension in ["json", "itp"]:
          meta_molecule = MetaMolecule.from_json(json_file=args.seq_file,
                                            force_field=force_field,
                                            mol_name=args.name)
       else:
         raise IOError("Cannot parse file with extension {}.".format(extension))
    else:
         raise IOError("You need to provide a sequence either via -seqf or -seq flag.")

    # Do transformationa and apply link
    meta_molecule = MapToMolecule().run_molecule(meta_molecule)
    meta_molecule = ApplyLinks().run_molecule(meta_molecule)

    with open(args.outpath, 'w') as outpath:
        vermouth.gmx.itp.write_molecule_itp(meta_molecule.molecule, outpath,
                                            moltype=args.name, header=["polyply-itp"])
