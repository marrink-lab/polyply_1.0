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

VERSION = 'polyply version {}'.format(polyply.__version__)

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
            meta_mol = MetaMolecule.from_json(raw_graph, force_field, name)
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

def main():
    """
    Parses commandline arguments and performs the logic.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-V', '--version', action='version', version=VERSION)
    parser.add_argument('-name', required=True, type=str, dest="name",
 		       help="name of the final molecule")

    file_group = parser.add_argument_group('Input and output files')
    file_group.add_argument('-f', dest='inpath', required=False, type=Path,
                            help='Input file (ITP|FF)',nargs="*" )
    file_group.add_argument('-o', dest='outpath', type=Path,
                            help='Output ITP (ITP)')
    file_group.add_argument('-seq', dest='raw_graph', required=True, type=str,
                            help='Either linear sequence or graph input file (JSON|ITP)')

    ff_group = parser.add_argument_group('Force field selection')
    ff_group.add_argument('-lib', dest='lib', required=False, type=str,
                           help='force-fields to include from library')
    ff_group.add_argument('-ff-dir', dest='extra_ff_dir', action='append',
                          type=Path, default=[],
                          help='Additional repository for custom force fields.')
    ff_group.add_argument('-list-lib', action='store_true', dest='list_ff',
                          help='List all known force fields, and exit.')
    ff_group.add_argument('-list-blocks', action='store_true', dest='list_blocks',
                          help='List all Blocks known to the'
                          ' force field, and exit.')
    ff_group.add_argument('-list-links', action='store_true', dest='list_links',
                          help='List all Links known to the'
                          ' force field, and exit.')

   #martini_group = parser.add_argument_group('Martini specifics for protein and DNA.')
   #martini_group.add_argument('-nt', dest='neutral_termini',
   #                        action='store_true', default=False,
   #                        help='Set neutral termini (charged is default)')
   #martini_group.add_argument('-scfix', dest='scfix',
   #                        action='store_true', default=False,
   #                        help='Apply side chain corrections.')
   #martini_group.add_argument('-cys', dest='cystein_bridge',
   #                        type=_cys_argument,
   #                        default='none', help='Cystein bonds')

   #PTM_group = parser.add_argument_group('Martini specifics for protein and DNA.')
   #PTM_group.add_argument('-nt', dest='neutral_termini',
   #                        action='store_true', default=False,
   #                        help='Set neutral termini (charged is default)')
   #                        default='none', help='Cystein bonds')


    args = parser.parse_args()
    # start by reading and extending the force-field
    known_force_fields = vermouth.forcefield.find_force_fields(
        Path(DATA_PATH) / 'force_fields'
    )

    for directory in args.extra_ff_dir:
        try:
            vermouth.forcefield.find_force_fields(directory, known_force_fields)
        except FileNotFoundError:
            msg = '"{}" given to the -ff-dir option should be a directory.'
            raise ValueError(msg.format(directory))

    if args.list_ff:
        print('The following force fields are known:')
        for idx, ff_name in enumerate(reversed(list(known_force_fields)), 1):
            print('{:3d}. {}'.format(idx, ff_name))
        parser.exit()

    if args.list_blocks:
        print('The following Blocks are known to force field {}:'.format(args.lib))
        print(', '.join(known_force_fields[args.lib].blocks))
    if args.list_links:
        print('The following Links are known to force field {}:'.format(args.lib))
        print(', '.join(known_force_fields[args.lib].links))
        parser.exit()

    # Here starts the main polyply itp generation
    if args.lib:
        force_field = known_force_fields[args.lib]
    else:
        force_field = vermouth.forcefield.ForceField(name=args.name)

    if args.inpath:
        read_ff_from_file(args.inpath, force_field)

    meta_molecule = generate_meta_molecule(args.raw_graph, force_field, args.name)
    meta_molecule = MapToMolecule().run_molecule(meta_molecule)
    meta_molecule = ApplyLinks().run_molecule(meta_molecule)

    with open('{}.itp'.format(args.outpath), 'w') as outfile:
        vermouth.gmx.itp.write_molecule_itp(meta_molecule.molecule, outfile,moltype=args.name, header=["polyply-itp"])

if __name__ == '__main__':
    main()
