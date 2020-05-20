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
High level API for the polyply itp generator
"""
from pathlib import Path
import vermouth
import vermouth.forcefield
import polyply
import polyply.src.polyply_parser
from polyply import (DATA_PATH, MetaMolecule, ApplyLinks, Monomer, MapToMolecule)
from .load_library import load_library

def split_seq_string(sequence):
    """
    Split a string definition for a linear sequence into monomer
    blocks and raise errors if the sequence is not valid.
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


    # Import of Itp and FF files
    force_field = load_library(args.name, args.lib, args.inpath)

    # Generate the MetaMolecule
    if args.seq:
       monomers = split_seq_string(args.seq)
       meta_molecule = MetaMolecule.from_monomer_seq_linear(monomers=monomers,
                                                            force_field=force_field,
                                                            mol_name=args.name)
    #ToDo
    # fix too broad except
    elif args.seq_file:
       extension = args.seq_file.suffix.casefold()[1:]
       try:
           parser = getattr(MetaMolecule, 'from_{}'.format(extension))
           meta_molecule = parser(json_file=args.seq_file,
                                  force_field=force_field,
                                  mol_name=args.name)
       except AttributeError:
         raise IOError("Cannot parse file with extension {}.".format(extension))

    # Do transformationa and apply link
    meta_molecule = MapToMolecule().run_molecule(meta_molecule)
    meta_molecule = ApplyLinks().run_molecule(meta_molecule)

    with open(args.outpath, 'w') as outpath:
        vermouth.gmx.itp.write_molecule_itp(meta_molecule.molecule, outpath,
                                            moltype=args.name, header=["polyply-itp"])
