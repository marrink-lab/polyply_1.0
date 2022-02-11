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
import sys
import networkx as nx
import vermouth
import vermouth.forcefield
from vermouth.log_helpers import StyleAdapter, get_logger
# patch to get rid of martinize dependency
try:
    from vermouth.file_writer import deferred_open
except ImportError:
    from vermouth.file_writer import open
    deferred_open = open
from vermouth.file_writer import DeferredFileWriter
from vermouth.citation_parser import citation_formatter
from polyply import (MetaMolecule, ApplyLinks, Monomer, MapToMolecule)
from .load_library import load_library

LOGGER = StyleAdapter(get_logger(__name__))

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
    raw_monomers = sequence
    monomers = []
    for monomer in raw_monomers:
        resname, n_blocks = monomer.split(":")
        n_blocks = int(n_blocks)
        monomers.append(Monomer(resname=resname, n_blocks=n_blocks))
    return monomers

def gen_params(args):
    # Import of Itp and FF files
    LOGGER.info("reading input and library files",  type="step")
    force_field = load_library(args.name, args.lib, args.inpath)

    # Generate the MetaMolecule
    if args.seq:
        LOGGER.info("reading sequence from command",  type="step")
        monomers = split_seq_string(args.seq)
        meta_molecule = MetaMolecule.from_monomer_seq_linear(monomers=monomers,
                                                             force_field=force_field,
                                                             mol_name=args.name)
    #ToDo
    # fix too broad except
    elif args.seq_file:
        LOGGER.info("reading sequence from file",  type="step")
        extension = args.seq_file.suffix.casefold()[1:]
        try:
            parser = getattr(MetaMolecule, 'from_{}'.format(extension))
            meta_molecule = parser(json_file=args.seq_file,
                                  force_field=force_field,
                                  mol_name=args.name)
        except AttributeError:
            raise IOError("Cannot parse file with extension {}.".format(extension))

    # Do transformationa and apply link
    LOGGER.info("mapping sequence to molecule",  type="step")
    meta_molecule = MapToMolecule(force_field).run_molecule(meta_molecule)
    LOGGER.info("applying links between residues",  type="step")
    meta_molecule = ApplyLinks().run_molecule(meta_molecule)

    # Raise warning if molecule is disconnected
    if not nx.is_connected(meta_molecule.molecule):
        n_components = len(list(nx.connected_components(meta_molecule.molecule)))
        msg = "You molecule consists of {:d} disjoint parts. Perhaps links were not applied correctly."
        LOGGER.warning(msg, (n_components))

    with deferred_open(args.outpath, 'w') as outpath:
        header = [ ' '.join(sys.argv) + "\n" ]
        header.append("Please cite the following papers:")
        for citation in meta_molecule.molecule.citations:
            cite_string =  citation_formatter(meta_molecule.molecule.force_field.citations[citation])
            LOGGER.info("Please cite: " + cite_string)
            header.append(cite_string)

        vermouth.gmx.itp.write_molecule_itp(meta_molecule.molecule, outpath,
                                            moltype=args.name, header=header)
    DeferredFileWriter().write()

# ducktape for renaming the itp tool
gen_itp = gen_params
