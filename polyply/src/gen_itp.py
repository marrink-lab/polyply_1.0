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
from pathlib import Path
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
from vermouth.graph_utils import make_residue_graph
from polyply import (MetaMolecule, ApplyLinks, Monomer, MapToMolecule, MakeIDP)
from polyply.src.graph_utils import find_missing_edges
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

def gen_params(name="polymer", outpath=Path("polymer.itp"), inpath=None, lib=None, seq=None, seq_file=None, idp=False, idp_override=False):
    """
    Top level function for running the polyply parameter generation.
    Parameters seq and seq_file are mutually exclusive. Set the other
    to None. Of inpath and lib only one has to be given. Set the other
    to None if not used.

    Parameters
    ----------
    name: str
        name of the molecule in the itp file
    outpath: :class:`pathlib.Path`
        file path for output file
    inpath: list[pathlib.Path]
        list of paths to files with input definitions
    library: str
        name of the force field library to use
    seq: list[str]
        list of strings with format "resname:#monomers"
    seqf: :class:`pathlib.Path`
        file path to valid sequence file (.json/.fasta/.ig/.txt)
    idp: bool
        if True, treat the input sequence as an IDP to generate using Martini IDP parameters
    """
    # Import of Itp and FF files
    LOGGER.info("reading input and library files",  type="step")
    force_field = load_library(name, lib, inpath)

    # Generate the MetaMolecule
    if seq:
        LOGGER.info("reading sequence from command",  type="step")
        monomers = split_seq_string(seq)
        meta_molecule = MetaMolecule.from_monomer_seq_linear(monomers=monomers,
                                                             force_field=force_field,
                                                             mol_name=name)
    elif seq_file:
        LOGGER.info("reading sequence from file",  type="step")
        meta_molecule = MetaMolecule.from_sequence_file(force_field, seq_file, name)

    # Do transformationa and apply link
    LOGGER.info("mapping sequence to molecule",  type="step")
    meta_molecule = MapToMolecule(force_field).run_molecule(meta_molecule)
    LOGGER.info("applying links between residues",  type="step")
    meta_molecule = ApplyLinks().run_molecule(meta_molecule)

    if idp == True and lib[0] == 'martini3':
        LOGGER.info("checking IDP sequence to ensure disorder", type="step")
        if seq:
            check = MakeIDP().check_seq(sequence = split_seq_string(seq), idp_override=idp_override)
        elif seq_file:
            check = MakeIDP().check_seq(sequence_file = seq_file, idp_override=idp_override)
        LOGGER.info(check, type="step")

        meta_molecule = MakeIDP().run_molecule(meta_molecule)
        LOGGER.warning(("Backbone virtual sites for increased interactions with "
                        "water have been added. Please define this in your itp file! "
                        "Note: the addition of the virtual sites will now raise "
                        "a disjointed molecule warning. This can be ignored."))
    elif idp == True and lib[0] != 'martini3':
        msg = ("The IDP flag has been specified but the forcefield is not "
               "Martini 3. The IDP functionality is not designed for other "
               "forcefields. Please check your input!")
        raise IOError(msg)


    # Raise warning if molecule is disconnected
    if not nx.is_connected(meta_molecule.molecule):
        LOGGER.warning("Your molecule consists of disjoint parts."
                       "Perhaps links were not applied correctly.")
        msg = "Missing link between residue {idxA} {resA} and residue {idxB} {resB}"
        for missing in find_missing_edges(meta_molecule, meta_molecule.molecule):
            LOGGER.warning(msg, **missing)

    with deferred_open(outpath, 'w') as outfile:
        header = [ ' '.join(sys.argv) + "\n" ]
        if idp == True and lib[0] == 'martini3':
            idp_header_msg0 = (f'This parameter file for {name} has used the Martini 3 IDP parameters. '
                              'The additional following lines MUST be included in your master itp file:'
                              )
            idp_header_msg1 = 'VS 0.00 0.000 A 0.0 0.0'
            idp_header_msg2 = 'VS    W   1 0.4650000000    0.5000000000 '
            idp_header_msg3 = 'in the [ atomtypes ] and [ nonbond_params ] directives respectively.\n'
            header.append(idp_header_msg0)
            header.append(idp_header_msg1)
            header.append(idp_header_msg2)
            header.append(idp_header_msg3)

        header.append("Please cite the following papers:")
        for citation in meta_molecule.molecule.citations:
            cite_string =  citation_formatter(meta_molecule.molecule.force_field.citations[citation])
            LOGGER.info("Please cite: " + cite_string)
            header.append(cite_string)

        vermouth.gmx.itp.write_molecule_itp(meta_molecule.molecule, outfile,
                                            moltype=name, header=header)
    DeferredFileWriter().write()

    # Print molecule Log messages
    if meta_molecule.molecule.log_entries:
        print("")
    for loglevel, entries in meta_molecule.molecule.log_entries.items():
        for entry, fmt_args in entries.items():
            for fmt_arg in fmt_args:
                fmt_arg = {str(k): meta_molecule.molecule.nodes[v] for k, v in fmt_arg.items()}
                LOGGER.log(loglevel, entry, **fmt_arg, type='model')

# ducktape for renaming the itp tool
gen_itp = gen_params
