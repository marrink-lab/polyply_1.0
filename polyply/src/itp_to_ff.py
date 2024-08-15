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
import networkx as nx
from vermouth.forcefield import ForceField
from vermouth.gmx.itp_read import read_itp
from polyply.src.topology import Topology
from polyply.src.molecule_utils import extract_block, extract_links, find_termini_mods
from polyply.src.fragment_finder import FragmentFinder
from polyply.src.ffoutput import ForceFieldDirectiveWriter
from polyply.src.charges import balance_charges, set_charges
from polyply.src.big_smile_mol_processor import DefBigSmileParser

def _read_itp_file(itppath):
    """
    small wrapper for reading itps
    """
    with open(itppath, "r") as _file:
        lines = _file.readlines()
    force_field = ForceField("tmp")
    read_itp(lines, force_field)
    block = next(iter(force_field.blocks.values()))
    mol = block.to_molecule()
    mol.make_edges_from_interaction_type(type_="bonds")
    return mol

def itp_to_ff(itppath, smile_str, outpath, res_charges=None):
    """
    Main executable for itp to ff tool.
    """
    # what charges belong to which resname
    if res_charges:
        crg_dict = dict(res_charges)

    # read the topology file
    if itppath.suffix == ".top":
        top = Topology.from_gmx_topfile(itppath, name="test")
        target_mol = top.molecules[0].molecule
    # read itp file
    elif itppath.suffix == ".itp":
        top = None
        target_mol = _read_itp_file(itppath)

    # read the big-smile representation
    meta_mol = DefBigSmileParser().parse(smile_str)

    # identify and extract all unique fragments
    unique_fragments, res_graph = FragmentFinder(target_mol).extract_unique_fragments(meta_mol.molecule)

    # extract the blocks with parameters
    force_field = ForceField("new")
    for name, fragment in unique_fragments.items():
        new_block = extract_block(target_mol, list(fragment.nodes), defines={})
        nx.set_node_attributes(new_block, 1, "resid")
        new_block.nrexcl = target_mol.nrexcl
        force_field.blocks[name] = new_block
        set_charges(new_block, res_graph, name)
        balance_charges(new_block,
                        topology=top,
                        charge=crg_dict[name])

    # extract the regular links
    force_field.links = extract_links(target_mol)
    # extract links that span the terminii
    find_termini_mods(res_graph, target_mol, force_field)

    with open(outpath, "w") as filehandle:
        ForceFieldDirectiveWriter(forcefield=force_field, stream=filehandle).write()
