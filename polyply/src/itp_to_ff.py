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
import numpy as np
import networkx as nx
try:
    import pysmiles
except ImportError:
    raise ImportError("To use polyply itp_to_ff you need to install pysmiles.")
import vermouth
from vermouth.forcefield import ForceField
from vermouth.gmx.itp_read import read_itp
from polyply.src.topology import Topology
from polyply.src.molecule_utils import extract_block, extract_links
from polyply.src.fragment_finder import FragmentFinder
from polyply.src.ffoutput import ForceFieldDirectiveWriter
from polyply.src.charges import balance_charges, set_charges

def itp_to_ff(itppath, fragment_smiles, resnames, term_prefix, outpath, charges=None):
    """
    Main executable for itp to ff tool.
    """
    # what charges belong to which resname
    if charges:
        crg_dict = dict(zip(resnames, charges))
    # read the topology file
    if itppath.suffix == ".top":
        top = Topology.from_gmx_topfile(itppath, name="test")
        mol = top.molecules[0].molecule
    # read itp file
    if itppath.suffix == ".itp":
        with open(itppath, "r") as _file:
            lines = _file.readlines()
        force_field = ForceField("tmp")
        read_itp(lines, force_field)
        block = next(iter(force_field.blocks.values()))
        mol = block.to_molecule()
        mol.make_edges_from_interaction_type(type_="bonds")

    # read the target fragments and convert to graph
    fragment_graphs = []
    for resname, smile in zip(resnames, fragment_smiles):
        fragment_graph = pysmiles.read_smiles(smile, explicit_hydrogen=True)
        nx.set_node_attributes(fragment_graph, resname, "resname")
        fragment_graphs.append(fragment_graph)

    # identify and extract all unique fragments
    unique_fragments, res_graph = FragmentFinder(mol, prefix=term_prefix).extract_unique_fragments(fragment_graphs)
    force_field = ForceField("new")
    for name, fragment in unique_fragments.items():
        new_block = extract_block(mol, list(fragment.nodes), defines={})
        nx.set_node_attributes(new_block, 1, "resid")
        new_block.nrexcl = mol.nrexcl
        force_field.blocks[name] = new_block
        set_charges(new_block, res_graph, name)
        if itppath.suffix == ".top":
            base_resname = name.split(term_prefix)[0].split('_')[0]
            print(base_resname)
            balance_charges(new_block,
                            topology=top,
                            charge=crg_dict[base_resname])

    force_field.links = extract_links(mol)

    with open(outpath, "w") as filehandle:
        ForceFieldDirectiveWriter(forcefield=force_field, stream=filehandle).write()
