# Copyright 2024 Dr. Fabian Gruenewald
#
# Licensed under the PolyForm Noncommercial License 1.0.0;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://polyformproject.org/licenses/noncommercial/1.0.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import networkx as nx
from vermouth.forcefield import ForceField
from vermouth.gmx.itp_read import read_itp
from polyply.src.meta_molecule import MetaMolecule
from polyply.src.topology import Topology
from polyply.src.molecule_utils import extract_block, extract_links, find_termini_mods
from polyply.src.fragment_finder import FragmentFinder
from polyply.src.ffoutput import ForceFieldDirectiveWriter
from polyply.src.charges import balance_charges, set_charges
from .load_library import load_ff_library

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

def gen_ff(itppath, smile_str, outpath, inpath=[], res_charges=None):
    """
    Main executable for itp to ff tool.
    """
    # load FF files if given
    if inpath:
        force_field = load_ff_library("new", None, inpath)
    # if none are given we create an empty ff
    else:
        force_field = ForceField("new")

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
    meta_mol = MetaMolecule.from_cgsmiles_str(force_field=force_field,
                                          mol_name="ref",
                                          cgsmiles_str=smile_str,
                                          seq_only=False,
                                          all_atom=True)

    # identify and extract all unique fragments
    unique_fragments, res_graph = FragmentFinder(target_mol).extract_unique_fragments(meta_mol.molecule)

    # extract the blocks with parameters
    for name, fragment in unique_fragments.items():
        # don't overwrite existing blocks
        if name in force_field.blocks:
            continue
        new_block = extract_block(target_mol, list(fragment.nodes), defines={})
        nx.set_node_attributes(new_block, 1, "resid")
        new_block.nrexcl = target_mol.nrexcl
        force_field.blocks[name] = new_block
        set_charges(new_block, res_graph, name)
        balance_charges(new_block,
                        topology=top,
                        charge=float(crg_dict[name]))

    # extract the regular links
    force_field.links += extract_links(target_mol)
    # extract links that span the terminii
    find_termini_mods(res_graph, target_mol, force_field)

    with open(outpath, "w") as filehandle:
        ForceFieldDirectiveWriter(forcefield=force_field, stream=filehandle).write()
