# Copyright 2022 University of Groningen
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

BASE_LIBRARY = {"DA": "DT", "DT": "DA", "DG": "DC", "DC": "DG",
                "DA5": "DT3", "DT5": "DA3", "DG5": "DC3", "DC5": "DG3",
                "DA3": "DT5", "DT3": "DA5", "DG3": "DC5", "DC3": "DG5"
                }

def complement_dsDNA(meta_molecule):
    """
    Given a meta-molecule, which represents the residue graph
    of a single strand of dsDNA add the complementray strand
    to get a meta-molecule with two disconnected strands. The
    update of the meta_molecule is done in place.

    By convention the other strand is generate 5'-> 3' end,
    meaning that the last residue of the first strand aligns
    with the first residue of the second strand.

    Parameters
    ----------
    meta_molecule: :class:`polyply.src.meta_molecule.MetaMolecule`

    Raises
    ------
    IOError
        when the resname does not match any of the know base-pair
        names an error is raised.
    """
    last_node = list(meta_molecule.nodes)[-1]
    resname = BASE_LIBRARY[meta_molecule.nodes[last_node]["resname"]]
    meta_molecule.add_monomer(last_node+1, resname, [])

    correspondance = {last_node: last_node+1}
    total = last_node+1
    for prev_node, next_node in nx.dfs_edges(meta_molecule, source=last_node):
        try:
            resname = BASE_LIBRARY[meta_molecule.nodes[next_node]["resname"]]
        except KeyError:
            msg = ("Trying to complete a dsDNA strand. However, resname { } with resid {} "
                   "does not match any of the known base-pair resnames. Note that polyply "
                   "at the moment only allows base-pair completion for molecules that only "
                   "consist of dsDNA. Please conact the developers if you whish to create a "
                   "more complicated molecule.")

            resname = meta_molecule.nodes[next_node]["resname"]
            resid = meta_molecule.nodes[next_node]["resid"]
            raise IOError(msg.format(resname, resid))

        meta_molecule.add_monomer(total+1, resname, [(correspondance[prev_node], total+1)])
        # make sure that edge attributes are carried over
        for attr, value in meta_molecule.edges[(prev_node, next_node)].items():
            meta_molecule.edges[(correspondance[prev_node], total+1)][attr] = value

        correspondance[next_node] = total + 1
        total += 1

    return meta_molecule

