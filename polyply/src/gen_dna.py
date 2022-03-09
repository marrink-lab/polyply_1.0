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

    Parameters
    ----------
    meta_molecule: :class:`polyply.src.meta_molecule.MetaMolecule`

    Raises
    ------
    IOError
        when the resname does not match any of the know base-pair
        names an error is raised.
    """
    incr = meta_molecule.number_of_nodes()
    first_node = 0
    resname = BASE_LIBRARY[meta_molecule.nodes[0]["resname"]]
    meta_molecule.add_monomer(first_node+incr, resname, [])
    old_edges = list(meta_molecule.edges)

    for ndx, jdx in old_edges:
        try:
            resname = BASE_LIBRARY[meta_molecule.nodes[jdx]["resname"]]
        except KeyError:
            msg = ("Trying to complete a dsDNA strand. However, resname { } with resid {} "
                   "does not match any of the known base-pair resnames. Note that polyply "
                   "at the moment only allows base-pair completion for molecules that only "
                   "consist of dsDNA. Please conact the developers if you whish to create a "
                   "more complicated molecule.")
            resname = meta_molecule.nodes[jdx]["resname"]
            resid = meta_molecule.nodes[jdx]["resid"]
            raise IOError(msg.format(resname, resid))
        meta_molecule.add_monomer(jdx+incr, resname, [(ndx+incr, jdx+incr)])
        # make sure that edge attributes are carried over
        for attr, value in meta_molecule.edges[(ndx, jdx)].items():
            meta_molecule.edges[(ndx+incr, jdx+incr)][attr] = value

    return meta_molecule

