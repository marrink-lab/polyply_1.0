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
from vermouth.graph_utils import (make_residue_graph,
                                  collect_residues)
from polyply.src.processor import Processor
from polyply.src.graph_utils import is_branched

def tag_exclusions(blocks, force_field):
    """
    Given the names of some `blocks` check if the
    corresponding molecules in `force_field` have
    the same number of default exclusions. If not
    find the minimum number of exclusions and tag all
    nodes with the original exclusion number. Then
    change the exclusion number.

    Note the tag is picked up by apply links where
    the excluions are generated.
    """
    excls = [force_field.blocks[mol].nrexcl for mol in blocks]

    if len(set(excls)) > 1:
        min_excl = min(excls)
        for excl, mol in zip(excls, blocks):
            block = force_field.blocks[mol]
            nx.set_node_attributes(block, excl, "exclude")
            block.nrexcl = min_excl

class MapToMolecule(Processor):
    """
    This processor takes a :class:`MetaMolecule` and generates a
    :class:`vermouth.molecule.Molecule`, which consists at this stage
    of disconnected blocks. These blocks can be connected using the
    :class:`ConnectMolecule` processor. It can either run on a
    single meta molecule or a system. The later is currently not
    implemented.
    """
    @staticmethod
    def expand_meta_graph(meta_molecule, block, meta_mol_node):
        """
        When a multiresidue block is encounterd the individual
        residues are added instead of the orignial block to
        the meta molecule.
        """
        # 0. make edges for the block
        for inter_type in ["bonds", "constraints", "virtual_sitesn",
                           "virtual_sites2", "virtual_sites3", "virtual_sites4"]:
            block.make_edges_from_interaction_type(inter_type)

        # 1. make residue graph and check it is not branched
        # we cannot do branched block expansion
        expanded_graph = make_residue_graph(block)

        # 2. clear out all old edges
        old_edges = list(meta_molecule.edges(meta_mol_node))
        meta_molecule.remove_edges_from(old_edges)

        # 3. relable nodes to make space for new nodes to be inserted
        mapping = {}
        offset = len(expanded_graph.nodes) - 1
        for node in meta_molecule.nodes:
            if node > meta_mol_node:
                mapping[node] = node + offset

        nx.relabel_nodes(meta_molecule, mapping, copy=False)
        # 4. add the new nodes to the meta molecule overwriting
        # the inital nodes inserting the graph
        nodes = sorted(expanded_graph.nodes)
        for node in nodes:
            node_key = node + meta_mol_node
            attrs = expanded_graph.nodes[node]
            attrs["links"] = False
            meta_molecule.add_node(node_key, **attrs)

        # 5. add all edges within the exapnded block
        for edge in expanded_graph.edges:
            meta_molecule.add_edge(edge[0] + meta_mol_node, edge[1] + meta_mol_node)

    def add_blocks(self, meta_molecule):
        """
        Add disconnected blocks to :class:`vermouth.molecule.Molecue`
        and if a multiresidue block is encountered expand the meta
        molecule graph to include the block at residue level.
        """
        force_field = meta_molecule.force_field
        resnames = set(nx.get_node_attributes(meta_molecule, "resname").values())
        tag_exclusions(resnames, force_field)

        block = force_field.blocks[meta_molecule.nodes[0]["resname"]]
        new_mol = block.to_molecule()
        # we store the block together with the residue node
        meta_molecule.nodes[0]["graph"] = new_mol.copy()
        # here we generate a residue dict which is a collection
        # of all unique residue regardless of connectivity
        # it is faster than making the complete residue graph
        res_dict = collect_residues(block, attrs=('resid', 'resname'))
        if len(res_dict) > 1:
            self.expand_meta_graph(meta_molecule, block, 0)

        node_keys = list(meta_molecule.nodes.keys())
        node_keys.sort()
        for node in node_keys[1:]:
            resname = meta_molecule.nodes[node]["resname"]

            if node + 1 in nx.get_node_attributes(new_mol, "resid").values():
                continue
            block = force_field.blocks[resname]
            correspondence = new_mol.merge_molecule(block)

            residue = nx.Graph()
            for res_node in correspondence.values():
                data = new_mol.nodes[res_node]
                residue.add_node(res_node, **data)

            meta_molecule.nodes[node]["graph"] = residue

            # here we generate a residue dict which is a collection
            # of all unique residue regardless of connectivity
            # it is faster than making the complete residue graph
            res_dict = collect_residues(block, attrs=('resid', 'resname'))
            if len(res_dict) > 1:
                self.expand_meta_graph(meta_molecule, block, node)

        return new_mol

    def run_molecule(self, meta_molecule):
        """
        Process a single molecule. Must be implemented by subclasses.
        Parameters
        ----------
        molecule: polyply.src.meta_molecule.MetaMolecule
             The meta molecule to process.
        Returns
        -------
        vermouth.molecule.Molecule
            Either the provided molecule, or a brand new one.
        """
        new_molecule = self.add_blocks(meta_molecule)
        meta_molecule.molecule = new_molecule
        return meta_molecule
