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
from difflib import SequenceMatcher
from itertools import permutations
import numpy as np
import networkx as nx
from .processor import Processor

BASE_LIBRARY = {"DA": "DT", "DT": "DA", "DG": "DC", "DC": "DG",
                "DA5": "DT5", "DT5": "DA5", "DG5": "DC5", "DC5": "DG5",
                "DA3": "DT3", "DT3": "DA3", "DG3": "DC3", "DC3": "DG3"
                }

class AnnotateDNA(Processor):
    """
    Given a topology and list indexes that indicate the molecules that
    contain DNA, the processor generates the appropriate meta_molecule
    structure. The DNA strands are identified using a greedy search method
    on the meta_molecule network and later combined with their complementary
    strand. The generated meta_molecule is later used for both building the
    DNA structure and backmapping to the desired forcefield.
    """

    def __init__(self, topology, includes_DNA):
        """
        Initalize the processor with a :class:`topology`
        and a list of DNA molecules in this topology.

        Parameters
        ------------
        topology: :class: `polyply.src.topology.Topology`
        includes_DNA: list[int]
            node keys list of the meta_molecules tagged as indluding DNA
        """
        self.topology = topology
        self.includes_DNA = [float(ndx) for ndx in includes_DNA]

    def _is_nucleobase(self, node_ndx, meta_molecule):
        """
        Determine whether a specific node in meta_molecule
        is a nucleobase.

        Parameters
        ------------
        node_ndx: int
            node index
        meta_molecule: :class:`polyply.src.MetaMolecule`

        Returns
        ---------
        bool:
            True if node is nucleobase, False otherwise
        """
        return meta_molecule.nodes[node_ndx]['resname'] in BASE_LIBRARY

    def _find_dna_strands(self, meta_molecule):
        """
        Extract the subgraphs of meta_molecule that are DNA strands

        Parameters
        ------------
        meta_molecule: :class:`polyply.src.MetaMolecule`

        Returns
        ---------
        strands: list[:class: `networkx.Graph`]
            list of DNA strands represented as subgraphs of meta_molecule
        """

        seperated_meta_mol = [meta_molecule.subgraph(graph) for graph
                              in nx.connected_components(meta_molecule)]

        # Find all strands by performing greedy search on
        # all connected_components of meta_molecule
        strands = []
        for meta_mol in seperated_meta_mol:
            queue = list(meta_mol.nodes.keys())[:1]
            visited = []
            strand = []
            while queue:
                # pop first node out of queue
                current_node = queue.pop()
                visited.append(current_node)

                if self._is_nucleobase(current_node, meta_molecule):
                    strand.append(current_node)
                elif strand:
                    strands.append(meta_mol.subgraph(strand))
                    strand = []

                # Determine neighbors nodes
                neighbors = [node for node in meta_molecule.neighbors(current_node) if node not in queue + visited]
                # update queue
                queue = neighbors + queue
                # Sort queue according to heuristic
                heuristic = lambda node: self._is_nucleobase(node, meta_mol)
                queue.sort(key=heuristic, reverse=True)

            if strand not in strands:
                strands.append(meta_molecule.subgraph(strand))
        return strands

    def _determine_sequence(self, strand):
        sequence = []
        if nx.cycle_basis(strand):
            start_base = list(strand.nodes.keys())[:1]
        else:
            condition = lambda node: strand.degree(node) == 1
            start_base = next(filter(condition, strand.nodes), None)
        for node_ndx in nx.dfs_tree(strand, start_base):
            sequence.append(strand.nodes[node_ndx]['resname'])
        return sequence

    def _match_ratio(self, ref_sequence, sequence):
        match_ratio = 0
        for ref_base, base in zip(ref_sequence, sequence):
            if BASE_LIBRARY[ref_base] == base:
                match_ratio += 1
        return match_ratio / len(ref_sequence)

    def _find_complementary_strands(self, ref_strand, strands):
        # Sequence we want to find match for
        ref_sequence = self._determine_sequence(ref_strand)

        # Determine all match_ratios with all sequences and reversed sequences
        match_ratios = {}
        for ndx, strand in enumerate(strands):
            sequence = self._determine_sequence(strand)
            # Determine match_ration sequence
            match_ratios[ndx, False] = self._match_ratio(ref_sequence, sequence)
            # Determine match_ration reverse sequence
            match_ratios[ndx, True] = self._match_ratio(ref_sequence, reversed(sequence))
        match, reverse =  max(match_ratios, key=match_ratios.get)
        match_strand = strands.pop(match)

        if reverse:
            mapping = dict(zip(match_strand, sorted(match_strand, reverse=True)))
            match_strand = nx.relabel_nodes(match_strand, mapping)

        return match_strand

    def _combine_complementary_strands(self, meta_molecule, strands):
        """
        Given a DNA molecule generate the appropriate meta_molecule used
        for the coordinate generation and backmapping. The constructed
        meta_molecule overwrites the one generated by polyply.

        Parameters
        ------------
        meta_molecule: :class:`polyply.src.MetaMolecule`
        strands: list[:class: `networkx.Graph`]
        """

        while len(strands) > 1:
            ref_strand = strands.pop()

            match_strand = self._find_complementary_strands(ref_strand, strands)

            for findex, bindex in zip(ref_strand, sorted(match_strand)):

                fbase = ref_strand.nodes[findex]
                bbase = match_strand.nodes[bindex]

                current_node = fbase

                # Combine attribute of strands
                current_node['resid'] = findex
                current_node['build'] = fbase['build'] | bbase['build']
                current_node['nnodes'] = fbase['nnodes'] + bbase['nnodes']
                current_node['nedges'] = fbase['nedges'] + bbase['nedges']
                current_node['graph'] = nx.compose(fbase['graph'], bbase['graph'])
                current_node['density'] = (fbase['density'] + bbase['density']) / 2
                current_node['resname'] = f"{fbase['resname']},{bbase['resname']}"

            meta_molecule.remove_nodes_from(match_strand)

    def run_molecule(self, meta_molecule, mol_idx):
        """
        Perform the meta_molecule generation for a single molecule.
        """
        strands = self._find_dna_strands(meta_molecule)
        self._combine_complementary_strands(meta_molecule, strands)
        return meta_molecule

    def run_system(self, system):
        """
        Process `system`.
        Parameters
        ----------
        system: vermouth.system.System
            The system to process. Is modified in-place.
        """
        mols = []
        for mol_idx, molecule in enumerate(system.molecules):
            if mol_idx in self.includes_DNA:
                mols.append(self.run_molecule(molecule, mol_idx))
        system.molecules = mols
