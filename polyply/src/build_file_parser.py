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
from collections import defaultdict, namedtuple
import numpy as np
import networkx as nx
from vermouth.parser_utils import SectionLineParser

def apply_node_distance_restraints(molecule, ref_node, target_node, distance):
    """
    Apply restraints to nodes.
    """
    for node in molecule.molecule.nodes:
        if node == target_node:
            graph_distance = 1.0
        else:
            graph_distance = nx.algorithms.shortest_path_length(molecule.molecule,
                                                                source=node,
                                                                target=target_node)
        ref_pos = ('node', ref_node)
        if 'restraint' in molecule.nodes[node]:
            molecule.nodes[node]['restraint'].append((graph_distance, ref_pos, distance))
        else:
            molecule.nodes[node]['restraint'] = [(graph_distance, ref_pos, distance)]

    return molecule

class BuildDirector(SectionLineParser):

    COMMENT_CHAR = ';'

    def __init__(self, molecules, topology):
        super().__init__()
        self.topology = topology
        self.molecules = molecules
        self.current_molidxs = []
        self.build_options = defaultdict(list)
        self.current_molname = None
        self.rw_options = dict()
        self.distance_restraints = defaultdict(dict)
        self.position_restraints = defaultdict(dict)
        self.presistence_length = {}

    @SectionLineParser.section_parser('molecule')
    def _molecule(self, line, lineno=0):
        """
        Parses the lines in the '[molecule]'
        directive and stores it.
        """
        tokens = line.split()
        self.current_molname = tokens[0]
        self.current_molidxs = np.arange(float(tokens[1]), float(tokens[2]), 1., dtype=int)

    @SectionLineParser.section_parser('molecule', 'cylinder', geom_type="cylinder")
    @SectionLineParser.section_parser('molecule', 'sphere', geom_type="sphere")
    @SectionLineParser.section_parser('molecule', 'rectangle', geom_type="rectangle")
    def _parse_geometry(self, line, lineno, geom_type):
        """
        Parses the lines in the '[<geom_type>]'
        directive and stores it. The format of the directive
        is as follows:

        <resname><resid_start><resid_stop><x><y><z><parameters>

        parameters depends on the geometry type and contains
        for:

        spheres - radius
        cylinder - radius and 1/2 d
        reactangle - 1/2 a, b, c which are the lengths
        """
        tokens = line.split()
        geometry_def = self._base_parser_geometry(tokens, geom_type)
        for idx in self.current_molidxs:
            self.build_options[(self.current_molname, idx)].append(geometry_def)

    # -> these should be named direction restraints
    @SectionLineParser.section_parser('molecule', 'rw_restriction')
    def _rw_restriction(self, line, lineno=0):
        """
        Restrict random walk in specific direction.
        """
        tokens = line.split()
        geometry_def = {"resname": tokens[0],
                        "start": int(tokens[1]),
                        "stop":  int(tokens[2]),
                        "parameters": [np.array([float(tokens[3]), float(tokens[4]), float(tokens[5])]),
                                       float(tokens[6])]}
        for idx in self.current_molidxs:
            self.rw_options[(self.current_molname, idx)] = geometry_def

    @SectionLineParser.section_parser('molecule', 'distance_restraints')
    def _distance_restraints(self, line, lineno=0):
        """
        Node distance restraints.
        """
        tokens = line.split()
        for idx in self.current_molidxs:
            self.distance_restraints[(self.current_molname, idx)][(int(tokens[0]), int(tokens[1]))] = float(tokens[2])

    @SectionLineParser.section_parser('molecule', 'position_restraints')
    def _position_restraints(self, line, lineno=0):
        """
        Node distance restraints.
        """
        tokens = line.split()
        target_node = tokens[0]
        ref_position = np.array(list(map(float, tokens[0:2])))

        for idx in self.current_molidxs:
            self.position_restraints[(self.current_molname, idx)][target_node] = ref_position

    @SectionLineParser.section_parser('molecule', 'presistence_length')
    def _presistence_length(self, line, lineno=0):
        """
        Generate a distribution of end-to-end distance restraints based on a set
        presistence length.
        """
        tokens = line.split()
        model = tokens.pop(0)
        lp = float(tokens.pop(0))
        start, stop = list(map(int, tokens))
        Presistence_specs = namedtuple("presist", ["model", "lp", "start", "stop", "mol_idxs"])
        specs = Presistence_specs(*[model, lp, start, stop, self.current_molidxs])
        self.topology.presistences.append(specs)

    @SectionLineParser.section_parser('molecule', 'chiral')
    def _chiral(self, line, lineno=0):
        """
        Parses the lines in the '[chiral]'
        directive and stores it.
        """
        pass

    @SectionLineParser.section_parser('molecule', 'isomer')
    def _isomer(self, line, lineno=0):
        """
        Parses the lines in the '[isomer]'
        directive and stores it.
        """
        pass

    def finalize(self, lineno=0):
        """
        Tag each molecule node with the chirality and build options
        if that molecule is mentioned in the build file by name.
        """
        for mol_idx, molecule in enumerate(self.molecules):

            if (molecule.mol_name, mol_idx) in self.build_options:
                for option in self.build_options[(molecule.mol_name, mol_idx)]:
                    self._tag_nodes(molecule, "restraints", option)

            if (molecule.mol_name, mol_idx)  in self.rw_options:
                self._tag_nodes(molecule, "rw_options",
                                self.rw_options[(molecule.mol_name, mol_idx)])

            if (molecule.mol_name, mol_idx) in self.distance_restraints:
                for ref_node, target_node in self.distance_restraints[(molecule.mol_name, mol_idx)]:
                    distance = self.distance_restraints[(molecule.mol_name, mol_idx)][(ref_node, target_node)]
                    molecule = apply_node_distance_restraints(molecule, ref_node, target_node, distance)

            if (molecule.mol_name, mol_idx) in self.presistence_length:
                specs = self.presistence_length[(molecule.mol_name, mol_idx)]
                molecule.meta.update({"presistence_length": specs})

        super().finalize

    @staticmethod
    def _tag_nodes(molecule, keyword, option):
        resids = np.arange(option['start'], option['stop'], 1.)
        resname = option["resname"]
        for node in molecule.nodes:
            if molecule.nodes[node]["resid"] in resids\
            and molecule.nodes[node]["resname"] == resname:
                molecule.nodes[node][keyword] = molecule.nodes[node].get(keyword, []) +\
                                                [option['parameters']]

    @staticmethod
    def _base_parser_geometry(tokens, _type):
        geometry_def = {}
        geometry_def["resname"] = tokens[0]
        geometry_def["start"] = float(tokens[1])
        geometry_def["stop"] = float(tokens[2])

        point = np.array([float(tokens[4]), float(tokens[5]), float(tokens[6])])
        parameters = [tokens[3], point]

        for param in tokens[7:]:
            parameters.append(float(param))

        parameters.append(_type)
        geometry_def["parameters"] = parameters
        return geometry_def

def read_build_file(lines, molecules, topology):
    """
    Parses `lines` of itp format and adds the
    molecule as a block to `force_field`.

    Parameters
    ----------
    lines: list
        list of lines of an itp file
    force_field: :class:`vermouth.forcefield.ForceField`
    """
    director = BuildDirector(molecules, topology)
    return list(director.parse(iter(lines)))
