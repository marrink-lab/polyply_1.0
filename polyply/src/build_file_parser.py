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
from vermouth.parser_utils import SectionLineParser

class BuildDirector(SectionLineParser):

    COMMENT_CHAR = ';'

    def __init__(self, molecules):
        super().__init__()
        self.molecules = molecules
        self.build_options = defaultdict(list)
        self.current_molname = None

    @SectionLineParser.section_parser('molecule')
    def _molecule(self, line, lineno=0):
        """
        Parses the lines in the '[molecule]'
        directive and stores it.
        """
        self.current_molname = line.split()

    @SectionLineParser.section_parser('molecule', 'cylinder')
    def _cylinder(self, line, lineno=0):
        """
        Parses the lines in the '[cylinder]'
        directive and stores it.
        """
        tokens = line.split()
        geometry_def = self._base_parser_geometry(tokens)
        self.build_options[self.current_molname].append(('cylinder', geometry_def))

    @SectionLineParser.section_parser('molecule', 'sphere')
    def _sphere(self, line, lineno=0):
        """
        Parses the lines in the '[sphere]'
        directive and stores it.
        """
        tokens = line.split()
        geometry_def = self._base_parser_geometry(tokens)
        self.build_options[self.current_molname].append(('sphere', geometry_def))

    @SectionLineParser.section_parser('molecule', 'rectangle')
    def _rectangle(self, line, lineno=0):
        """
        Parses the lines in the '[rectangle]'
        directive and stores it.
        """
        tokens = line.split()
        geometry_def = self._base_parser_geometry(tokens)
        self.build_options[self.current_molname].append(('rectangle', geometry_def))

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
        for molecule in self.molecules:
            if molecule.mol_name in self.build_options:
                for _type, option in self.build_options[molecule.mol_name].items():
                    self._tag_nodes(molecule, _type, option)

        super().finalize

    @staticmethod
    def _tag_nodes(molecule, _type, option):
        resids = np.arange(option['start'], option['stop'], 1.)
        resname = option["resname"]
        for node in molecule.nodes:
            if molecule.nodes[node]["resid"] in resids\
            and molecule.nodes[node]["resname"] == resname:
                if "restraints" in molecule.nodes[node]:
                    molecule.nodes[node]["restraints"].append((_type, option["parameters"]))
                else:
                    molecule.nodes[node]["restraints"] = (_type, option["parameters"])

    @staticmethod
    def _base_parser_geometry(tokens):
        geometry_def = {}
        geometry_def["resname"] = tokens[0]
        geometry_def["start"] = float(tokens[1])
        geometry_def["stop"] = float(tokens[1])

        point = np.array([float(tokens[4]), float(tokens[5]), float(tokens[6])])
        parameters = [tokens[3], point]

        for param in tokens[3:]:
            parameters.append(float(param))

        geometry_def["parameters"] = parameters
        return geometry_def

def read_build_file(lines, molecules):
    """
    Parses `lines` of itp format and adds the
    molecule as a block to `force_field`.

    Parameters
    ----------
    lines: list
        list of lines of an itp file
    force_field: :class:`vermouth.forcefield.ForceField`
    """
    director = BuildDirector(molecules)
    return list(director.parse(iter(lines)))
