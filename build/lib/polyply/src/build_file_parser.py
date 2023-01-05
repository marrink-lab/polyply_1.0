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
import vermouth
from vermouth.parser_utils import SectionLineParser
from vermouth.log_helpers import StyleAdapter, get_logger
from .generate_templates import map_from_CoG, compute_volume

LOGGER = StyleAdapter(get_logger(__name__))

PersistenceSpecs = namedtuple("persistence", ["model", "lp", "start", "stop", "mol_idxs"])

class BuildDirector(SectionLineParser):
    """
    Parser for the build file which dictates additional information
    about how to generate the system in the random-walk.
    """
    COMMENT_CHAR = ';'

    def __init__(self, molecules, topology):
        super().__init__()
        self.topology = topology
        self.molecules = molecules
        self.current_molidxs = []
        self.build_options = defaultdict(list)
        self.current_molname = None
        self.rw_options = {}
        self.persistence_length = {}
        self.templates = {}
        self.current_template = None

    @SectionLineParser.section_parser('molecule')
    def _molecule(self, line, lineno=0):
        """
        Parses the lines in the '[molecule]'
        directive and stores it.
        """
        tokens = line.split()
        self.current_molname = tokens[0]
        self.current_molidxs = np.arange(float(tokens[1]), float(tokens[2]), 1., dtype=int)
        for idx in self.current_molidxs:
            if idx not in self.topology.mol_idx_by_name[tokens[0]]:
                LOGGER.warning("parsing build file: could not find molecule with name {name} and index {index}.",
                              **{"index": idx, "name": tokens[0]})

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

    @SectionLineParser.section_parser('molecule', 'rw_restriction')
    def _rw_restriction(self, line, lineno=0):
        """
        Restrict random walk in specific direction.
        """
        tokens = line.split()
        geometry_def = {"resname": tokens[0],
                        "start": int(tokens[1]),
                        "stop":  int(tokens[2]),
                        "parameters": [np.array(tokens[3:6], dtype=float),
                                       float(tokens[6])]}

        for idx in self.current_molidxs:
            self.rw_options[(self.current_molname, idx)] = geometry_def

    @SectionLineParser.section_parser('molecule', 'distance_restraints')
    def _distance_restraints(self, line, lineno=0):
        """
        Node distance restraints.
        """
        tokens = line.split()
        nodes = tuple(map(int, tokens[:2]))
        dist = float(tokens[2])
        # parse tolerance if given
        if len(tokens) == 4:
            tol = float(tokens[3])
        else:
            tol = 0.0

        for idx in self.current_molidxs:
            msg = "Could not find atom {node} in molecule {molname} with index {idx}."
            if nodes[0] not in self.topology.molecules[idx]:
                raise IOError(msg.format(node=nodes[0], idx=idx, molname=self.current_molname))
            elif nodes[1] not in self.topology.molecules[idx]:
                raise IOError(msg.format(node=nodes[1], idx=idx, molname=self.current_molname))

            self.topology.distance_restraints[(self.current_molname, idx)][nodes] = (dist, tol)

    @SectionLineParser.section_parser('molecule', 'persistence_length')
    def _persistence_length(self, line, lineno=0):
        """
        Generate a distribution of end-to-end distance restraints based on a set
        persistence length.
        """
        tokens = line.split()
        model = tokens.pop(0)
        persistence_length = float(tokens.pop(0))
        start, stop = list(map(int, tokens))
        specs = PersistenceSpecs(*[model, persistence_length, start, stop, self.current_molidxs])
        self.topology.persistences.append(specs)

    @SectionLineParser.section_parser('template')
    def _template(self, line, lineno=0):
        """
        Parses the lines in the '[template]'
        directive and stores it. The line should be
        'resname ALA' for example.
        """
        # we only need the residue name
        name = line.split()[1]
        self.current_template = vermouth.molecule.Block()
        self.current_template.name = name

    @SectionLineParser.section_parser('template', 'atoms')
    def _template_atoms(self, line, lineno=0):
        """
        Parses the lines in the '[atoms]'
        directive of the '[template]' section
        """
        tokens = line.split()
        node_name, atype = tokens[0], tokens[1]
        position = np.array(tokens[2:], dtype=float)
        self.current_template.add_node(node_name,
                                       atype=atype,
                                       position=position)

    @SectionLineParser.section_parser('template', 'bonds')
    def _template_bonds(self, line, lineno=0):
        """
        Parses the lines in the '[bonds]'
        directive of the '[template]' section
        """
        tokens = line.split()
        nodeA = tokens[0]
        nodeB = tokens[1]
        self.current_template.add_edge(nodeA, nodeB)

    @SectionLineParser.section_parser('volumes')
    def _volume(self, line, lineno=0):
        """
        Parses the lines in the '[volumes]'
        directive and stores it.
        """
        resname, volume = line.split()
        self.topology.volumes[resname] = float(volume)

    def finalize_section(self, previous_section, ended_section):
        """
        Called once a section has finished. Here we perform all
        operations that are required when a section has ended.
        Here comes a list of end-section wrap ups:

        Templates
        ---------
        - compute volume from template if it is not defined yet
        - store coordinates as vectors from center of geometry
        """
        if previous_section == ["template", "bonds"]:
            coords = nx.get_node_attributes(self.current_template, "position")
            # if the volume is not defined yet compute the volume, this still
            # can be overwritten by an explicit volume directive later
            resname = self.current_template.name
            if resname not in self.topology.volumes:
                self.topology.volumes[resname] = compute_volume(self.current_template,
                                                                coords,
                                                                self.topology.nonbond_params,)
            # internally a template is defined as vectors from the
            # center of geometry
            mapped_coords = map_from_CoG(coords)
            self.templates[resname] = mapped_coords
            self.current_template = None

    def finalize(self, lineno=0):
        """
        Tag each molecule node with the chirality and build options
        if that molecule is mentioned in the build file by name.
        """
        for mol_idx, molecule in enumerate(self.molecules):

            if (molecule.mol_name, mol_idx) in self.build_options:
                for option in self.build_options[(molecule.mol_name, mol_idx)]:
                    self._tag_nodes(molecule, "restraints", option, molecule.mol_name)

            if (molecule.mol_name, mol_idx)  in self.rw_options:
                self._tag_nodes(molecule, "rw_options",
                                self.rw_options[(molecule.mol_name, mol_idx)],
                                molecule.mol_name)
            molecule.templates = self.templates

        super().finalize(lineno=lineno)

    @staticmethod
    def _tag_nodes(molecule, keyword, option, molname=""):
        resids = np.arange(option['start'], option['stop'], 1.)
        resname = option["resname"]
        for node in molecule.nodes:
            if molecule.nodes[node]["resid"] in resids\
            and molecule.nodes[node]["resname"] == resname:
                molecule.nodes[node][keyword] = molecule.nodes[node].get(keyword, []) +\
                                                [option['parameters']]
            # broadcast warning if we find the resid but it doesn't match the resname
            elif molecule.nodes[node]["resid"] in resids and not\
                 molecule.nodes[node]["resname"] == resname:
                 msg = "parsing build file: could not find resid {resid} with resname {resname} in molecule {molname}."
                 LOGGER.warning(msg, **{"resid": molecule.nodes[node]["resid"], "resname": resname,
                                          "molname": molname})

            # broadcast warning if we find the resname but it doesn't match the resid
            elif molecule.nodes[node]["resname"] == resname and not\
                 molecule.nodes[node]["resid"]:
                 msg = "parsing build file: could not find residue {resname} with resid {resid} in molecule {molname}."
                 LOGGER.warning(msg, **{"resid": molecule.nodes[node]["resid"], "resname": resname,
                                        "molname": molname})

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
