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

class ForceFieldDirectiveWriter():
    """
    Write force-field files according to the
    vermouth force-field definition.

    Note that this is a leightweight writer
    which does not offer the complete rich
    syntax of the ff file format.
    """
    def __init__(self, forcefield, stream):
        """
        Parameters
        ----------
        forcefield: `:class:vermouth.forcefield.ForceField`
            the force-field object to write

        stream: ``
            the stream to which to write; must have a write method
        """
        self.forcefield = forcefield
        self.stream = stream
        # these attributes have a specific order in the moleculetype section
        self.normal_order_block_atoms = ["atype", "resid", "resname",
                                         "atomname", "charge_group", "charge", "mass"]

    def write(self):
        """
        Write the forcefield to file.
        """
        for name, block in self.forcefield.blocks.items():
            self.stream.write("[ moleculetype ]\n")
            excl = str(block.nrexcl)
            self.stream.write(f"{name} {excl}\n")
            self.write_atoms_block(block.nodes(data=True))
            self.write_interaction_dict(block.interactions)

        for link in self.forcefield.links:
            self.write_link_header()
            self.write_atoms_link(link.nodes(data=True))
            self.write_interaction_dict(link.interactions)
            self.write_edges(link.edges)

    def write_interaction_dict(self, inter_dict):
        """
        Writes interactions to `self.stream`, with a new
        interaction directive per type. Meta attributes
        are kept and written as json parasable dicts.

        Parameters
        ----------
        inter_dict: `class:dict[list[vermouth.molecule.Interaction]]`
            the interaction dict to write
        """
        for inter_type in inter_dict:
            self.stream.write(f"[ {inter_type} ]\n")
            for interaction in inter_dict[inter_type]:
                atom_string = " ".join(interaction.atoms)
                param_string = " ".join(interaction.parameters)
                meta_string = "{" + " ,".join([f"\"{key}\": \"{value}\"" for key, value in interaction.meta.items()]) + "}"
                line = atom_string + " " + param_string + " " + meta_string + "\n"
                self.stream.write(line)

    def write_edges(self, edges):
        """
        Writes edges to `self.stream` into the edges directive.

        Parameters
        ----------
        edges: abc.iteratable
            pair-wise iteratable edge list
        """
        self.stream.write("[ edges ]\n")
        for idx, jdx in edges:
            self.stream.write(f"{idx} {jdx}\n")

    def write_atoms_block(self, nodes):
        """
        Writes the nodes/atoms of the block atomtype directive to `self.stream`.
        All attributes are written following the GROMACS atomtype directive
        style.

        Parameters
        ----------
        edges: abc.iteratable
            pair-wise iteratable edge list
        """
        self.stream.write("[ atoms ]\n")
        for idx, (node, attrs) in enumerate(nodes):
            idx += 1
            attr_line = " ".join([str(attrs[attr]) for attr in self.normal_order_block_atoms ])
            line = f"{idx} " + attr_line + "\n"
            self.stream.write(line)

    def write_atoms_link(self, nodes):
        """
        Writes the nodes/atoms of the link atomtype directive to `self.stream`.
        All attributes are written as json style dicts.

        Parameters:
        -----------
        nodes: abc.itertable[tuple(abc.hashable, dict)]
            list of nodes in form of a list with hashable node-key and dict
            of attributes. The format is the same as returned by networkx.nodes(data=True)
        """
        self.stream.write("[ atoms ]\n")
        for node_key, attributes  in nodes:
            attr_line = " {" + " ,".join([f"\"{key}\": \"{value}\"" for key, value in attributes.items()]) + "}"
            line = str(node_key) + attr_line + "\n"
            self.stream.write(line)

    def write_link_header(self):
        """
        Write the link directive header, with the resnames written
        in form readable to geenerate a `:class:vermouth.molecule.Choice`
        object.

        Prameters
        ---------
        resnames: `abc.itertable[str]`
        """
        self.stream.write("[ link ]\n")
