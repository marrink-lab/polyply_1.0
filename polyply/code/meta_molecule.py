from collections import namedtuple
import json
import networkx as nx


Monomer = namedtuple('Monomer', 'resname, n_blocks')

class MetaMolecule(nx.Graph):
    """
    Graph that describes molecules at the residue level.
    """

    def __init__(self, *args, **kwargs):
        self._force_field = kwargs.pop('force_field', None)
        super().__init__(*args, **kwargs)
        self.molecules = []

    def add_monomer(self, current, resname, connections):
        """
        This method adds a single node and an unlimeted number
        of edges to an instance of :class::`MetaMolecule`. Note
        that matches may only refer to already existing nodes.
        But connections can be an empty list.
        """
        self.add_node(current, attr_dict={"resname": resname})

        for edge in connections:
            if self.has_node(edge[0]) and self.has_node(edge[1]):
                self.add_edge(edge[0], edge[1])
            else:
                msg = ("Edge {} referes to nodes that currently do"
                       "not exist. Cannot add edge to unkown nodes.")
                raise IOError(msg.format(edge))

    def _get_edge_resname(self, edge):
        return self.nodes[edge[0]] + "_" + self.nodes[edge[1]]

    def _get_links(self, link_name, length, attrs):
        links = []

        for link in self._force_field.links:
            if link.name == link_name:
                if length and length == len(link.nodes) and link.attributes_match(attrs):
                    links.append(link)
                elif link.attributes_match(attrs):
                    links.append(link)

        return links

    @classmethod
    def from_linear(cls, force_field, monomers, mol_name):
        """
        Constructs a meta graph for a linear molecule
        which is the default assumption from
        """

        meta_mol_graph = cls(force_field=force_field, name=mol_name)
        res_count = 0

        for monomer in monomers:
            while trans <= monomer.n_blocks:

                if res_count != 0:
                    connect = [(res_count-1, res_count)]
                else:
                    connect = []
                trans += 1

                meta_mol_graph.add_monomer(res_count, monomer.name, connect)

        return meta_mol_graph

    @classmethod
    def from_json(cls, force_field, json_file, mol_name):
        """
        Constructs a :class::`MetaMolecule` from a json file
        format based graph.
        """
        meta_mol = cls(nx.Graph(), force_field=force_field, mol_name=mol_name)
        with open(json_file) as file_:
            data = json.load(file_)

        def recursive_grapher(_dict, current_node=0, graph=meta_mol):
            prev_node = current_node
            for value in _dict.values():
                graph.add_monomer(current_node, value["resname"], [(prev_node, current_node)])
                current_node += 1
                try:
                    current_node, graph = recursive_grapher(value["branches"],
                                                            current_node, graph)
                except KeyError:
                    continue

            return current_node, graph

        return recursive_grapher(data)
