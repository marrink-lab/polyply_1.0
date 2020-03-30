    def from_json(cls, force_field, json_file, mol_name):
        """
        Constructs a :class::`MetaMolecule` from a json file
        format based graph.
        """
        meta_mol = cls(nx.Graph(), force_field=force_field, mol_name=mol_name)
        with open(json_file) as file_:
            data = json.load(file_)

        def recursive_grapher(_dict, current_node=0, graph=meta_mol, start=True):
            if not start:
               prev_node = current_node
               branch_node = current_node + 1

            for value in _dict.values():

                if start:
                   prev_node    = 0
                   current_node = 0
                   branch_node  = 0
                   start = False
                   edge = []
                else:
                   current_node += 1
                   edge = [(prev_node, current_node)]

                graph.add_monomer(current_node, value["resname"], edge)

                try:
                    current_node, graph = recursive_grapher(value["branches"],
                                                            current_node, graph, start)
                    prev_node = branch_node
                except KeyError:
                    prev_node = current_node
                    branch_node = current_node

            return current_node, graph

        return recursive_grapher(data, current_node=0, graph=meta_mol, start=True)[1]
