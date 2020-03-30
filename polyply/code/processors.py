
class Map_to_molecule():


    def _get_links(self, link_name, length, attrs):
        links = []

        for link in self._force_field.links:
            if link.name == link_name:
                if length and length == len(link.nodes) and link.attributes_match(attrs):
                    links.append(link)
                elif link.attributes_match(attrs):
                    links.append(link)

        return links

     def map_to_molecule(self, mol=Vermouth.Molecule(nx.Graph())):
         new_mol = self._force_field.blocks[block_names[0]].to_molecule()
         new_mol._force_field = self._force_field
         new_mol.nrexcl = 1

         exclusions={}

         for node in self.nodes[1:]:
             resname = self.nodes[node]["resname"]
             new_mol.merge_molecule(self._force_field.blocks[resname])

         for edge in self.edges:
             link_name = self._get_edge_resname(edge)
             links = self._get_links(link_name, length=2, attrs={})
             for link in links:
                 new_mol = self.apply_link_for_residues(link, [edge[0], edge[1]] ,attrs={})

        return new_mol


class DoLinks(DoLinks):
      pass

class DoPTM(DoPTM):
      pass

