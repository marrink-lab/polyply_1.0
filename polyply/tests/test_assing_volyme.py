import polyply.src.AssignVolume

def test_from_itp():
   file_name = "test_data/itp/PEO.itp"
   edges = [(0,1),(1,2)]
   nodes = [0, 1, 2]
   attrs = {0: 'PEO', 1: 'PEO', 2: 'PEO'}

   ff = vermouth.forcefield.ForceField(name='test_ff')
   name = "PEO"
   meta_mol = MetaMolecule.from_itp(ff, file_name, name)
   
