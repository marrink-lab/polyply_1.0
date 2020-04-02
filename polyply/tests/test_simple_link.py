import textwrap
import pytest
import numpy as np
import networkx as nx
import vermouth.forcefield
import vermouth.molecule
import polyply.src.meta_molecule
import polyply.src.map_to_molecule
import polyply.src.parsers
import polyply.src.apply_links
from polyply.src.meta_molecule import (MetaMolecule, Monomer)
from vermouth.molecule import Interaction

class TestApplyLinks:
    @staticmethod
    def test():
        lines="""
         [ moleculetype ]
         PS 1
         [ atoms ]
            1    STY            1  PS       R1       1     0.00000E+00   45
            2    STY            1  PS       R2       2     0.00000E+00   45
            3    STY            1  PS       R3       3     0.00000E+00   45
            4    SCY            1  PS       B        4     0.00000E+00   45
         [ bonds ]
            1     4   1     0.270000 8000
            4     5   1     0.270000 8000

         [ angles ]
           4     1     2    1   136  100
           4     1     3    1   136  100
           4     5     6    1   136  100
           4     5     7    1   136  100
           1     4     5    1   120   25
           4     5     8    1    52  550
        """
       # [ constraints ]
       #    2     3    1     0.270000
       #    3     1    1     0.270000
       #    1     2    1     0.270000
       # """
        mon=[Monomer(resname="PS",n_blocks=2)]
        interactions={"bonds":[
         Interaction(atoms=(0, 3), parameters=['1', '0.270000', '8000'], meta={}),
         Interaction(atoms=(3, 4), parameters=['1', '0.270000', '8000'], meta={'version': 1}),
         Interaction(atoms=(4, 7), parameters=['1', '0.270000', '8000'], meta={})],
        "angles":[
        Interaction(atoms=(0, 3, 4), parameters=['1', '120', '25'],  meta={"version":1}),
        Interaction(atoms=(3, 0, 1), parameters=['1', '136', '100'], meta={}),
        Interaction(atoms=(3, 0, 2), parameters=['1', '136', '100'], meta={}),
        Interaction(atoms=(3, 4, 5), parameters=['1', '136', '100'], meta={"version":1}),
        Interaction(atoms=(3, 4, 6), parameters=['1', '136', '100'], meta={"version":1}),
        Interaction(atoms=(3, 4, 7), parameters=['1', '52', '550'],  meta={"version":1}),
        Interaction(atoms=(7, 4, 5), parameters=['1', '136', '100'], meta={}),
        Interaction(atoms=(7, 4, 6), parameters=['1', '136', '100'], meta={})]}
#,
#        "constraints":[
#        Interaction(atoms=(0, 1), parameters=['1', '0.270000'], meta={}),
#        Interaction(atoms=(1, 2), parameters=['1', '0.270000'], meta={}),
#        Interaction(atoms=(2, 0), parameters=['1', '0.270000'], meta={}),
#        Interaction(atoms=(4, 5), parameters=['1', '0.270000'], meta={}),
#        Interaction(atoms=(5, 6), parameters=['1', '0.270000'], meta={}),
#        Interaction(atoms=(6, 4), parameters=['1', '0.270000'], meta={})],
#        "exclusions":[
#        Interaction(atoms=(0, 1), parameters=[], meta={}),
#        Interaction(atoms=(0, 2), parameters=[], meta={}),
#        Interaction(atoms=(0, 3), parameters=[], meta={}),
#        Interaction(atoms=(0, 4), parameters=[], meta={}),
#        Interaction(atoms=(0, 5), parameters=[], meta={}),
#        Interaction(atoms=(0, 6), parameters=[], meta={}),
#        Interaction(atoms=(0, 7), parameters=[], meta={}),
#        Interaction(atoms=(1, 2), parameters=[], meta={}),
#        Interaction(atoms=(1, 3), parameters=[], meta={}),
#        Interaction(atoms=(1, 4), parameters=[], meta={}),
#        Interaction(atoms=(2, 3), parameters=[], meta={}),
#        Interaction(atoms=(2, 4), parameters=[], meta={}),
#        Interaction(atoms=(3, 4), parameters=[], meta={}),
#        Interaction(atoms=(3, 5), parameters=[], meta={}),
#        Interaction(atoms=(3, 6), parameters=[], meta={}),
#        Interaction(atoms=(3, 7), parameters=[], meta={}),
#        Interaction(atoms=(4, 5), parameters=[], meta={}),
#        Interaction(atoms=(4, 6), parameters=[], meta={}),
#        Interaction(atoms=(4, 7), parameters=[], meta={}),
#        Interaction(atoms=(5, 6), parameters=[], meta={}),
#        Interaction(atoms=(5, 7), parameters=[], meta={}),
#        Interaction(atoms=(6, 7), parameters=[], meta={})]}

        lines = textwrap.dedent(lines).splitlines()
        ff = vermouth.forcefield.ForceField(name='test_ff')
        polyply.src.parsers.read_polyply(lines, ff)
        meta_mol = MetaMolecule.from_monomer_seq_linear(force_field=ff, monomers=mon, mol_name="test")
        new_meta_mol = polyply.src.map_to_molecule.MapToMolecule().run_molecule(meta_mol)
        new_meta_mol = polyply.src.apply_links.ApplyLinks().run_molecule(meta_mol)
       
        for angle in new_meta_mol.molecule.interactions['angles']:
            print(angle)

        for key in interactions:
            for interaction in interactions[key]:
                assert interaction in new_meta_mol.molecule.interactions[key]
