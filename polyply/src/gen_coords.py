"""
High level API for the polyply coordinate generator
"""
import argparse
from pathlib import Path
import vermouth
import vermouth.forcefield
import polyply
import polyply.src.parsers
from polyply import (DATA_PATH, MetaMolecule, ApplyLinks, Monomer, MapToMolecule)

def gen_coords(args):

    known_force_fields = vermouth.forcefield.find_force_fields(
        Path(DATA_PATH) / 'force_fields'
    )

    known_mappings = read_mapping_directory(Path(DATA_PATH) / 'mappings',
                                            known_force_fields)

    if args.lib:
        force_field = known_force_fields[args.lib]
    else:
        force_field = vermouth.forcefield.ForceField(name=args.name)

    if args.inpath:
        read_ff_from_file(args.inpath, force_field)

    read_mapping_directory("./", known_force_fields)


    meta_molecule = MetaMolecule.from_itp(args.inpath, force_field, args.name)
    meta_molecule.molecule = meta_molecule.force_field[args.name].to_molecule()
    AssignVolume.run_molecule(meta_molecule, vdwradii)
    RandomWalk.run_molecule(meta_molecule)


import vermouth.forcefield
from polyply.src.assing_volume import GenerateTemplates
from polyply.src.random_walk import RandomWalk
from polyply.src.backmap import Backmap
from polyply import MetaMolecule
from polyply.src.parsers import read_polyply
import time
import numpy as np
#FF = vermouth.forcefield.ForceField("/coarse/fabian/current-projects/polymer_itp_builder/polyply_2.0/polyply/data/force_fields/martini30b32")

FF = vermouth.forcefield.ForceField("test")

with open("PMMA.gromos.2016.itp", 'r') as _file:
      lines = _file.readlines()
      read_polyply(lines, FF)

meta_mol = MetaMolecule.from_itp(FF, "test.itp", "test")
GenerateTemplates().run_molecule(meta_mol)
RandomWalk().run_molecule(meta_mol)
Backmap().run_molecule(meta_mol)

#with open("cg.xyz", 'w') as _file:
#    _file.write('{}\n\n'.format(str(len(meta_mol.nodes))))
#    for node in meta_mol.nodes:
#        xyz = 10* meta_mol.nodes[node]["position"]
#        _file.write('{} {} {} {}\n'.format('B', xyz[0], xyz[1], xyz[2]))

def write_gro_file(meta_molecule, name, box):

    out_file = open(name, 'w')
    out_file.write('Monte Carlo generated PEO'+'\n')
    n = len(meta_mol.molecule.nodes)
    out_file.write('{:>3.3s}{:<8d}{}'.format('',n,'\n'))
    count = 0
    resnum = 1
    atomtype="BB"
    for xyz in meta_molecule.coords:
        resname = "PMA" #meta_mol.nodes[node]["resname"]
        resnum = count +1
        out_file.write('{:>5d}{:<5.5s}{:>5.5s}{:5d}{:8.3F}{:8.3F}{:8.3F}{}'.format(resnum, resname, atomtype, count, xyz[0], xyz[1], xyz[2],'\n'))
        count += 1

    out_file.write('{:>2s}{:<.5F} {:<.5F} {:<.5F}'.format('',float(box[0]), float(box[1]), float(box[2])))
    out_file.close()

write_gro_file(meta_mol, "cg_init.gro", np.array([20.,20.,20.]))
