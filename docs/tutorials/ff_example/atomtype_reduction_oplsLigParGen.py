#!/usr/bin/env python3

"""
atomtype_reduction.py

Use:
./atomtype_reduction.py -f PEO_n4/system.top
"""


import numpy as np
import argparse
import re
import sys
import src.itp_handling as itp_handling 

known_OPLS_atomtypes = {'opls_135' : ['3.50000E-01','2.76144E-01'], # CT - carbon   (1) aliphatic
                        'opls_002' : ['2.96000E-01','8.78640E-01'], #    - oxygen   (1) carbonyl
                        'opls_179' : ['2.90000E-01','5.85760E-01'], #    - oxygen   (2) ester
                        'opls_003' : ['3.25000E-01','7.11280E-01'], # N  - nitrogen (1) "standard" (nitroxide, aromatic)
                        'opls_761' : ['2.96000E-01','7.11280E-01'], # ON - oxygen   (3) nitroxide 
                        'opls_140' : ['2.50000E-01','1.25520E-01'], # HC - hydrogen (1) "standard"
                        'opls_145' : ['3.55000E-01','2.92880E-01'], # CA - carbon   (2) aromatic 
                        'opls_146' : ['2.42000E-01','1.25520E-01'], # HA - hydrogen (2) aromatic
                        'opls_154' : ['3.12000E-01','7.11280E-01'], # OH - oxygen   (4) alcohol
                        'opls_004' : ['0.00000E+00','0.00000E+00'], #    - hydrogen (3) alcohol
                        'opls_151' : ['3.40000E-01','1.25520E+00'], # Cl - chlorine
                        'opls_760' : ['3.25000E-01','5.02080E-01'], # NO - nitrogen (2) nitro group
                        'opls_812' : ['3.45000E-01','3.47272E-01'], #    - carbon   (3) 3-membered ring (e.g., oxiran) !! NOT in the oplsaa.ff/ffnonbonded.itp incl. in GMX !!
                        'opls_202' : ['3.60000E-01','1.48532E+00'], # S  - sulphur  (1) aromatic
                        'opls_349' : ['3.50000E-01','3.34720E-01'], # CB - carbon   (4) aromatic bonded to *only* other aromatics
                        'opls_150' : ['3.55000E-01','3.17984E-01'], # C= - carbon   (4) double-bond; also C# and CM
                        'opls_900' : ['3.30000E-01','7.11280E-01'], # NT - nitrogen (3) tertiary nitrogen
                        'opls_164' : ['2.94000E-01','2.55224E-01'], # F  - fluorine (1) most common parameters in oplsaa.ff
                        'opls_035' : ['3.55000E-01','1.04600E+00'], # S  - sulfur   (1) generic sulfur (most sulfur atoms have these LJ parameters in oplsaa.ff/ffnonbonded.itp)
                        'oplsFlpg' : ['2.90000E-01','2.51040E-01'], # F' - fluorine (2) fluorine parameters in LigParGen !! NOT in the oplsaa.ff/ffnonbonded.itp incl. in GMX !!
                        'oplsSIlpg': ['4.00000E-01','4.18400E-01']} # Si' - silicon (1) Silicon parameters (LigParGen) !! NOT in the oplsaa.ff/ffnonbonded.itp incl. in GMX !!


# Parse the arguments
parser = argparse.ArgumentParser(description='Read in charges from TOP file and prints them out along with their sum.')
parser.add_argument('-f', '--top-file', required=True, type=str  , help='name of TOP file')
args = parser.parse_args()
TOP_FILE    = args.top_file 
OUT_FILE    = 'system_OK.top' 
print(f"\nReading data from {TOP_FILE}.")


def replace_string_in_line(string_old, string_new, line):
    """
    Replaces "string_old" by "string_new" in that particular "line", i.e., works like UNIX's `sed` (on a single line).
    """
    return re.sub(string_old, string_new, line)


def find_key(input_dict, value_1, value_2):
    """
    https://stackoverflow.com/questions/16588328/return-key-by-value-in-dictionary
    """
    key_set = {k for k, v in input_dict.items() if (v[0] == value_1 and v[1] == value_2)}
    return list(key_set)


def map_LigParGen_to_known_atomtypes(itpfile):
    """
    Returns a dictionary that maps the atomtypes read from the LigParGen-generated provided system.top
    to the 'known_OPLS_atomtypes' dictionary defined at the top of this file.
    """
    with open(itpfile, "r") as inpfile:
        lines = inpfile.readlines()
        status = None
        old_to_new_atomtype_mapping = {}

        for line in lines:
            elements = line.split()
            if len(elements) != 0:
                if line[0] == "[":
                    status = itp_handling.check_itp_line(line)
                    continue
                elif status == 'atomtypes' and not elements[0] == ";" and not line[0] == '#':
                    LigParGen_atomtype_name = line.split()[0]
                    sig = line[45:80].split()[0]
                    eps = line[45:80].split()[1]
                    key = find_key(known_OPLS_atomtypes, sig, eps)

                    if len(key) == 1:
                        old_to_new_atomtype_mapping[LigParGen_atomtype_name] = key[0]
                    elif len(key) > 1:
                        sys.exit("Two matches found - why? Check the 'known_OPLS_atomtypes' dictionary.")
                    else:
                        sys.exit(f"NEW atomtype! {LigParGen_atomtype_name}, {sig}, {eps}; it needs to be added to the 'known_OPLS_atomtypes' dictionary.")
        return old_to_new_atomtype_mapping


print("INFO - step 1 - map LigParGen to known atomtypes")
old_to_new_atomtype_mapping = map_LigParGen_to_known_atomtypes(TOP_FILE)


print("INFO - step 2 - generate new [atoms] section")
with open(TOP_FILE, "r") as inpfile:

    lines = inpfile.readlines()
    status = None
    new_atom_lines = []

    for line in lines:
        elements = line.split()

        for LigParGen_atomtype_name, new_atomtype_name in old_to_new_atomtype_mapping.items():

            if len(elements) != 0:
                if line[0] == "[":
                    status = itp_handling.check_itp_line(line)
                    continue
                elif status == 'atoms' and not elements[0] == ";" and not line[0] == '#':
                    if LigParGen_atomtype_name in line:
                        newline = replace_string_in_line(LigParGen_atomtype_name, new_atomtype_name, line)
                        new_atom_lines.append(newline)
                elif status == 'bonds':
                    break


print("INFO - step 3 - write system_OK.top")
with open(TOP_FILE, "r") as inpfile, open(OUT_FILE, "w") as outfile:

    lines = inpfile.readlines()
    status = None
    atom_count = 0

    for line in lines:
        elements = line.split()
        if len(elements) != 0:
            if line[0] == "[":
                status = itp_handling.check_itp_line(line)
                outfile.write(line)
                continue
            elif status == 'atoms' and not elements[0] == ";" and not line[0] == '#':
                outfile.write(new_atom_lines[atom_count])
                atom_count = atom_count + 1 
            else:
                outfile.write(line)
        else:
            outfile.write(line)


