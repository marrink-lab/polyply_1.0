#!/usr/bin/env python3

import sys

def check_itp_line(line):
    """
    Checks which Gromacs section of the itp file we are at, returning a 'status' accordingly.

    Parameters
    ----------
    line: string
        A line of the itp file received as input file.

    Returns
    --------
    status: string
        A string which tells in which section of the itp we are. 
    """
    if 'defaults' in line:
        return 'defaults'
    elif 'atomtypes' in line:
        return 'atomtypes'
    elif 'system' in line:
        return 'system'
    elif 'molecules' in line:
        return 'molecules'
    elif 'moleculetype' in line:
        return 'mtype'
    elif 'atoms' in line:
        return 'atoms'
    elif 'virtual_sitesn' in line:
        return 'virtual_sitesn'
    elif 'bonds' in line:
        return 'bonds'
    elif 'constraints' in line:
        return 'constraints'
    elif 'pairs' in line:
        return 'pairs'
    elif 'angles' in line:
        return 'angles'
    elif 'exclusions' in line:
        return 'exclusions'
    elif 'dihedrals' in line:
        return 'dihedrals'
    elif 'link' in line:
        return 'link'
    else:
        sys.exit('! ERROR ! Something is wrong. Just found the following line in the input itp file: ' + line)

