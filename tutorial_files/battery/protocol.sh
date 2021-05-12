#!/bin/bash

# the script will be run in its own directory
mkdir example_battery
cd example_battery

# 1 - Generate an itp file for PS-b-PEO

polyply gen_itp -lib martini3 -seq PS:62 PEO:166 -o PS_PEO.itp -name PS_PEO

# 2 - Generate the salt configureations inside the PEO domains
#     Note this step requires a build-file specifying the geometric
#     restrictions where to place the Salt and a top file only specifying
#     the salt

cat >> system.top << END
#include "../input/martini_v3.0.0.itp"
#include "../input/LiTFSI.itp"
#include "PS_PEO.itp"

[ system ]
test
[ molecules ]
LI 18200
TFSI 18200
END

polyply gen_coords -p system.top -b ../input/build_file.bld -o salts.gro -box 10.0 63.0 63.0 -nr 10 -name salts

# 3 - Next we grow the polymer around the salt starting on the domain bounderies.  To do that we need to specify 4 files:
#     a) the domain bounderies are specified as grid points stored in the file grid.dat. Users need to specify this customly.
#        the grid is provided via the -grid option
#     b) the build file specifies where in the box PEO and PS are allowed as done previously for the salt
#     c) the salt input configuration is provided via the '-c' option
#     d) the number of polymers has to be added to the topology file. A good number can typically be guessed based on the density
#        or by running iteratively serval systems until a good number is found. However, if the number is off it only means
#        the relaxation phase will be larger.

# let's add the number of polymers to the topology file
cat >> system.top << END
PS_PEO 1377
END

# finally when running the polyply command using the -start option we tell the program to start for the molecules with name PS_PEO with residue 63
# of name PEO. That is the PEO residue connecting to the PS part of the chains. The full command is:
polyply gen_coords -p system.top -b ../input/build_file.bld -c salts.gro -o polymers.gro -box 10 63 63 -name polymers -grid ../input/grid.dat -start PS_PEO-PEO#63 -nr 15

# 4 - Finally we run an energy minimization with flexible bonds.
gmx grompp -f ../input/min.mdp -c polymers.gro -p system.top
gmx mdrun -v

# 5 - Now we can start the equilbriations. First we run x ns keeping the z dimension fixed. This will allow the chains to obtain a more
#     compact packing in case we understimated the number of polymers per unit area
gmx grompp -f ../input/constz.mdp -c confout.gro -p system.top
gmx mdrun -v -s topol.tpr -deffnm constz
# 6 - After the intial relaxation of the xy dimnesion we switch to constant area to let z fluctuate, if needed. We run at constant area
#     because otherwise the simulation will try to minize the interface smearing out the simulation and eventually leading to a crash.
gmx grompp -f ../input/constant_area.mdp -c constz.gro -p system.top
gmx mdrun -v -s topol.tpr -deffnm const_area
