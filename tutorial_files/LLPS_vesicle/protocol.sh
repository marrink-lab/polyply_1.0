#!/bin/bash

mkdir tutorial
cd tutorial

# 1 - first we generate itp files for the PEGylated lipids. To that end we have to extent
#     the corresponding Martini lipid by a PEG tail. This is done by calling polyply as follows
polyply gen_itp -lib martini3 -f ../input/lipids.itp -name PEL -seq POPE:1 PEO:45 -o PEL.itp

# 2 - next we genreate the itp file for PEO
polyply gen_itp -lib martini3 -seq PEO:145 -name PEO -o PEO.itp

# 3 - finally we have to generate input files for our dextran composition
#    the script random_lin_comb.py in the input folder will generate a
#    number of residue graphs corresponding to 5% chance of forming a branch
#    with the number of monomers drawn from a random normal distribution. To generate
#    a single dextran only 1 graph has to be provided.

mkdir dextran_graphs
mkdir dextran_itps
python3 ../input/random_lin_comb.py "dextran_graphs/" 451

# for each graph we have to make an input file, so we run polyply in a loop
count=0
for graph in dextran_graphs/*
do
polyply gen_itp -lib martini3 -seqf ${graph} -o dextran_itps/dextran${count}.itp -name dex${count}
let "count = count + 1"
done

# because we don't want to include every itp file we combine all itps into a single file
touch dextrans.itp
for itp in dextran_itps/*
do
cat ${itp} >> dextrans.itp
done

# now that we have all the input files we make a topology file
cat >> system.top << END
#include "../../martini_v3_parameters/martini_v3.0.0.itp"
#include "../../martini_v3_parameters/martini_v3.0.0_ions_v1.itp"
#include "../input/lipids.itp"
#include "dextrans.itp"
#include "PEO.itp"
#include "PEL.itp"

[ system ]
test
[ molecules ]
DOPC 4470
DPPC 4470
CHOL 3944
PEL  262
DOPC 3287
DPPC 3287
CHOL 2900
PEL  193
PEO  620
END

# this command adds a line for all different dextran itps
cp -f ../input/build_file.bld ./
for count in {0..450}
do
echo dex${count} 1 >> system.top
let "molidx = count + 23433"
let "molidx_stop = count + 23433 + 1"

cat >> build_file.bld << END
[ molecule ]
; molname
dex${count} ${molidx} ${molidx_stop}
[ sphere ]
; resname start stop  inside-out  x  y  z    r
 GLC     1    600    in        32.5 32.5  32.5   22.5
[ rectangle ]
; resname start stop  inside-out  x  y  z            a   b    c
 GLC     1    600    out        7.0 32.25 32.25   23.5 25.5 25.5
END

done

# finally we add the salt
cat >> system.top << END
NA  262
NA  193
END

# 4 - now that everything is in place we call the structure generation command and provide 3 input files
#     a) the topology file
#     b) a build file specifying the restrictions on where to place the polymers inside the vesicle
#     c) the strarting coordinates of the vesicle
#     now the only thing we also need to tell polyply is that it does not have to rebuild the resdiues
#     corresponding to the lipids. This is done by setting
polyply gen_coords -p system.top -c ../input/vesicle.gro -res PEO W GLC -o start.gro -name test -box 65 65 65 -b build_file.bld -split POPE:HEAD-NH3:TAIL-PO4,GL1,GL2,C1A,D2A,C3A,C4A,C1B,C2B,C3B,C4B

../input/SOL -in start.gro -o solvated.gro -tem ../input/W.gro
cat info.txt >> system.top

gmx grompp -f ../input/min_TI.mdp -c solvated.gro -p system.top -maxwarn 1
gmx mdrun -v -nsteps 50

gmx grompp -f ../input/min.mdp -c confout.gro -p system.top -maxwarn 1
gmx mdrun -v
