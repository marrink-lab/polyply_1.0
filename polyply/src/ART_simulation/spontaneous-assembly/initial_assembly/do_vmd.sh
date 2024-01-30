#!/bin/bash 

shopt -s expand_aliases
#source ~/.bashrc
#source /gromacs/gromacs-2021.1/bin/GMXRC

rm -f dppc-md-start.gro dppc-md-viz.xtc

echo 0 | gmx trjconv -s dppc-md.tpr -f dppc-md.xtc -pbc whole -dump 0 -o dppc-md-start.gro
echo 0 | gmx trjconv -s dppc-md.tpr -f dppc-md.xtc -pbc whole -o dppc-md-viz.xtc

vmd -e viz.vmd

exit
