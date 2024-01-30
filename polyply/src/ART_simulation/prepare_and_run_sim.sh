~/Desktop/gromacs-2022.5/build/bin/gmx grompp -f minimization.mdp -c ART.gro -p topol.top -o minim.tpr -maxwarn 10

~/Desktop/gromacs-2022.5/build/bin/gmx mdrun -deffnm minim -v -c minimized.gro



