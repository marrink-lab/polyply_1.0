# Tutorial: Single-Stranded (Circular) DNA

## 1 - Introduction

Polyply can be used to generate input parameters and coordinates for all sorts of polymers.
When building coordinates it is also possible to fine-tune some of the conformational
properties. For example, the persistence length captures the stiffness of a polymer. Polymers
with higher persistence length are on average straighter. One such polymer with high
persistence length is DNA. DNA, whether single-stranded (ss) or double-stranded (ds), has a
much higher persistence length than many of the famous synthetic polymers. For instance,
ss-DNA has a persistence length that is typically 10 times higher than that of, for example,
PEG and even about 4 times higher than PS, which is typically considered a stiff polymer
[1,2].

To make matters even more complicated, the persistence length and hence the conformation of
the DNA depends on the salt concentration of the solution. At 12.5 mM salt the persistence
length of a single strand of polyT is about 3.2 nm, whereas at a concentration of 1 M the
persistence length decreases to about 1.0 nm [1]. A similar effect has been reported recently
for RNA [3].

In this tutorial we will learn how to generate input files and coordinates for all-atom ssDNA
using the Parmbsc1 force-field [4] and for coarse-grained DNA using the Martini 2 force-field
[5]. In addition, it will be shown how to include the persistence length to generate initial
chain coordinates which correspond to specific salt concentrations, as well as how to generate
circular ssDNA.

!!! tip
    For some tips and tricks make sure to check out the [Useful Notes](#5-useful-notes)
    section.

!!! note
    This tutorial currently requires the latest polyply version from GitHub (1.2.2.dev4) in
    combination with the latest vermouth library version (0.7.3.dev34).

## 2 - Generating a simple itp file

First let us generate an itp file for the DNA strand we want to study. Parameters for the
all-atom Parmbsc1 force-field [4] are part of the polyply library. Thus we only need the DNA
sequence. Let's start with a polyT chain of 10 residues. To generate the parameters (itp
file) we simply run a command as follows:

```bash
polyply gen_params -lib parmbsc1 -o polyT.itp -name ssDNA -seq DT5:1 DT:10 DT3:1
```

In the previous command we specify the library we want to use (`-lib parmbsc1`), the name of
the output file with the `-o` flag, as well as a name of the molecule with `-name`. Finally,
the `-seq` option specifies the number of polyT residues that we want to have, as well as the
5' and 3' end of the polyT residue. Note that these should always be specified in order to
have the proper ends of the DNA.

DNA residues follow the naming scheme DA, DT, DC, and DG for single-stranded DNA residues and
are appended with a 3 or 5 to specify the corresponding 3'/5' end of the strand. By definition
the sequence always starts with the 3' residue and ends with the 5' residue. Exchanging the
two will result in a disconnected DNA strand and a warning being issued. Of course we can also
specify more complex sequences as shown in section 4. But first we will generate some starting
coordinates.

## 3 - Generating starting coordinates

Now we are in the position to generate coordinates for the ssDNA strand. To do so we first
have to get the force-field parameters, which are usually available
[here](https://www.gromacs.org/Downloads/User_contributions/Force_fields). However, the site
is currently unavailable. To be able to continue, download the parameters from our
[repository copy](https://github.com/marrink-lab/polyply_regression_tests/tree/main/polyply_regression_tests/data/force_fields/amber14sb_parmbsc1.ff).
Next we will have to write a topology file that contains all the system components that we
need. In this case we need the DNA, water, and some ions corresponding to the salt
concentration. Note that we will generate not only 1 but 5 copies of the small polyT strand.

```
#include "amber14sb_parmbsc1.ff/forcefield.itp"
#include "amber14sb_parmbsc1.ff/ions.itp"
#include "amber14sb_parmbsc1.ff/tip3p.itp"
#include "polyT.itp"
[ system ]
DNA in water
[molecules]
ssDNA 5
SOL 25000
NA xxx
CL xxx
```

Following this we replace the `xxx` by the number of ion molecules corresponding to the target
salt concentration. Make sure to save a separate topology file for each example.

### 3.1 - Starting coordinates for low salt concentration

We take a low salt concentration of 125 mM. Considering the concentration of water in water
(see Notes for the detailed calculation), with the number of waters as given above, and the
fact that each polyT strand has a charge of about -10, we have to add about 57 ions of each
species — so 57 NA ions and 7 CL ions to keep the solution close to neutral. Edit the topology
file with the corresponding number of ions and save it as `low_conc.top`. Next, we know that
the persistence length of DNA at this concentration is about 3 nm [2]. To specify that our DNA
has to obey this value we need to write a build file. The file looks as shown below:

```
[ molecule ]
ssDNA 0 5
[ persistence_length ]
; model lp first residue last residue
WCM 3.0 0 11
```

In the above file the molecule directive contains the molecule name as well as the molecule
indices starting at 0. Because we have 5 DNA strands we have to specify molecules from 0 to 5.
The range given is not strictly taken into account; non-matching molecule indices and names
are simply skipped and a warning is issued.

After the molecule directive we specify the `persistence_length` directive: first you provide
the WCM keyword followed by the persistence length in nm. Subsequently, the first and last
residue of the chain have to be specified. Polyply takes the persistence length into account
by sampling end-to-end distances from the distribution generated by the WCM model. The keyword
WCM indicates that we use the Worm-Like-Chain Model to model end-to-end distances. In principle
other models can be implemented, but at the moment only the WCM model is available.

To generate the chain coordinates simply call polyply in the following fashion:

```bash
polyply gen_coords -p low_conc.top -b build_file.bld -name DNA -dens 1000 -o low_conc.gro
```

Next we have to run an energy minimization on the system. You can find the corresponding mdp
files
[here](https://github.com/marrink-lab/polyply_regression_tests/blob/main/examples/porcine_genome/input/min.mdp).

!!! warning
    For the energy minimization we have to run with the **`-DFLEXIBLE`** option as well as
    with the **`-Deq_polyply`** option. The first takes care that all bonds are flexible,
    which is important because GROMACS constraints don't energy minimize well. The second flag
    tells GROMACS to utilize some special dihedrals that are defined in the itp file. Those
    dihedral angles make sure that the chirality of the DNA is properly produced in the
    coordinates. Running without the flag can result in inverted chirality.

### 3.2 - Starting coordinates for high salt concentration

The starting coordinates for the high salt concentration are generated in the same fashion as
in the low salt concentration regime. However, now we have to change the number of salt
molecules and save the new topology file as `high_conc.top`. The concentration of salt is
supposed to be 1 M. Given the same amount of water molecules we need to add 450 ions of sodium
and 400 ions of chloride. The persistence length at this salt concentration decreases to about
1 nm [2]. Copy the topology file and the build file. Then edit them, fill in the new value, and
use these new files to again generate coordinates for the DNA.

```bash
polyply gen_coords -p high_conc.top -b build_file_high.bld -name DNA -dens 1000 -o high_conc.gro
```

Next run again an energy minimization for these coordinates.

### 3.3 - Comparing chain conformations

After you run the energy minimization you can optionally run a short equilibration; otherwise
let us analyze the created coordinates. To do so, let's have a look at the chains. You should
see that the chains at lower salt concentration are more extended than at higher salt
concentration. To be more quantitative we can compute the end-to-end distance and radius of
gyration. This can be done using `gmx polystat`, as shown below:

```bash
gmx polystat -f min_low.gro -s low_conc.tpr -o low_conc.xvg
gmx polystat -f min_high.gro -s high_conc.tpr -o high_conc.xvg
```

If you run an equilibration you can also analyze the xtc file. If you open the files you will
see that at low salt concentration the chain has a larger end-to-end distance and radius of
gyration compared to the high salt concentration. Note that if you were not to define any
persistence length, your DNA would be even more compact. If you want, you can give it a try
and see the difference.

## 4 - Simple coarse-grained DNA

Of course we can specify more complex sequences and even cyclic ssDNA. To keep the
computational cost lower, this example will be done using coarse-grained Martini 2 DNA [5]. The
topology file and all needed input files can be found
[here](http://cgmartini.nl/index.php/force-field-parameters/dna).

### 4.1 - Linear complex sequence

The easiest way to specify a more complex sequence is to get it in a plain, fasta, or ig
formatted file, which can then be read by polyply. For example, let's consider the example
sequence string shown below. You can also find a larger example sequence
[here](https://www.genomatix.de/online_help/help/sequence_formats.html).

```
> 100bp DNA sequence
ACAAGATGCCATTGTCCCCCGGCCTCCTGCTGCTGCTGCTCTCCGGGGCCACGGCCACCGCTGCCCTGCCCCTGGAGGGTGGCCCCACCGGCCGAGACAG
```

First copy the sequence and comment to a file called `DNA.fasta`. Note that both fasta and ig
file formats allow for comments. To identify if the sequence belongs to DNA or RNA, polyply
reads the comments looking for the keyword DNA or RNA. Not providing the DNA or RNA keyword in
the comments will result in an error. Using the fasta file, an itp file is generated as simply
as:

```bash
polyply gen_params -lib martini2 -o complexDNA.itp -name ssDNA -seqf DNA.fasta
```

Next you will need to download the Martini 2 parameters from
[here](http://cgmartini.nl/index.php/force-field-parameters/dna). And then create a file
called `martini_v2.0_ions.itp` with the following content:

```
[ moleculetype ]
NA        1
[atoms]
;id     type    resnr   residu  atom    cgnr    charge
 1  Qd  1   NA   NA+     1   1.0
[ moleculetype ]
CL        1
[atoms]
;id     type    resnr   residu  atom    cgnr    charge
1  Qa  1  CL     CL-     1   -1.0
```

To generate the coordinates we can use the same syntax and input as for the all-atom
test-case; however, we need the coarse-grained topology file. Assuming a salt concentration of
1 M and that 4 water atoms map to 1 Martini water bead, the topology looks as follows. Note
that we also need to add neutralizing ions again.

```
#include "martini_v2.1-dna.itp"
#include "martini_v2.0_ions.itp"
#include "complexDNA.itp"
[ system ]
DNA in water
[ molecules ]
ssDNA 1
W 25000
NA 450
CL 350
```

As before we utilize a build file to set the persistence length. The coarse-grained build
file as shown below is almost the same as for the atomistic test case, except that now we have
100 base-pairs for only a single chain, so we have to adjust the number of residues and
molecule count:

```
[ molecule ]
ssDNA 0 1
[ persistence_length ]
; model lp first residue last residue
WCM 1.0 0 99
```

Next let's run the standard command for coordinate generation as shown below:

```bash
polyply gen_coords -p martini2.top -b build_file_martini.bld -name DNA -dens 1000 -o martini2.gro
```

When running the command you will see several warnings about the Martini nucleobases which
cannot be optimized. The warnings are harmless in this case, and originate from the fact that
one angle in the nucleobase cannot be optimized. This is however quickly fixed with the energy
minimization at the martini level. Now it should complete without issues and take less than 5
minutes.

### 4.2 - Circular ssDNA sequence

Finally let us have a look at how to make circular ssDNA. The simplest way to proceed is to
specify the sequence in ig format, which specifies a circular sequence by adding a 2 as the
last character of the sequence. Given the sequence above we can simply convert it to ig format
that looks as shown below.

```
; This is a comment
Circular DNA Sequence 100bp
ACAAGATGCCATTGTCCCCCGGCCTCCTGCTGCTGCTGCTCTCCGGGGCCACGGCCACCGCTGCCCTGCCCCTGGAGGGTGGCCCCACCGGCCGAGACAG2
```

To generate the circular DNA itp file we run:

```bash
polyply gen_params -lib martini2 -o circularDNA.itp -name csDNA -seqf circularDNA.ig
```

Next copy the topology file from the previous Martini example to `martini_circle.top`, replace
the itp file, and change the molecule name to csDNA. Coordinates are generated as before; we
just have to add a command line flag that tells polyply to generate it as a circle, which is
done by adding the `-cycle` flag followed by the name of the molecule.

```bash
polyply gen_coords -p martini_circle.top -name DNA -dens 1000 -o circle.gro -cycle csDNA
```

To verify that the structure is indeed circular you can compute the end-to-end distance using
`gmx polystat` as shown before.

## 5 - Useful Notes

- Supported formats for DNA sequences are fasta, ig, and plain.
- To distinguish RNA and DNA, polyply reads the comments of the ig or fasta file, so one of
  the keywords RNA/DNA needs to be present in the file comments before the sequence.
- DNA residues must be named DA, DG, DT, DC.
- You always have to specify the first residue as 5' and the last as 3' end, except when your
  sequence is circular.
- Terminal residue names are the same as above but appended with a 3 or 5 (e.g. DA5 for an
  adenine 5' end).
- For fasta and ig files the terminals are renamed automatically.
- Run energy minimization and equilibration with the `-Deq_polyply` flag.
- For energy minimization use the `-DFLEXIBLE` flag.
- Number of ions given the number of water molecules:
  `((number of water molecules) x conc in molar) / 55.5556`.
- dsDNA is currently not supported, but an extension is on the way.
- RNA force-fields are currently not supported, because the 3D organization of RNA currently
  goes much beyond the capabilities of polyply.

## References

[1] Sim, A.Y., Lipfert, J., Herschlag, D. and Doniach, S., 2012. Salt dependence of the radius
of gyration and flexibility of single-stranded DNA in solution probed by small-angle X-ray
scattering. *Physical Review E*, 86(2), p.021901.

[2] Colby, R.H. and Rubinstein, M. *Polymer Physics*; Oxford University Press: New York, 2003.

[3] Plumridge, A., Andresen, K. and Pollack, L., 2019. Visualizing disordered single-stranded
RNA: connecting sequence, structure, and electrostatics. *Journal of the American Chemical
Society*, 142(1), pp.109-119.

[4] Ivani, I., Dans, P.D., Noy, A., Pérez, A., Faustino, I., Hospital, A., Walther, J., Andrio,
P., Goñi, R., Balaceanu, A. and Portella, G., 2016. Parmbsc1: a refined force field for DNA
simulations. *Nature Methods*, 13(1), pp.55-58.

[5] Uusitalo, J.J., Ingólfsson, H.I., Akhshi, P., Tieleman, D.P. and Marrink, S.J., 2015.
Martini coarse-grained force field: extension to DNA. *Journal of Chemical Theory and
Computation*, 11(8), pp.3932-3945.

[6] Uusitalo, J.J., Ingólfsson, H.I., Marrink, S.J. and Faustino, I., 2017. Martini
coarse-grained force field: extension to RNA. *Biophysical Journal*, 113(2), pp.246-256.
