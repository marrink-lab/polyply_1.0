# Martini_PolyPly (beta-version)

## Functionality 
PolyPly can be used to **generate GROMACS itp files** of polymers from a monomer itp-file and **generate coordinates of polymer molecules** from a complete topology file. In principle the program can be used with any type of force-field (FF) as long as the files are in GROMACS format. It has mainly been developed and tested for [MARTINI polymers](http://www.cgmartini.nl/index.php/force-field-parameters/polymers). Make sure to always **verify the results** and **give appropriate credit** to the developers of the force-field, molecule parameters and this program (i.e. ref[1]). If you use this program in your research, please cite reference 1. The current (beta) release has been tested for the following molecules: 

| Molecule         | Force-Field | itp-generation | structure-generation | reference |
|------------------|-------------|----------------|----------------------|-----------|
| PEO              | MARTINI 2   | yes            | yes                  | 1         |
| PEO              | MARTINI 3*  | yes            | yes                  | 1         |
| PS               | MARTINI 2   | yes            | yes                  | 3         |
| PSS              | MARTINI 2   | yes            | yes                  | 4         |
| PDADMA           | MARTINI 2   | yes            | yes                  | 4         |
| PE               | MARTINI 2   | yes            | no                   | 5         |
| PP               | MARTINI 2   | yes            | yes                  | 5         |
| P3HT             | MARTINI 2   | yes            | no                   | 6         |
| PEGylated lipids | MARTINI 2   | yes            | yes                  | 1         |

&ast; MARTINI 3 is currently still in the test phase and not all properties are correct; for production it is recommended to use the MARTINI 2 model
### Installation
PolyPly can be installed from GitHub directly by downloading the repository and within the downloaded repository executing the command:
```
pip3 install ./
```
Alternatively when using pip, PolyPly can be installed directly via the following command: 
```
pip3 install git+https://github.com/fgrunewald/Martini_PolyPly.git#polyply
```
Since for instance the itp-files but also the formats are subject to more or less frequent changes, it is recommended to install PolyPly via with the -e option to pip. In this way any updates can directly be fatched from GitHub using the git update command. Alternatively new changes can be installed by using the -upgrade flag to pip.

### Itp-file generation from library
To generate a polymer itp-file this program offers two options. Either polymers from the library (mostly MARTINI) can be generated or one can generate polymers from custom monomer itp-files. To generate a polymer, which is part of the library simply execute the following command:
```
polyply -polymer [name of polymer] -n_mon [repeat units] -o [outfile] -name [outname of polymer]
```
The 'n_mon' option sets the number of repeat units (i.e. n monomers) and the name option gives the possibility to give another than the default name to the polymer. The names of the polymers in the library can be found by executing the command:
```
polyply -lib
```
Note that the polymer name is composed of three components: an abbreviation, the force-field and the version of the force-field and/or model. The three parts are separated by a period. For example to get PEO MARTINI 2 use “PEO.martini.2” as name of the polymer. Feel free to propose new monomer itp files to be included in the monomer directory. More information on generating itp-files for block-copolymers, adding end-groups or custom itp-fiels can be found in the wiki pages. 

### Initial structure generation
PolyPly also offers the possibility to grow polymers into existing systems or generate a single polymer chain. To indicate that a structure needs to be generated use the -env file with the appropriate selection and supply a topology file via the -p option. If you want the polymer to be grown onto a bilayer or in a solvent use the -sys option to supply a box or the bilayer. More details are outlined on the wiki-page. 
```
polyply -env [ vac, bilayer, sol] -p [topfile] -o [outfile] -name [name of polymer] -sys [system structure file]
```
## References 
1. F. Grunewald, G. Rossi, A.H. De Vries, S.J. Marrink, L. Monticelli. A Transferable MARTINI Model of Polyethylene Oxide. JPCB, 2018, online. doi:10.1021/acs.jpcb.8b04760 
2. C. Senac, W. Urbach, E. Kurtisovski, P. H. Hünenberger, B. A. Horta, N. Taulier, P. F. Fuchs. Simulating bilayers of nonionic surfactants with the GROMOS-compatible 2016H66 force field. Langmuir 2017, 33(39), 10225-10238. doi:10.1021/acs.langmuir.7b01348
3. G. Rossi, L. Monticelli, S. R. Puisto, I. Vattulainen, T. Ala-Nissila. Coarse-graining polymers with the MARTINI force-field: polystyrene as a benchmark case. Soft Matter, 2011, 7(2), pp.698-708. doi:10.1039/C0SM00481B
4. M. Vögele, C. Holm, J. Smiatek. Coarse-grained simulations of polyelectrolyte complexes: MARTINI models for poly (styrene sulfonate) and poly (diallyldimethylammonium). The Journal of chemical physics. 2015 Dec 28;143(24):243151. doi:10.1063/1.4937805
5. E. Panizon, D. Bochicchio, L. Monticelli, G. Rossi MARTINI coarse-grained models of polyethylene and polypropylene. JPCB, 2015 Jun 9;119(25):8209-16. doi: 10.1021/acs.jpcb.5b03611
6. R. Alessandri, J. J. Uusitalo, A. H. de Vries, R. W. A. Havenith, S. J. Marrink. Bulk heterojunction morphologies with atomistic resolution from coarse-grain solvent evaporation simulations. JACS. 2017 ;139(10):3697-705. doi:/jacs.6b11717

## Authors

## License & Legal Notes

This software is distributed under the GNU General Public License v3.0 (G.P.L 3), which permits commercial and non-commercial usage free of charge, modification and redistribution of the software and/or parts of the code under applicable conditions outlined in the license. The [license text](LICENSE) is distributed with the software. Any copy right or license infringements resulting from improper usage and/or modification of the code are punishable under applicable country law. In accordance with the license no warranty is granted with respect to correctness of the outputs. If you or your company suffer material or immaterial losses due to incorrect results, the authors assume no financial liability for those or other losses resulting from the program’s usage. Any user is him/her-self responsible to appropriately give credit to all publications and individual’s, who contributed to the program and/or developed force-field parameters distributed with the program.

