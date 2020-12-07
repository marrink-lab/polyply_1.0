# polyply

[![codecov](https://codecov.io/gh/marrink-lab/polyply_1.0/branch/master/graph/badge.svg)](https://codecov.io/gh/marrink-lab/polyply_1.0)

## Functionality
Polyply is a python suite designed to facilitate the generation of input files for simulating
bio-macromolecules with GROMACS. It is possible to generate both itp and coordinates files for most
(bio) macromolecules such as synthetic polymers or polysaccharides. It also facilitates
manipulation of itp files and structures to extend or change already existing files. A library for
some commonly used macro molecules using different force-fields (martini, gromos) is included.
In principle the program can be used with any type of force-field (FF).

Make sure to always verify the results and give appropriate credit to the developers of the
force-field, molecule parameters and this program.

### Installation
Polyply requires python 3.6 or greater. It is distributed via [PyPi][pypi_polyply], and can be installed 
using the pip command:
```
pip install polyply
```
This installs the last released version. You can update an existing installation by running pip install -U polyply. 
In some cases you may want to experiment with running the latest development version. You can install this 
version with the following command:
```
pip install git+https://github.com/fgrunewald/polyply_1.0.git#polyply_1.0
```
The behavior of the pip command can vary depending of the specificity of your python installation. See the 
[documentation on installing a python package][pipdoc] to learn more.

### Polymer Library
Some macro molecules are implemented in our library for a range of different force-fields.
To get a list of all libraries available run:
```
polyply -list-lib
```
To get a detailed list of all the individual macro molecules:
```
polyply -list-blocks <Force Field Name>
```
Note that while you can combine fragments from different libraries (e.g. the martini-3 polymer
library with martini-3 proteins), we cannot guarantee that all the links are present to make a
meaningful itp. All blocks within a library can safely be combined. For more information on how
to implement links between different blocks see the wiki.

### Itp file generation
To generate a linear polymer chain using parameters, provided in the library run:
```
polyply gen_itp -lib <library_name> -name <name> -seq <monomer:#number> -o <name_outfile + .itp>
```

For more information on how to generate itp-files for more complex polymers or how
to combine them with existing itp-files see the wiki pages.

### Initial structure generation
To generate an initial structure run:
```
polyply gen_coords -p <top> -o <name_outfile + .gro> -name <name of molecule> -dens <density>
```
or:
```
polyply gen_coords -p <top> -o <name_outfile + .gro> -name <name of molecule> -box <x, y, z>
```
In order to append coordinates to an already existing coordinate file run:
```
polyply gen_coords -p <top> -o <name_outfile + .gro> -name <name of molecule> -c <init_coords.gro> -box/-dens
```
Note that at the moment polyply can only generate disordered structures of polymers. All molecules
that have secondary structure (e.g. DNA, proteins) cannot be generated accurately. At the moment
chirality is also not taken into account. All polymers are atactic unless a dihedral specifies the 
chirality.

## Contributions
The development of polyply is done on [github]. Contributions
are welcome as [bug reports] and [pull requests]. Note however that the
decision of whether or not contributions can give authorship on the resulting
academic paper is left to our sole discretion.

## License

Polyply is distributed under the Apache 2.0 license.

    Copyright 2020 University of Groningen

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

		http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.

The full text of the license is available in the source repository.

[github]: https://github.com/marrink-lab/polyply_1.0
[bug reports]: https://github.com/marrink-lab/polyply_1.0/issues
[pull requests]: https://github.com/marrink-lab/polyply_1.0/pulls
[pypi_polyply]: https://pypi.org/project/polyply/
[pipdoc]: https://packaging.python.org/tutorials/installing-packages/#installing-packages
