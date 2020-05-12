# polyply VERSION_TAG

## Functionality
Polyply is a python tool designed to generate itp files and coordinates for (bio)macromolecules such as syntehtic polymers, polysaccharides, proteins and DNA. It also facilitates manipulation of itp files and structures to extend or change already existing files. A library for of some commonly used macromolecules using different force-fields is included. In principle the program can be used with any type of force-field (FF). However, some polymers  It has mainly been developed and tested for [MARTINI polymers](http://www.cgmartini.nl/index.php/force-field-parameters/polymers). Make sure to always verify the results and give appropriate credit to the developers of the force-field, molecule parameters and this program. 

### Installation
Polyply can be installed from GitHub directly by downloading the repository and within the downloaded repository executing the command:
```
pip3 install ./
```
Alternatively when using pip, polyply can be installed directly via the following command: 
```
pip3 install git+https://github.com/fgrunewald/polyply_1.0.git#polyply_1.0
```

### Polymer Library
Some macromolecules are implmented in our library for a range of different force-fields. To get a list of all libraries available run:
```
polyply -list-libs <Force Field Name> 
```
To get a detailed list of all the individual macormolecules us:
```
polyply -list-blocks <Force Field Name> 
```
Note that while you can combine fragments from different libraries (e.g. the martini-3 polymer library with martini-3 proteins), we cannot gurantee that all the links are present to make a meaningful itp. All blocks within a library can safely be combined. For more information on how to implement links between different blocks see the wiki.  

### Initial structure generation
To generate an initial structure run:
```
polyply gen_coords -p <top> -o <name_outfile + .gro> -name <name of molecule>
```
In order to append coordinates to an already existing coordinate file run:
```
polyply gen_coords -p <top> -o <name_outfile + .gro> -name <name of molecule> -c <init_coords.gro>
```
Note that at the moment polyply can only generate disordered structures of polymers. All molecules that have secondary structure (e.g. DNA, proteins) cannot be generated accurately. However, disordered proteins are possible to generate. 

## Contributions
The development of polyply is done on [github]. Contributions
are welcome as [bug reports] and [pull requests]. Note however that the
decision of whether or not contributions can give authorship on the resulting
academic paper is left to our sole discretion.

## Authors
F. Grunewald
P. C. Kroon

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

The [full text of the license][license] is available in the source repository.
