# polyply VERSION_TAG

## Functionality
PolyPly can be used to **generate GROMACS itp files** of polymers from a monomer itp-file and **generate coordinates of polymer molecules** from a complete topology file. In principle the program can be used with any type of force-field (FF) as long as the files are either in the polyply special itp file format or the vermouth file-format. It has mainly been developed and tested for [MARTINI polymers](http://www.cgmartini.nl/index.php/force-field-parameters/polymers). Make sure to always **verify the results** and **give appropriate credit** to the developers of the force-field, molecule parameters and this program 

### Installation
PolyPly can be installed from GitHub directly by downloading the repository and within the downloaded repository executing the command:
```
pip3 install ./
```
Alternatively when using pip, PolyPly can be installed directly via the following command: 
```
pip3 install git+https://github.com/fgrunewald/polyply_1.0.git#polyply_1.0
```
Since for instance the itp-files but also the formats are subject to more or less frequent changes, it is recommended to install PolyPly via with the -e option to pip. In this way any updates can directly be fatched from GitHub using the git update command. Alternatively new changes can be installed by using the -upgrade flag to pip.

### Itp-file generation from library

### Initial structure generation

## Authors

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

## Contributions

The development of polyply is done on [github]. Contributions
are welcome as [bug reports] and [pull requests]. Note however that the
decision of whether or not contributions can give authorship on the resulting
academic paper is left to our sole discretion.
