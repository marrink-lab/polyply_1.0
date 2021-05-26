# polyply

[![codecov](https://codecov.io/gh/marrink-lab/polyply_1.0/branch/master/graph/badge.svg)](https://codecov.io/gh/marrink-lab/polyply_1.0)

## Functionality
Polyply is a python suite designed to facilitate the generation of input files and coordinates for simulating
(bio)macromolecules such as synthetic polymers or polysaccharides. Input files can be generated either from user
specified building blocks or by using the polymers available in the library. The library currently includes polymer
definitions for the GROMOS and Martini force-fields. The [quick start](https://github.com/marrink-lab/polyply_1.0/wiki/Quick-Start)
section in the wiki gives an overview of the most important commands. In addition some [tutorials][wiki] are provided for more
in-depth information on how to use the program. Tutorials for example include, how to generate
[Martini polymer systems](https://github.com/marrink-lab/polyply_1.0/wiki/Tutorial:-martini-polymer-melts) or
[write input files](https://github.com/marrink-lab/polyply_1.0/wiki/Tutorial:-writing-.ff-input-files).
More details on the algorithm and verification can be found in the [publication](https://arxiv.org/abs/2105.05890).

Make sure to always verify the results and give appropriate credit to the developers of the
force-field, molecule parameters and this program. 

## Quick references
[Installation Guide](https://github.com/marrink-lab/polyply_1.0/wiki/Installation)\
[FAQs](https://github.com/marrink-lab/polyply_1.0/wiki/FAQs)\
[Submissions to Martini Polymer Library](https://github.com/marrink-lab/polyply_1.0/wiki/Submit-polymer-parameters)\
[Tutorial Martini Polymers](https://github.com/marrink-lab/polyply_1.0/wiki/Tutorial:-martini-polymer-melts)\
[Tutorial GROMOS Polymers](https://github.com/marrink-lab/polyply_1.0/wiki/Tutorial:-GROMOS-polymer-melts)

## Contributions
We are happy to accept submissions of polymer parameters (mostly for the Martini force-field). After a small quality control
procedure, parameters are distributed via the Martini library including appropiate citations. To submit parameters simply 
open an [issue][bug reports]. More details on submitting parameters can be found 
[here](https://github.com/marrink-lab/polyply_1.0/wiki/Submit-polymer-parameters). The code development of polyply is done 
on [github]. Contributions are welcome as [bug reports] and [pull requests] from everyone.

## Citation
```
@article{grunewald2021polyply,
  title={Polyply: a python suite for facilitating simulations of (bio-) macromolecules and nanomaterials},
  author={Gr{\"u}newald, Fabian and Alessandri, Riccardo and Kroon, Peter C and 
  	  Monticelli, Luca and Souza, Paulo CT and Marrink, Siewert J},
  journal={arXiv preprint arXiv:2105.05890},
  year={2021}
}
```

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
[wiki]:https://github.com/marrink-lab/polyply_1.0/wiki
[pypi_polyply]: https://pypi.org/project/polyply/
[pipdoc]: https://packaging.python.org/tutorials/installing-packages/#installing-packages
