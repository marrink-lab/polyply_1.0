# polyply

[![codecov](https://codecov.io/gh/marrink-lab/polyply_1.0/branch/master/graph/badge.svg)](https://codecov.io/gh/marrink-lab/polyply_1.0)
[![Build Status](https://github.com/marrink-lab/polyply_1.0/actions/workflows/python-app.yml/badge.svg)](https://github.com/marrink-lab/polyply_1.0/actions)
[![PyPI version](https://badge.fury.io/py/polyply.svg)](https://badge.fury.io/py/polyply)
![license](https://img.shields.io/github/license/marrink-lab/polyply_1.0)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/marrink-lab/polyply_1.0/pypi_deploy.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2105.05890-b31b1b.svg)](https://arxiv.org/abs/2105.05890)
[![DOI:10.1038/s41467-021-27627-4](https://zenodo.org/badge/DOI/10.1038/s41467-021-27627-4.svg)](https://doi.org/10.1038/s41467-021-27627-4)

## Functionality
Polyply is a python suite designed to facilitate the generation of input files and system coordinates for simulating
(bio)macromolecules such as synthetic polymers or polysaccharides. Input files can be generated either from user
specified building blocks or by using the polymers available in the library. The library currently includes polymer
definitions for the GROMOS (2016H66 & 54A6), OPLS, Parmbsc1, and Martini (2 & 3) force-fields. Coordinates are generated
by a multiscale random-walk protocol that is able to generate condensed phase systems at target density, as well as
more heterogeneous systems such as aqueous two phase systems. In addition, polyply allows to tailor initial chain
conformations by providing a build file. For example, the persistence length can be used to control the initial chain
dimensions. The [quick start](https://github.com/marrink-lab/polyply_1.0/wiki/Quick-Start) section in the wiki gives
an overview of the most important commands. In addition, [tutorials][wiki] are provided for more in-depth information
on how to use the program. Tutorials include how to generate
[Martini polymer systems](https://github.com/marrink-lab/polyply_1.0/wiki/Tutorial:-martini-polymer-melts) and
[write input files](https://github.com/marrink-lab/polyply_1.0/wiki/Tutorial:-writing-.ff-input-files).
More details on the algorithm and verification can be found in the [publication](https://doi.org/10.1038/s41467-021-27627-4).

Make sure to always verify the results and give appropriate credit to the developers of the
force-field, molecule parameters and this program.

## Quick references
[Installation Guide](https://github.com/marrink-lab/polyply_1.0/wiki/Installation)\
[FAQs](https://github.com/marrink-lab/polyply_1.0/wiki/FAQs)\
[Current Polyply Polymer Library](./LIBRARY.md)\
[Submissions to Martini Polymer Library](https://github.com/marrink-lab/polyply_1.0/wiki/Submit-polymer-parameters)\
[Tutorial: Martini Polymers](https://github.com/marrink-lab/polyply_1.0/wiki/Tutorial:-martini-polymer-melts)\
[Tutorial: GROMOS Polymers](https://github.com/marrink-lab/polyply_1.0/wiki/Tutorial:-GROMOS-polymer-melts)\
[Tutorial: PEGylated lipid bilayers](https://github.com/marrink-lab/polyply_1.0/wiki/Tutorial:-PEGylated-lipid-bilayers)\
[Tutorial: Single-stranded DNA](https://github.com/marrink-lab/polyply_1.0/wiki/Tutorial:-Single-stranded-circular-DNA)
## News 
- (Feb 8, 22') **Featured Research Article in Nature Communcations.** Our article on the polyply software suite is now featured on the [Editors' Highlights](https://www.nature.com/collections/hhfigaahch) for Structural biology, biochemistry and biophysics in Nature Communications. The Editors’ Highlights pages aims to showcase the 50 best papers recently published in an area. The development team is beyond happy to receive this honor.   
- (May 23, 22') **Fighting Cancer with polyply.** Dane et al. used polyply to setup simulations of vesicles and lipid nanodiscs (LNDs) containing PEGylated lipids, which are used as nanocarriers for cancer therapeutics. They find that LNDs are more effective in delivery likely due to their higher flexibility.  Check it out in [Nature Materials](https://www.nature.com/articles/s41563-022-01251-z). 
- (Jan 18, 23') **Towards whole cell simulations with polyply.** In [a perspective on whole-cell simulations](https://www.frontiersin.org/articles/10.3389/fchem.2023.1106495/full) using the Martini force field, Stevens *et al.* utilize polyply to construct the full 0.5 Mio bp chromosome of the Syn3A minimal cell. This impressive task is a good example of the power of the upcoming DNA implementation into polyply and the role of polyply in the Martini Ecosystem.
- (May 11, 23') **Electronic coarse-grained models in polyply.** Curious about simulating non-conjugated radical-containing polymers for all-organic batteries? The models developed [in this work](https://doi.org/10.1021/acs.macromol.3c00141) for PTMA, or poly(TEMPO acrylamide), are avaiable in polyply. There is an all-atom model and different coarse-grained model (Martini 3, iterative Boltzmann inversion), all of which can be flexibly handled by polyply. 

## Contributions & Support
We are happy to accept submissions of polymer parameters to the polyply library. To submit parameters simply 
open an [issue][bug reports]. More details on submitting parameters can be found 
[here](https://github.com/marrink-lab/polyply_1.0/wiki/Submit-polymer-parameters). The code development of polyply is done 
on [github]. Contributions are welcome as [bug reports] and [pull requests] from everyone. We are also happy to discuss
any of your projects or hear about how you used polyply in your research project. Let us know on the 
[discussions board](https://github.com/marrink-lab/polyply_1.0/discussions) or by tweeting with #CG_MARTINI or #polyplyMD.

## Citation
```
@article{Grunewald2022Polyply,
  title={Polyply; a python suite for facilitating simulations of (bio-) macromolecules and nanomaterials},
  author={Gr{\"u}newald, Fabian and Alessandri, Riccardo and Kroon, Peter C and 
  	  Monticelli, Luca and Souza, Paulo CT and Marrink, Siewert J},
  journal={Nature Communications},
  volume={13},
  pages={68},
  doi={https://doi.org/10.1038/s41467-021-27627-4},
  year={2022}
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
