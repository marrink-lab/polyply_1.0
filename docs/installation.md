# Installation

Production versions of polyply are distributed via PyPI, and development versions can be
installed from GitHub directly. Installation with Conda is possible as well, although not
extensively tested. We recommend installing polyply in a virtual environment to prevent
unexpected updates of dependencies or clashes with existing ones.

## Prerequisites and dependencies

Polyply requires Python version 3.6 or greater, and depends on the following libraries,
with versions as indicated:

| Name | Version |
|------|---------|
| numpy | >= 1.18.1 |
| vermouth | >= 0.7.1 |
| networkx | ~=2.0 |
| scipy | >= 1.5.4 |
| pbr | |
| decorator | 4.4.2 |
| tqdm | >= 4.43.0 |
| numba (optional) | >= 0.51.2 |

## Quick start

Install polyply from PyPI using:

```bash
pip install polyply
```

Alternatively, install it from GitHub using:

```bash
pip install git+https://github.com/marrink-lab/polyply_1.0.git#polyply_1.0
```

## Install in a virtual environment with numba

First create a virtual environment:

```bash
python3 -m venv polyply-env
```

Next activate the environment on Unix or macOS using:

```bash
source polyply-env/bin/activate
```

Or on Windows:

```bat
polyply-env\Scripts\activate.bat
```

Next install numba:

```bash
pip install numba
```

And finally install polyply:

```bash
pip install polyply
```

!!! tip
    Installing numba alongside polyply gives significant speed-ups during coordinate
    generation.
