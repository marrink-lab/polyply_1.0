# Copyright 2020 University of Groningen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pbr.version
#from .log_helpers import StyleAdapter, get_logger

__version__ = pbr.version.VersionInfo('polyply').release_string()

# Find the data directory once.
try:
    from importlib.resources import files, as_file
    import atexit
    from contextlib import ExitStack
except ImportError:
    from pathlib import Path
    DATA_PATH = Path(__file__).parent / 'data'
    TEST_DATA = Path(__file__).parent / 'tests/test_data'
    del Path
else:
    ref_data = files('polyply') / 'data'
    ref_test = files('polyply') / 'tests'/ 'test_data'
    file_manager = ExitStack()
    atexit.register(file_manager.close)
    DATA_PATH = file_manager.enter_context(as_file(ref_data))
    TEST_DATA = file_manager.enter_context(as_file(ref_test))

    del files, as_file, atexit, ExitStack

del pbr

# import numba if available
import functools
try:
    from numba import jit
except:
    jit = lambda x: x  # See also https://docs.python.org/3/library/functools.html#functools.wraps
    print("INFO - Couldn't import numba. Install it for a speed boost.")
else:
    jit = functools.partial(jit,  nopython=True, cache=True, fastmath=True)

# This could be useful for the high level API
from .src.meta_molecule import (Monomer, MetaMolecule)
from .src.apply_links import ApplyLinks
from .src.map_to_molecule import MapToMolecule
from .src.gen_itp import gen_itp, gen_params
from .src.gen_coords import gen_coords
from .src.gen_seq import gen_seq
