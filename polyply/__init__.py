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
    import pkg_resources
except ImportError:
    import os
    DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
    TEST_DATA = os.path.join(os.path.dirname(__file__), 'tests/test_data')
    del os
else:
    DATA_PATH = pkg_resources.resource_filename('polyply', 'data')
    TEST_DATA = pkg_resources.resource_filename('polyply', 'tests/test_data')
    del pkg_resources

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
