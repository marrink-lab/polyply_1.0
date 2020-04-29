import pbr.version
#from .log_helpers import StyleAdapter, get_logger

__version__ = pbr.version.VersionInfo('polyply').release_string()

# Find the data directory once.
try:
    import pkg_resources
except ImportError:
    import os
    DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
    del os
else:
    DATA_PATH = pkg_resources.resource_filename('polyply', 'data')
    del pkg_resources

del pbr

# This could be useful for the high level API
from .src.meta_molecule import (Monomer, MetaMolecule)
from .src.apply_links import ApplyLinks
from .src.map_to_molecule import MapToMolecule
from .src.gen_itp import gen_itp
from .src.gen_coords import gen_coords
from .src.gen_seq import gen_seq
#from .system import System
