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
"""
Test linear algebra aux functions.
"""

import textwrap
import pytest
import numpy as np
import math
import networkx as nx
import vermouth
import polyply

#from polyply import DATA_PATH

#class TestLoadLibrary():
#
#      @staticmethod
#      def test__resolve_lib_paths():
#          libnames = ["martini", "2016H66"]
#          paths = 
#          _resolve_lib_paths(lib_names, DATA_PATH):
