from collections import Counter, defaultdict
import numpy as np
import networkx as nx
import vermouth
import vermouth.gmx
from vermouth.gmx.itp_read import ITPDirector

class GMXTopologyParser(SectionLineParser):
    '''
    Parser for polyply input format.
    '''


