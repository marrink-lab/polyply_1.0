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
import networkx as nx
import vermouth

def neighborhood(graph, source, max_length, min_length=1):
    """
    Returns all neighbours of `source` that are less or equal
    to `cutoff` nodes away and more or equal to `start` away
    within a graph excluding the node itself.

    Parameters
    ----------
    graph: :class:`networkx.Graph`
        A networkx graph definintion
    source:
        A node key matching one in the graph
    max_length: :type:`int`
        The maxdistance between the node and its
        neighbours.
    min_length: :type:`int`
        The minimum length of a path. Default
        is zero

    Returns
    --------
    list
       list of all nodes distance away from reference
    """
    paths = nx.single_source_shortest_path(G=graph, source=source, cutoff=max_length)
    neighbours = [ node for node, path in paths.items() if min_length <= len(path)]
    return neighbours

def find_nodes_with_attributes(graph, **attrs):
    """
    Yields all nodes in graph, which have all
    attributes in attrs set to value.

    Parameters:
    -----------
    graph: nx.Graph
    **attrs: collections.abc.Mapping
        The attributes and their desired values.

    Yields:
    --------
    collections.abc.Hashable
    """
    for node in graph.nodes:
        for attr, value in attrs.items():
            if attr not in graph.nodes[node] or graph.nodes[node][attr] != value:
                break
        else:
            yield node

def is_branched(graph):
    """
    Check if any node has a degree larger than 2

    Parameters:
    -----------
    graph: :class:`networkx.Graph`
        A networkx graph definintion

    Returns:
    --------
    bool
       is branched
    """
    for node, deg in graph.degree:
        if deg > 2:
           return True
    return False
