# https://igraph.org/python/doc/tutorial/tutorial.html#creating-a-graph-from-scratch
from igraph import *

g = Graph()

g.add_vertices(3) # adds vertices 0,1,2

g.add_edges([(0,1), (1,2)])
g.add_edges([(2, 0)])
g.add_vertices(3)
g.add_edges([(2, 3), (3, 4), (4, 5), (5, 3)])
print(g)
print('-----------------------')
print(summary(g))


# to get edge indexes to delete them
g.get_eid(2, 3) # retursn 3
g.delete_edges(3)


# random musings and clippings
mapper = g.vs["name"] # to list mapping between index and name
# https://igraph.org/python/doc/api/igraph._igraph.GraphBase.html#get_all_shortest_paths
actual = g.shortest_paths(source='a', target='d', weights='weight', mode='out') # just returns distancd


g.random_walk("a",3, mode='out', stuck='return')

g=Graph.TupleList([("a", "b", 3.0), ("b", "c", 4.0), ("c", "d", 5.0), ("a", "d", 7.0)], weights=True, directed=True)
path = g.get_shortest_paths('a', to='d', weights=None, mode='out', output='vpath')[0]

def process_path(g, path):
    string_arr = []
    weight = 1
    for i, path_member in enumerate(path):
        string_arr.append(g.vs['name'][path_member])
        if i!=len(path)-1: 
            edge_weight = g[path_member, path[i+1]]
            weight = weight*edge_weight
    print(" ".join(string_arr))
    print(weight)
process_path(g, path)
