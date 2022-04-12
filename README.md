# nimnet

![nimnet-logo](nimnet-logo.png)

Network Science in Nim!

- porting `NetworkX` in Nim

# Usage

```
# TODO: give a example here
```

# Install

```
# TODO: publish to nimble
```

# Test

```
$ nimble test
  Verifying dependencies for nimnet@0.1.0
  Compiling C:\Users\oaz77\src\nimnet\tests\test_algorithms (from package nimnet) using c backend
[OK] triangles in graph
[OK] transitivity in graph
[OK] transitivity in directed graph
[OK] core number of graph
[OK] core number of directed graph
[OK] k core of graph
[OK] k core of directed graph
[OK] k shell of graph
[OK] k shell of directed graph
[OK] k crust of graph
[OK] k crust of directed graph
[OK] k corona of graph
[OK] k corona of directed graph
[OK] k truss of graph
[OK] onion layers of graph
[OK] is edge cover
[OK] dominating set of graph
[OK] dominating set of directed graph
[OK] check node set is dominating set of graph
[OK] check node set is dominating set of directed graph
[OK] check whether the degree sequence is valid
[OK] check whether the degree sequence is valid
[OK] check whether the degree sequence is valid
[OK] check whether the degree sequence is valid
[OK] isolated nodes in graph
[OK] isolated nodes in directed graph
[OK] pagerank on graph
[OK] pagerank on directed graph
[OK] hits on graph
[OK] hits on directed graph
[OK] maximal matching in graph
[OK] moral graph on directed graph
[OK] complement of graph
[OK] complement of directed graph
[OK] reverse directed graph
[OK] reverse directed graph in place
[OK] compose graphs
[OK] compose directed graphs
[OK] try to union graphs
[OK] try to union directed graphs
[OK] union disjoint graphs
[OK] union disjoin directed graphs
[OK] try to union graphs and success
[OK] try to union directed graphs and success
[OK] intersection of graphs
[OK] intersection of directed graphs
[OK] difference of graphs
[OK] difference of directed graphs
[OK] symmetric difference of graphs
[OK] symmetric difference of directed graphs
[OK] full join of graphs
[OK] full join of directed graphs
[OK] compose all graphs
[OK] compose all directed graphs
[OK] union all graphs
[OK] union all directed graphs
[OK] union disjoint all graphs
[OK] union disjoint all directed graphs
[OK] intersection all graphs
[OK] intersection all directed graphs
[OK] try to apply intersectionAll to empty graph and fail
[OK] try to apply intersectionAll to empty directed graph and fail
[OK] power product of graph
[OK] check whether graph is regular
[OK] check whether directed graph is regular
[OK] check whether graph is regular
[OK] single source shortest path length on graph
[OK] single source shortest path length on directed graph
[OK] single target shortest path length on graph
[OK] single target shortest path length on directed graph
[OK] all pairs shortest path length on graph
[OK] all pairs shortest path length on directed graph
[OK] bidirectional shortest path on graph
[OK] bidirectional shortest path on directed graph
[OK] single source shortest path on graph
[OK] single source shortest path on directed graph
[OK] single target shortest path on graph
[OK] single target shortest path on directed graph
[OK] all pairs shortest path on graph
[OK] all pairs shortest path on directed graph
[OK] predecessors on graph
[OK] predecessors on directed graph
[OK] predecessors on graph
[OK] predecessors on directed graph
[OK] multi source dijkstra on graph
[OK] multi source dijkstra on directed graph
[OK] multi source dijkstra path length on graph
[OK] multi source dijkstra path length on directed graph
[OK] multi source dijkstra path length on graph
[OK] multi source dijkstra path length on directed graph
[OK] single source dijkstra on graph
[OK] single source dijkstra on directed graph
[OK] single source dijkstra path length on graph
[OK] single source dijkstra path length on directed graph
[OK] single source dijkstra path on graph
[OK] single source dijkstra path on directed graph
[OK] dijkstra path length on graph
[OK] dijkstra path length on directed graph
[OK] dijkstra path on graph
[OK] dijkstra path on directed graph
[OK] all pairs dijkstra on graph
[OK] all pairs dijkstra on directed graph
[OK] all pairs dijkstra path length on graph
[OK] all pairs dijkstra path length on directed graph
[OK] all pairs dijkstra path on graph
[OK] all pairs dijkstra path on directed graph
[OK] dijkstra predecessor and distance on graph
[OK] dijkstra predecessor and distance on directed graph
[OK] bellman ford path length on graph
[OK] bellman ford path length on directed grpah
[OK] bellman ford path length on directed grpah
[OK] single source bellman ford on graph
[OK] single source bellman ford on directed graph
[OK] single source bellman ford path length on graph
[OK] single source bellman ford path length on directed graph
[OK] single source bellman ford path on graph
[OK] single source bellman ford path on directed graph
[OK] bellman ford path on graph
[OK] bellman ford path on directed graph
[OK] all pairs bellman ford path length on graph
[OK] all pairs bellman ford path length on directed graph
[OK] all pairs bellman ford path on graph
[OK] all pairs bellman ford path on directed graph
[OK] bellman ford predecessor and distance on graph
[OK] bellman ford predecessor and distance on directed graph
[OK] bellman ford predecessor and distance on directed graph
[OK] goldberg radzik on graph
[OK] goldberg radzik on directed graph
[OK] negative edge cycle on directed graph
[OK] find negative cycle in graph
[OK] find negative cycle in directed graph
[OK] find negative cycle in directed graph
[OK] bidirectional dijkstra on graph
[OK] bidirectional dijkstra on directed graph
[OK] johnson on graph
[OK] johnson on directed graph
[OK] floyd warshall predecessor and distance on graph
[OK] floyd warshall predecessor and distance on directed graph
[OK] floyd warshall on graph
[OK] floyd warshall on directed graph
[OK] astar path on graph
[OK] astar path on directed graph
[OK] astar path length on graph
[OK] astar path length on directed graph
[OK] average shortest path length on graph
[OK] average shortest path length on directed graph
[OK] all shortest paths on graph
[OK] all shortest paths on directed graph
[OK] simple path on graph
[OK] simple path on directed graph
[OK] compute s-metric for graph
[OK] compute s-metric for directed graph
[OK] dfs edges on graph
[OK] dfs edges on directed graph
[OK] dfs tree on graph
[OK] dfs tree on directed graph
[OK] dfs predecessor on graph
[OK] dfs predecessor on directed graph
[OK] dfs successors on graph
[OK] dfs successors on directed graph
[OK] dfs labeled edges on graph
[OK] dfs labeled edges on directed graph
[OK] dfs post order nodes on graph
[OK] dfs post order nodes on directed graph
[OK] dfs pre order nodes on graph
[OK] dfs pre order nodes on directed graph
[OK] edge dfs on graph
[OK] edge dfs on directed graph
[OK] bfs edges on graph
[OK] bfs edges on directed graph
[OK] bfs tree on graph
[OK] bfs tree on directed graph
[OK] bfs predecessor on graph
[OK] bfs predecessor on directed graph
[OK] bfs successors on graph
[OK] bfs successors on directed graph
[OK] descendants at distance on graph
[OK] descendants at distance on directed graph
[OK] bfs beam edges on graph
[OK] bfs beam edges on directed graph
[OK] edge bfs on graph
[OK] edge bfs on directed graph
[OK] chain decomposition of graph
[OK] check whether it is connected graph
[OK] check wheter null graph is connected and fail
[OK] connected components in graph
[OK] check whether directed graph is a single attracting component
[OK] articulation points in grpah
[OK] biconnected components in graph
[OK] biconnected component edges in graph
[OK] check whether it is biconnected graph
[OK] check whether graph is eulerian
[OK] check whether directed graph is eulerian
[OK] degree centrality for graph
[OK] in-degree centrality for directed graph
[OK] out-degree centrality for directed graph
[OK] eigenvector centrality for graph
[OK] eigenvector centrality for directed graph
[OK] katz centrality for graph
[OK] katz centrality for directed graph
[OK] closeness centrality for graph
[OK] closeness centrality for directed graph
[OK] incremental closeness centrality for graph
[OK] harmonic centrality on grpah
[OK] harmonic centrality on directed grpah
[OK] descendants on directed graph
[OK] ancestors on directed graph
[OK] topological generations in directed graph
[OK] topological sort in directed graph
[OK] check whether directed graph has cycle
[OK] check whether directed graph is acyclic
[OK] check whether directed graph is aperiodic
[OK] lexicographical sort in directed graph
[OK] all topological sorts on directed graph
[OK] transitive closure in directed graph
[OK] transitive closure dag in directed graph
[OK] transitive reduction of directed graph
[OK] antichain on directed grpah
[OK] resource allocation index
[OK] jaccard coefficient
[OK] adamic adar index
[OK] common neighbor centrality
[OK] prefential attachment
[OK] cn soundarajan hopcroft
[OK] ra index soundarajan hopcroft
[OK] within inter cluster
[OK] d-separation
[OK] check whether graph is a tree
[OK] check whether directed graph is a tree
[OK] check whether graph is a forest
[OK] check whether directed graph is a forest
[OK] check whether directed graph is a branching
[OK] check whether directed graph is an arborescence
[OK] wiener index for graph
[OK] wiener index for directed graph
[OK] closeness vitality on graph
[OK] closeness vitality on directed graph
[OK] extremaBoundingDiameterRadius for graph
[OK] extremaBoundingCenterPeriphery for graph
[OK] eccentricity on graph
[OK] eccentricity on directed graph
[OK] diameter on graph
[OK] diameter on directed graph
[OK] radius on graph
[OK] radius on directed graph
[OK] periphery on graph
[OK] periphery on directed graph
[OK] center on graph
[OK] center on directed graph
[OK] barycenter on graph
[OK] barycenter on directed graph
[OK] efficiency on graph
[OK] global efficiency on graph
[OK] global efficiency on graph
[OK] flow hierachy on directed graph
[OK] kl connected subgraph on graph
[OK] kl connected subgraph on directed graph
[OK] check whether graph is kl connected
[OK] check whether directed graph is kl connected
[OK] check bridges in graph
[OK] check wheter graph has bridges
   Success: Execution finished
  Verifying dependencies for nimnet@0.1.0
  Compiling C:\Users\oaz77\src\nimnet\tests\test_basics (from package nimnet) using c backend
[OK] create empty graph
[OK] create graph with nodes
[OK] create graph with edges
[OK] add node at graph
[OK] add none at graph
[OK] add nodes at graph
[OK] remove node at graph
[OK] remove nodes at graph
[OK] add edge at graph
[OK] add edges at graph
[OK] remove edge at graph
[OK] remove edges at graph
[OK] clear grpah
[OK] clear edges at graph
[OK] get nodes at graph
[OK] report if node exists or not at graph
[OK] get edges at graph
[OK] report if edge exists or not at graph
[OK] get adjacency at graph
[OK] report number of nodes in graph
[OK] report number of edges in graph
[OK] report degree at graph
[OK] report density at graph
[OK] get subgraph at graph
[OK] get subgraph by edges at graph
[OK] show info at graph
[OK] add star and check it at graph
[OK] add path and check it at graph
[OK] add cycle and check it at graph
[OK] get non-neighbors at graph
[OK] get common neighbors at graph
[OK] get non-edges at graph
[OK] get nodes with selfloop edge at graph
[OK] get selfloop edges at graph
[OK] custom operators at graph
[OK] create empty graph
[OK] create graph with nodes
[OK] create graph with edges
[OK] add node at digraph
[OK] add none at digraph
[OK] add nodes at digraph
[OK] remove node at digraph
[OK] remove nodes at digraph
[OK] add edge at digraph
[OK] add edges at digraph
[OK] remove edge at digraph
[OK] remove edges at digraph
[OK] clear grpah
[OK] clear edges at digraph
[OK] get nodes at digraph
[OK] report if node exists or not at digraph
[OK] get edges at digraph
[OK] report if edge exists or not at digraph
[OK] get predecence at digraph
[OK] get succession at digraph
[OK] report number of nodes in graph
[OK] report number of edges in graph
[OK] report in-degree at digraph
[OK] report out-degree at digraph
[OK] report density at digraph
[OK] get subgraph at digraph
[OK] get subgraph by edges at digraph
[OK] show info at digraph
[OK] add star and check it at digraph
[OK] add path and check it at digraph
[OK] add cycle and check it at digraph
[OK] get non-predecessors at digraph
[OK] get non-successors at digraph
[OK] get non-neighbors at digraph
[OK] get common predecessors at digraph
[OK] get common successors at digraph
[OK] get common neighbors at digraph
[OK] get non-edges at digraph
[OK] get nodes with selfloop edge at digraph
[OK] get selfloop edges at digraph
[OK] custom operators at digraph
[OK] copy graph as graph
[OK] copy graph as directed graph
[OK] copy directed graph as directed graph
[OK] copy directed graph as graph
[OK] convert graph to directed graph
[OK] convert directed graph to graph
[OK] create empty copy of graph
[OK] create emtpy copy of directed graph
[OK] create empty copy of directed graph
[OK] create emtpy copy of directed graph
[OK] check empty graph
[OK] check empty directed graph
[OK] reverse edge
[OK] reverse directed graph
   Success: Execution finished
  Verifying dependencies for nimnet@0.1.0
  Compiling C:\Users\oaz77\src\nimnet\tests\test_generators (from package nimnet) using c backend
[OK] generate balanced tree
[OK] generate balanced directed tree
[OK] generate barbell graph
[OK] generate barbell directed graph
[OK] generate binomial tree
[OK] generate complete graph
[OK] generate complete directed graph
[OK] generate circular ladder graph
[OK] generate circular ladder directed graph
[OK] generate circulant graph
[OK] generate circulant directed graph
[OK] generate cycle grpah
[OK] generate cycle directed graph
[OK] generate dorogovtsev goltsev mendes graph
[OK] generate empty graph
[OK] generate empty directed graph
[OK] generate full rary tree
[OK] generate full rary directed tree
[OK] generate ladder graph
[OK] generate ladder directed graph
[OK] generate lollipop graph
[OK] generate null graph
[OK] generate null directed graph
[OK] generate path graph
[OK] generate path directed graph
[OK] generate star graph
[OK] generate star directed graph
[OK] generate trivial graph
[OK] generate trivial directed graph
[OK] generate wheel graph
[OK] generate wheel directed graph
[OK] generate bull graph
[OK] generate chvatal graph
[OK] generate cubical graph
[OK] generate diamond graph
[OK] generate frucht graph
[OK] generate frucht directed graph
[OK] generate hoffman singleton graph
[OK] generate house graph
[OK] generate house with x graph
[OK] generate icosahedral graph
[OK] generate krackhardt kite graph
[OK] generate octahedral graph
[OK] generate petersen graph
[OK] generate sedgewick maze graph
[OK] generate sedgewick maze directed graph
[OK] generate tetrahedral graph
[OK] generate tetrahedral directed graph
[OK] generate truncated cube graph
[OK] generate truncated tetrahedron graph
[OK] generate truncated tetrahedron directede graph
[OK] generate tutte graph
[OK] generate karate graph
[OK] generate davis southern women graph
[OK] generate florentine families graph
[OK] generate les miserables graph
   Success: Execution finished
  Verifying dependencies for nimnet@0.1.0
  Compiling C:\Users\oaz77\src\nimnet\tests\test_readwrite (from package nimnet) using c backend
[OK] read adjlist as graph
[OK] read adjlist as directed graph
[OK] read edgelist as graph
[OK] read edgelist as directed graph
   Success: Execution finished
   Success: All tests passed
```