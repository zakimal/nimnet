import unittest

import sets
import tables

import nimnet

test "create empty graph":
  let g = newGraph()
  check g.numberOfNodes() == 0
  check g.numberOfEdges() == 0

test "create graph with nodes":
  let g = newGraph(@[0, 1, 2, 3, 4, 5])
  check g.numberOfNodes() == 6
  check g.numberOfEdges() == 0

test "create graph with edges":
  let g = newGraph(@[(0, 1), (1, 2), (3, 4), (5, 6)])
  check g.numberOfNodes() == 7
  check g.numberOfEdges() == 4
  check g.edges() == @[(0, 1), (1, 2), (3, 4), (5, 6)]

test "add node at graph":
  let g = newGraph()
  check g.numberOfNodes() == 0
  g.addNode(0)
  check g.numberOfNodes() == 1

test "add none at graph":
  let g = newGraph()
  try:
    g.addNode(None)
  except NNError as e:
    check e.msg == "None cannot be a node"

test "add nodes at graph":
  let g = newGraph()
  check g.numberOfNodes() == 0
  g.addNodesFrom(@[0, 1, 2, 3, 4])
  check g.numberOfNodes() == 5

test "remove node at graph":
  let g = newGraph()
  g.addNodesFrom(@[0, 1, 2, 3, 4])
  check g.numberOfNodes() == 5
  g.removeNode(4)
  check g.numberOfNodes() == 4

test "remove nodes at graph":
  let g = newGraph()
  g.addNodesFrom(@[0, 1, 2, 3, 4])
  check g.numberOfNodes() == 5
  g.removeNodesFrom(@[0, 4])
  check g.numberOfNodes() == 3

test "add edge at graph":
  let g = newGraph()
  g.addEdge(0, 1)
  check g.numberOfNodes() == 2
  check g.numberOfEdges() == 1

test "add edges at graph":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2)])
  check g.numberOfNodes() == 3
  check g.numberOfEdges() == 2

test "remove edge at graph":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2)])
  check g.numberOfNodes() == 3
  check g.numberOfEdges() == 2
  g.removeEdge(0, 1)
  check g.numberOfNodes() == 3
  check g.numberOfEdges() == 1

test "remove edges at graph":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (3, 4)])
  check g.numberOfNodes() == 5
  check g.numberOfEdges() == 3
  g.removeEdgesFrom(@[(0, 1), (3, 4)])
  check g.numberOfNodes() == 5
  check g.numberOfEdges() == 1

test "clear grpah":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (3, 4)])
  check g.numberOfNodes() == 5
  check g.numberOfEdges() == 3
  g.clear()
  check g.numberOfNodes() == 0
  check g.numberOfEdges() == 0

test "clear edges at graph":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (3, 4)])
  check g.numberOfNodes() == 5
  check g.numberOfEdges() == 3
  g.clearEdges()
  check g.numberOfNodes() == 5
  check g.numberOfEdges() == 0

test "get nodes at graph":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (3, 4)])
  check g.nodes() == @[0, 1, 2, 3, 4]
  var expected = @[0, 1, 2, 3, 4].toHashSet()
  check g.nodesSet() == expected

test "report if node exists or not at graph":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (3, 4)])
  check g.hasNode(0) == true
  check g.hasNode(5) == false
  check g.contains(0) == true
  check g.contains(5) == false

test "get edges at graph":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 0), (1, 2), (3, 4)])
  check g.edges() == @[(0, 1), (1, 2), (3, 4)]
  var expected = @[(0, 1), (1, 2), (3, 4)].toHashSet()
  check g.edgesSet() == expected

test "report if edge exists or not at graph":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (3, 4)])
  check g.hasEdge(0, 1) == true
  check g.hasEdge((0, 1)) == true
  check g.hasEdge(1, 0) == true
  check g.hasEdge((1, 0)) == true
  check g.hasEdge(5, 6) == false
  check g.hasEdge((5, 6)) == false
  check g.contains(0, 1) == true
  check g.contains((0, 1)) == true
  check g.contains(5, 6) == false
  check g.contains((5, 6)) == false

test "get adjacency at graph":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (3, 4)])
  let expected = @[
    (0, @[1]),
    (1, @[0, 2]),
    (2, @[1]),
    (3, @[4]),
    (4, @[3])
  ]
  check g.adjacency() == expected

test "report number of nodes in graph":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (3, 4)])
  check g.order() == 5
  check g.numberOfNodes() == 5
  check g.len() == 5
  check len(g) == 5

test "report number of edges in graph":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (3, 4), (5, 5)])
  check g.size() == 4
  check g.numberOfEdges() == 4
  check g.numberOfSelfloop() == 1

test "report degree at graph":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (3, 4), (5, 5)])
  var expected = initTable[Node, int]()
  expected[0] = 1
  expected[1] = 2
  expected[2] = 1
  expected[3] = 1
  expected[4] = 1
  expected[5] = 1
  check g.degree() == expected
  var expectedHist = @[0, 5, 1]
  check g.degreeHistogram() == expectedHist

test "report density at graph":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (3, 4), (5, 5)])
  check g.density() == 4.float / (6 * 5).float * 2.float

test "get subgraph at graph":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  let ns = @[1, 2, 3].toHashSet()
  let sg = g.subgraph(ns)
  check sg.edges() == @[(1, 2), (1, 3), (2, 3)]

test "get subgraph by edges at graph":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  let es = @[(0, 1), (1, 3), (3, 5)].toHashSet()
  let sg = g.edgeSubgraph(es)
  check sg.edges() == @[(0, 1), (1, 3), (3, 5)]

test "show info at graph":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  check g.info() == "type: undirected graph\n#nodes: 6\n#edges: 7"
  check g.info(1) == "node 1 has following properties:\ndegree: 4\nneighbors: @[0, 2, 3, 4]"

test "add star and check it at graph":
  let g = newGraph()
  g.addStar(@[0, 1, 2, 3, 4])
  check g.edges() == @[(0, 1), (0, 2), (0, 3), (0, 4)]
  check g.isStar(@[0, 1, 2, 3, 4]) == true
  check g.isStar(@[0, 1, 2]) == true

test "add path and check it at graph":
  let g = newGraph()
  g.addPath(@[0, 1, 2, 3, 4])
  check g.edges() == @[(0, 1), (1, 2), (2, 3), (3, 4)]
  check g.isPath(@[0, 1, 2, 3, 4]) == true
  check g.isPath(@[0, 1, 2]) == true
  check g.isPath(@[2, 1, 0]) == true

test "add cycle and check it at graph":
  let g = newGraph()
  g.addCycle(@[0, 1, 2, 3, 4])
  check g.edges() == @[(0, 1), (0, 4), (1, 2), (2, 3), (3, 4)]
  check g.isCycle(@[0, 1, 2, 3, 4]) == true
  check g.isCycle(@[0, 1, 2]) == false
  check g.isCycle(@[1, 2, 3, 4, 0]) == true
  check g.isCycle(@[4, 3, 2, 1, 0]) == true

test "get non-neighbors at graph":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  check g.nonNeighbors(1) == @[5]
  check g.nonNeighbors(3) == @[0, 4]

test "get common neighbors at graph":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  check g.commonNeighbors(1, 4) == @[2]
  check g.commonNeighbors(3, 4) == @[1, 2]

test "get non-edges at graph":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  check g.nonEdges() == @[
    (0, 2), (0, 3), (0, 4), (0, 5), (1, 5), (2, 5), (3, 4), (4, 5)
  ]
  check g.nonEdges.toHashSet() == @[
    (0, 2), (0, 3), (0, 4), (0, 5), (1, 5), (2, 5), (3, 4), (4, 5)
  ].toHashSet()

test "get nodes with selfloop edge at graph":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 1), (1, 2), (2, 3)])
  check g.nodesWithSelfloopEdge() == @[1]

test "get selfloop edges at graph":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 1), (1, 2), (2, 3)])
  check g.selfloopEdges() == @[(1, 1)]

test "custom operators at graph":
  var g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  check g[1] == @[0, 2, 3, 4]
  check 0 in g == true
  check 0 notin g == false
  check 6 in g == false
  check 6 notin g == true
  g = g + 99
  check 99 in g == true
  check 99 notin g == false
  check g.nodes() == @[0, 1, 2, 3, 4, 5, 99]
  check g.degree(99) == 0
  g = g - 99
  check 99 in g == false
  check 99 notin g == true
  check g.nodes() == @[0, 1, 2, 3, 4, 5]
  g = g + @[97, 98, 99]
  check 99 in g == true
  check 99 notin g == false
  check g.nodes() == @[0, 1, 2, 3, 4, 5, 97, 98, 99]
  check g.degree(97) == 0
  check g.degree(98) == 0
  check g.degree(99) == 0
  g = g - @[97, 98, 99]
  check g.nodes() == @[0, 1, 2, 3, 4, 5]
  g += 99
  check 99 in g == true
  check 99 notin g == false
  check g.nodes() == @[0, 1, 2, 3, 4, 5, 99]
  check g.degree(99) == 0
  g -= 99
  check 99 in g == false
  check 99 notin g == true
  check g.nodes() == @[0, 1, 2, 3, 4, 5]
  g += @[97, 98, 99]
  check g.nodes() == @[0, 1, 2, 3, 4, 5, 97, 98, 99]
  check g.degree(97) == 0
  check g.degree(98) == 0
  check g.degree(99) == 0
  g -= @[97, 98, 99]
  check g.nodes() == @[0, 1, 2, 3, 4, 5]

  check (0, 1) in g == true
  check (0, 5) in g == false
  check (0, 1) notin g == false
  check (1, 0) in g == true
  check (0, 5) notin g == true
  g = g + (0, 5)
  check (0, 5) in g == true
  g = g - (0, 5)
  check (0, 5) in g == false
  g = g + @[(0, 5), (2, 0)]
  check (0, 5) in g == true
  check (2, 0) in g == true
  g = g - @[(0, 5), (2, 0)]
  check (0, 5) in g == false
  check (2, 0) in g == false
  check (0, 2) in g == false
  g += @[(0, 5), (2, 0)]
  check (0, 5) in g == true
  check (2, 0) in g == true
  g -= @[(0, 5), (2, 0)]
  check (0, 5) in g == false
  check (2, 0) in g == false
  check (0, 2) in g == false

test "create empty graph":
  let dg = newDiGraph()
  check dg.numberOfNodes() == 0
  check dg.numberOfEdges() == 0

test "create graph with nodes":
  let dg = newDiGraph(@[0, 1, 2, 3, 4, 5])
  check dg.numberOfNodes() == 6
  check dg.numberOfEdges() == 0

test "create graph with edges":
  let dg = newDiGraph(@[(0, 1), (1, 2), (3, 4), (5, 6)])
  check dg.numberOfNodes() == 7
  check dg.numberOfEdges() == 4
  check dg.edges() == @[(0, 1), (1, 2), (3, 4), (5, 6)]

test "add node at digraph":
  let dg = newDiGraph()
  check dg.numberOfNodes() == 0
  dg.addNode(0)
  check dg.numberOfNodes() == 1

test "add none at digraph":
  let dg = newDiGraph()
  try:
    dg.addNode(None)
  except NNError as e:
    check e.msg == "None cannot be a node"

test "add nodes at digraph":
  let dg = newDiGraph()
  check dg.numberOfNodes() == 0
  dg.addNodesFrom(@[0, 1, 2, 3, 4])
  check dg.numberOfNodes() == 5

test "remove node at digraph":
  let dg = newDiGraph()
  dg.addNodesFrom(@[0, 1, 2, 3, 4])
  check dg.numberOfNodes() == 5
  dg.removeNode(4)
  check dg.numberOfNodes() == 4

test "remove nodes at digraph":
  let dg = newDiGraph()
  dg.addNodesFrom(@[0, 1, 2, 3, 4])
  check dg.numberOfNodes() == 5
  dg.removeNodesFrom(@[0, 4])
  check dg.numberOfNodes() == 3

test "add edge at digraph":
  let dg = newDiGraph()
  dg.addEdge(0, 1)
  check dg.numberOfNodes() == 2
  check dg.numberOfEdges() == 1
  check dg.inDegree(0) == 0
  check dg.outDegree(0) == 1
  check dg.inDegree(1) == 1
  check dg.outDegree(1) == 0

test "add edges at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2)])
  check dg.numberOfNodes() == 3
  check dg.numberOfEdges() == 2
  check dg.inDegree(0) == 0
  check dg.outDegree(0) == 1
  check dg.inDegree(1) == 1
  check dg.outDegree(1) == 1
  check dg.inDegree(2) == 1
  check dg.outDegree(2) == 0

test "remove edge at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 0), (1, 2)])
  check dg.numberOfNodes() == 3
  check dg.numberOfEdges() == 3
  dg.removeEdge(0, 1)
  check dg.numberOfNodes() == 3
  check dg.numberOfEdges() == 2

test "remove edges at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (3, 4), (4, 3)])
  check dg.numberOfNodes() == 5
  check dg.numberOfEdges() == 4
  dg.removeEdgesFrom(@[(0, 1), (3, 4)])
  check dg.numberOfNodes() == 5
  check dg.numberOfEdges() == 2

test "clear grpah":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (3, 4)])
  check dg.numberOfNodes() == 5
  check dg.numberOfEdges() == 3
  dg.clear()
  check dg.numberOfNodes() == 0
  check dg.numberOfEdges() == 0

test "clear edges at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (3, 4)])
  check dg.numberOfNodes() == 5
  check dg.numberOfEdges() == 3
  dg.clearEdges()
  check dg.numberOfNodes() == 5
  check dg.numberOfEdges() == 0

test "get nodes at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (3, 4)])
  check dg.nodes() == @[0, 1, 2, 3, 4]
  var expected = @[0, 1, 2, 3, 4].toHashSet()
  check dg.nodesSet() == expected

test "report if node exists or not at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (3, 4)])
  check dg.hasNode(0) == true
  check dg.hasNode(5) == false
  check dg.contains(0) == true
  check dg.contains(5) == false

test "get edges at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 0), (1, 2), (3, 4)])
  check dg.edges() == @[(0, 1), (1, 0), (1, 2), (3, 4)]
  var expected = @[(0, 1), (1, 0), (1, 2), (3, 4)].toHashSet()
  check dg.edgesSet() == expected

test "report if edge exists or not at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (3, 4)])
  check dg.hasEdge(0, 1) == true
  check dg.hasEdge((0, 1)) == true
  check dg.hasEdge(1, 0) == false
  check dg.hasEdge((1, 0)) == false
  check dg.hasEdge(5, 6) == false
  check dg.hasEdge((5, 6)) == false
  check dg.contains(0, 1) == true
  check dg.contains((0, 1)) == true
  check dg.contains(5, 6) == false
  check dg.contains((5, 6)) == false

test "get predecence at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (3, 4)])
  let expected = @[
    (0, @[]),
    (1, @[0]),
    (2, @[1]),
    (3, @[]),
    (4, @[3])
  ]
  check dg.predecence() == expected

test "get succession at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (3, 4)])
  let expected = @[
    (0, @[1]),
    (1, @[2]),
    (2, @[]),
    (3, @[4]),
    (4, @[])
  ]
  check dg.succession() == expected

test "report number of nodes in graph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (3, 4)])
  check dg.order() == 5
  check dg.numberOfNodes() == 5
  check dg.len() == 5
  check len(dg) == 5

test "report number of edges in graph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (3, 4), (5, 5)])
  check dg.size() == 4
  check dg.numberOfEdges() == 4
  check dg.numberOfSelfloop() == 1

test "report in-degree at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (3, 4), (5, 5)])
  var expected = initTable[Node, int]()
  expected[0] = 0
  expected[1] = 1
  expected[2] = 1
  expected[3] = 0
  expected[4] = 1
  expected[5] = 1
  check dg.inDegree() == expected
  var expectedHist = @[2, 4]
  check dg.inDegreeHistogram() == expectedHist

test "report out-degree at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (3, 4), (5, 5)])
  var expected = initTable[Node, int]()
  expected[0] = 1
  expected[1] = 1
  expected[2] = 0
  expected[3] = 1
  expected[4] = 0
  expected[5] = 1
  check dg.outDegree() == expected
  var expectedHist = @[2, 4]
  check dg.outDegreeHistogram() == expectedHist

test "report density at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (3, 4), (5, 5)])
  check dg.density() == 4.float / (6 * 5).float

test "get subgraph at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  let ns = @[1, 2, 3].toHashSet()
  let sg = dg.subgraph(ns)
  check sg.edges() == @[(1, 2), (1, 3), (2, 3)]

test "get subgraph by edges at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  let es = @[(0, 1), (1, 3), (3, 5)].toHashSet()
  let sg = dg.edgeSubgraph(es)
  check sg.edges() == @[(0, 1), (1, 3), (3, 5)]

test "show info at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  check dg.info() == "type: directed graph\n#nodes: 6\n#edges: 7"
  check dg.info(1) == "node 1 has following properties:\nin-degree: 1\npredecessors: @[0]\nout-degree: 3\nsuccessors: @[2, 3, 4]"

test "add star and check it at digraph":
  let dg = newDiGraph()
  dg.addStar(@[0, 1, 2, 3, 4])
  check dg.edges() == @[(0, 1), (0, 2), (0, 3), (0, 4)]
  check dg.isStar(@[0, 1, 2, 3, 4]) == true
  check dg.isStar(@[0, 1, 2]) == true

test "add path and check it at digraph":
  let dg = newDiGraph()
  dg.addPath(@[0, 1, 2, 3, 4])
  check dg.edges() == @[(0, 1), (1, 2), (2, 3), (3, 4)]
  check dg.isPath(@[0, 1, 2, 3, 4]) == true
  check dg.isPath(@[0, 1, 2]) == true
  check dg.isPath(@[2, 1, 0]) == false

test "add cycle and check it at digraph":
  let dg = newDiGraph()
  dg.addCycle(@[0, 1, 2, 3, 4])
  check dg.edges() == @[(0, 1), (0, 4), (1, 2), (2, 3), (3, 4)]
  check dg.isCycle(@[0, 1, 2, 3, 4]) == true
  check dg.isCycle(@[0, 1, 2]) == false
  check dg.isCycle(@[1, 2, 3, 4, 0]) == false
  check dg.isCycle(@[4, 3, 2, 1, 0]) == false

test "get non-predecessors at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  check dg.nonPredecessors(1) == @[2, 3, 4, 5]
  check dg.nonPredecessors(3) == @[0, 4, 5]

test "get non-successors at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  check dg.nonSuccessors(1) == @[0, 5]
  check dg.nonSuccessors(3) == @[0, 1, 2, 4]

test "get non-neighbors at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  check dg.nonNeighbors(1) == @[5]
  check dg.nonNeighbors(3) == @[0, 4]

test "get common predecessors at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  check dg.commonPredecessors(3, 4) == @[1, 2]

test "get common successors at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  check dg.commonSuccessors(1, 2) == @[3, 4]

test "get common neighbors at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  check dg.commonNeighbors(3, 2) == @[1]

test "get non-edges at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  check dg.nonEdges() == @[
    (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 0), (1, 5),
    (2, 0), (2, 1), (2, 5),
    (3, 0), (3, 1), (3, 2), (3, 4),
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 5),
    (5, 0), (5, 1), (5, 2), (5, 3), (5, 4)
  ]

test "get nodes with selfloop edge at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 1), (1, 2), (2, 3)])
  check dg.nodesWithSelfloopEdge() == @[1]

test "get selfloop edges at digraph":
  let dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 1), (1, 2), (2, 3)])
  check dg.selfloopEdges() == @[(1, 1)]

test "custom operators at digraph":
  var dg = newDiGraph()
  dg.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  check dg[1] == @[0, 2, 3, 4]
  check 0 in dg == true
  check 0 notin dg == false
  check 6 in dg == false
  check 6 notin dg == true
  dg = dg + 99
  check 99 in dg == true
  check 99 notin dg == false
  check dg.nodes() == @[0, 1, 2, 3, 4, 5, 99]
  check dg.inDegree(99) == 0
  check dg.outDegree(99) == 0
  dg = dg - 99
  check 99 in dg == false
  check 99 notin dg == true
  check dg.nodes() == @[0, 1, 2, 3, 4, 5]
  dg = dg + @[97, 98, 99]
  check 99 in dg == true
  check 99 notin dg == false
  check dg.nodes() == @[0, 1, 2, 3, 4, 5, 97, 98, 99]
  check dg.inDegree(97) == 0
  check dg.inDegree(98) == 0
  check dg.inDegree(99) == 0
  check dg.outDegree(97) == 0
  check dg.outDegree(98) == 0
  check dg.outDegree(99) == 0
  dg = dg - @[97, 98, 99]
  check dg.nodes() == @[0, 1, 2, 3, 4, 5]
  dg += 99
  check 99 in dg == true
  check 99 notin dg == false
  check dg.nodes() == @[0, 1, 2, 3, 4, 5, 99]
  check dg.inDegree(99) == 0
  check dg.outDegree(99) == 0
  dg -= 99
  check 99 in dg == false
  check 99 notin dg == true
  check dg.nodes() == @[0, 1, 2, 3, 4, 5]
  dg += @[97, 98, 99]
  check dg.nodes() == @[0, 1, 2, 3, 4, 5, 97, 98, 99]
  check dg.inDegree(97) == 0
  check dg.inDegree(98) == 0
  check dg.inDegree(99) == 0
  check dg.outDegree(97) == 0
  check dg.outDegree(98) == 0
  check dg.outDegree(99) == 0
  dg -= @[97, 98, 99]
  check dg.nodes() == @[0, 1, 2, 3, 4, 5]

  check (0, 1) in dg == true
  check (0, 5) in dg == false
  check (0, 1) notin dg == false
  check (1, 0) in dg == false
  check (0, 5) notin dg == true
  dg = dg + (0, 5)
  check (0, 5) in dg == true
  dg = dg - (0, 5)
  check (0, 5) in dg == false
  dg = dg + @[(0, 5), (2, 0)]
  check (0, 5) in dg == true
  check (2, 0) in dg == true
  dg = dg - @[(0, 5), (2, 0)]
  check (0, 5) in dg == false
  check (2, 0) in dg == false
  check (0, 2) in dg == false
  dg += @[(0, 5), (2, 0)]
  check (0, 5) in dg == true
  check (2, 0) in dg == true
  dg -= @[(0, 5), (2, 0)]
  check (0, 5) in dg == false
  check (2, 0) in dg == false
  check (0, 2) in dg == false

test "copy graph as graph":
  let original = newGraph()
  original.addEdgesFrom(@[(0, 1), (1, 2)])
  let copied = original.copyAsGraph()
  check copied.isDirected() == false
  check copied.numberOfNodes() == 3
  check copied.numberOfEdges() == 2
  check copied.edges() == @[(0, 1), (1, 2)]

test "copy graph as directed graph":
  let original = newGraph()
  original.addEdgesFrom(@[(0, 1), (1, 2)])
  let copied = original.copyAsDiGraph()
  check copied.isDirected() == true
  check copied.numberOfNodes() == 3
  check copied.numberOfEdges() == 4
  check copied.edges() == @[(0, 1), (1, 0), (1, 2), (2, 1)]

test "copy directed graph as directed graph":
  let original = newDiGraph()
  original.addEdgesFrom(@[(0, 1), (1, 2)])
  let copied = original.copyAsDiGraph()
  check copied.isDirected() == true
  check copied.numberOfNodes() == 3
  check copied.numberOfEdges() == 2
  check copied.edges() == @[(0, 1), (1, 2)]

test "copy directed graph as graph":
  let original = newDiGraph()
  original.addEdgesFrom(@[(0, 1), (1, 0), (1, 2), (2, 1)])
  let copied = original.copyAsGraph()
  check copied.isDirected() == false
  check copied.numberOfNodes() == 3
  check copied.numberOfEdges() == 2
  check copied.edges() == @[(0, 1), (1, 2)]

test "convert graph to directed graph":
  let G = newGraph()
  G.addEdgesFrom(@[(0, 1), (1, 2)])
  let DG = G.toDirected()
  check DG.isDirected() == true
  check DG.numberOfNodes() == 3
  check DG.numberOfEdges() == 4
  check DG.edges() == @[(0, 1), (1, 0), (1, 2), (2, 1)]

test "convert directed graph to graph":
  let DG = newDiGraph()
  DG.addEdgesFrom(@[(0, 1), (1, 0), (1, 2), (2, 1)])
  let G = DG.toUndirected()
  check G.isDirected() == false
  check G.numberOfNodes() == 3
  check G.numberOfEdges() == 2
  check G.edges() == @[(0, 1), (1, 2)]