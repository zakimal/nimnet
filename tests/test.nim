# This is just an example to get you started. You may wish to put all of your
# tests into a single file, or separate them into multiple `test1`, `test2`
# etc. files (better names are recommended, just make sure the name starts with
# the letter 't').
#
# To run these tests, simply execute `nimble test`.

import unittest

import sets
import tables

import nimnet

test "create empty graph":
  let g = newGraph()
  check g.numberOfNodes() == 0

test "create graph with nodes":
  let g = newGraph(@[0, 1, 2, 3, 4, 5])
  check g.numberOfNodes() == 6

test "create graph with edges":
  let g = newGraph(@[(0, 1), (1, 2), (3, 4), (5, 6)])
  check g.numberOfNodes() == 7
  check g.numberOfEdges() == 4

test "add node":
  let g = newGraph()
  check g.numberOfNodes() == 0
  g.addNode(0)
  check g.numberOfNodes() == 1

test "add none":
  let g = newGraph()
  try:
    g.addNode(None)
  except NNError as e:
    check e.msg == "None cannot be a node"

test "add nodes":
  let g = newGraph()
  check g.numberOfNodes() == 0
  g.addNodesFrom(@[0, 1, 2, 3, 4])
  check g.numberOfNodes() == 5

test "remove node":
  let g = newGraph()
  g.addNodesFrom(@[0, 1, 2, 3, 4])
  check g.numberOfNodes() == 5
  g.removeNode(4)
  check g.numberOfNodes() == 4

test "remove nodes":
  let g = newGraph()
  g.addNodesFrom(@[0, 1, 2, 3, 4])
  check g.numberOfNodes() == 5
  g.removeNodesFrom(@[0, 4])
  check g.numberOfNodes() == 3

test "add edge":
  let g = newGraph()
  g.addEdge(0, 1)
  check g.numberOfNodes() == 2
  check g.numberOfEdges() == 1

test "add edges":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2)])
  check g.numberOfNodes() == 3
  check g.numberOfEdges() == 2

test "remove edge":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2)])
  check g.numberOfNodes() == 3
  check g.numberOfEdges() == 2
  g.removeEdge(0, 1)
  check g.numberOfNodes() == 3
  check g.numberOfEdges() == 1

test "remove edges":
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

test "clear edges":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (3, 4)])
  check g.numberOfNodes() == 5
  check g.numberOfEdges() == 3
  g.clearEdges()
  check g.numberOfNodes() == 5
  check g.numberOfEdges() == 0

test "get nodes":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (3, 4)])
  check g.nodes() == @[0, 1, 2, 3, 4]
  var expected = @[0, 1, 2, 3, 4].toHashSet()
  check g.nodesSet() == expected

test "report if node exists or not":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (3, 4)])
  check g.hasNode(0) == true
  check g.hasNode(5) == false
  check g.contains(0) == true
  check g.contains(5) == false

test "get edges":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (3, 4)])
  check g.edges() == @[(0, 1), (1, 2), (3, 4)]
  var expected = @[(0, 1), (1, 2), (3, 4)].toHashSet()
  check g.edgesSet() == expected

test "report if edge exists or not":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (3, 4)])
  check g.hasEdge(0, 1) == true
  check g.hasEdge((0, 1)) == true
  check g.hasEdge(5, 6) == false
  check g.hasEdge((5, 6)) == false
  check g.contains(0, 1) == true
  check g.contains((0, 1)) == true
  check g.contains(5, 6) == false
  check g.contains((5, 6)) == false

test "get adjacency":
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

test "report degree":
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

test "report density":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (3, 4), (5, 5)])
  check g.density() == 4.float / (6 * 5).float

test "get subgraph":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  let ns = @[1, 2, 3].toHashSet()
  let sg = g.subgraph(ns)
  check sg.edges() == @[(1, 2), (1, 3), (2, 3)]

test "get subgraph by edges":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  let es = @[(0, 1), (1, 3), (3, 5)].toHashSet()
  let sg = g.edgeSubgraph(es)
  check sg.edges() == @[(0, 1), (1, 3), (3, 5)]

test "show info":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  check g.info(1) == "node 1 has following properties:\nDegree: 4\nNeighbors: @[0, 2, 3, 4]\n"

test "add star and check it":
  let g = newGraph()
  g.addStar(@[0, 1, 2, 3, 4])
  check g.edges() == @[(0, 1), (0, 2), (0, 3), (0, 4)]
  check g.isStar(@[0, 1, 2, 3, 4]) == true
  check g.isStar(@[0, 1, 2]) == true

test "add path and check it":
  let g = newGraph()
  g.addPath(@[0, 1, 2, 3, 4])
  check g.edges() == @[(0, 1), (1, 2), (2, 3), (3, 4)]
  check g.isPath(@[0, 1, 2, 3, 4]) == true
  check g.isPath(@[0, 1, 2]) == true

test "add cycle and check it":
  let g = newGraph()
  g.addCycle(@[0, 1, 2, 3, 4])
  check g.edges() == @[(0, 1), (0, 4), (1, 2), (2, 3), (3, 4)]
  check g.isCycle(@[0, 1, 2, 3, 4]) == true
  check g.isCycle(@[0, 1, 2]) == false

test "get non-neighbors":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  check g.nonNeighbors(1) == @[5]
  check g.nonNeighbors(3) == @[0, 4]

test "get common neighbors":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  check g.commonNeighbors(1, 4) == @[2]
  check g.commonNeighbors(3, 4) == @[1, 2]

test "get non-edges":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5)])
  check g.nonEdges() == @[
    (0, 2), (0, 3), (0, 4), (0, 5), (1, 5), (2, 5), (3, 4), (4, 5)
  ]
  check g.nonEdges.toHashSet() == @[
    (0, 2), (0, 3), (0, 4), (0, 5), (1, 5), (2, 5), (3, 4), (4, 5)
  ].toHashSet()

test "get nodes with selfloop edge":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 1), (1, 2), (2, 3)])
  check g.nodesWithSelfloopEdge() == @[1]

test "get selfloop edges":
  let g = newGraph()
  g.addEdgesFrom(@[(0, 1), (1, 1), (1, 2), (2, 3)])
  check g.selfloopEdges() == @[(1, 1)]

test "custom operators":
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

