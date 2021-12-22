import unittest

import nimnet
import nimnet/readwrite

test "read adjlist as graph":
  let G = readAdjlistAsGraph("tests/input/test.adjlist", delimiter=' ')
  check G.isDirected() == false
  check G.edges() == @[
    (0, 1), (0, 4),
    (1, 2), (1, 4),
    (2, 3),
    (3, 4), (3, 5)
  ]

test "read adjlist as directed graph":
  let DG = readAdjlistAsDiGraph("tests/input/test.adjlist", delimiter=' ')
  check DG.isDirected() == true
  check DG.edges() == @[
    (0, 1), (0, 4),
    (1, 0), (1, 2), (1, 4),
    (2, 1), (2, 3),
    (3, 2), (3, 4), (3, 5),
    (4, 0), (4, 1), (4, 3),
    (5, 3)
  ]

test "read edgelist as graph":
  let G = readEdgelistAsGraph("tests/input/test.edgelist", delimiter=' ')
  check G.isDirected() == false
  check G.edges() == @[(0, 1), (0, 4), (1, 2), (1, 4), (2, 3), (3, 5), (4, 5)]

test "read edgelist as directed graph":
  let DG = readEdgelistAsDiGraph("tests/input/test.edgelist", delimiter=' ')
  check DG.isDirected() == true
  check DG.edges() == @[(0, 1), (0, 4), (1, 0), (1, 2), (1, 4), (2, 3), (3, 5), (4, 5)]
