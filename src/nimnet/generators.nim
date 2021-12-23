import deques
import math

import ../nimnet.nim

# -------------------------------------------------------------------
# TODO:
# Atlas
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Classic Graph Generator
# TODO: complete_multipartite_graph(*subset_sizes)
# TODO: turan_graph(n, r)
# -------------------------------------------------------------------

iterator treeEdges(n: int, r: int): Edge =
  if n == 0:
    raise newNNError("n == 0")

  var nodes: seq[int] = @[]
  for i in 0..<n:
    nodes.add(i)
  var parents = @[0].toDeque()
  var idx = 0

  var flag = false
  while len(parents) != 0:
    let source = parents.popFirst()
    for i in 0..<r:
      idx += 1
      if idx == n: # StopIteration
        flag = true
        break
      let target = nodes[idx]
      parents.addLast(target)
      yield (source, target)
    if flag:
      break

proc balancedTree*(r: int, h: int): Graph =
  var n: int
  if r == 1:
    n = h + 1
  else:
    n = (1 - r ^ (h + 1)) div (1 - r)
  let G = newGraph()
  for edge in treeEdges(n, r):
    G.addEdge(edge)
  return G
proc balancedDiTree*(r: int, h: int): DiGraph =
  var n: int
  if r == 1:
    n = h + 1
  else:
    n = (1 - r ^ (h + 1)) div (1 - r)
  let DG = newDiGraph()
  for edge in treeEdges(n, r):
    DG.addEdge(edge)
  return DG

proc barbellGraph*(m1, m2: int): Graph =
  ## Returns barbell graph: 2 complete graph connected by a path
  ## 2 <= m1, 0 <= m2
  ## barbell graph contains `2*m1+m2` nodes.
  ## left barbell: `0, ..., m1-1` (clique)
  ## path: `m1, ..., m1+m2-1` (path)
  ## right barbell: `m1+m2, ..., 2*m1+m2` (clique)
  ## if `m2 == 0`, this is just 2 complete graphs jointed together
  if m1 < 2:
    raise newNNError("invalid graph description: 2 <= m1")
  if m2 < 0:
    raise newNNError("invalid graph description: 0 <= m2")

  # left barbell
  let G = newGraph()
  for i in 0..<m1:
    for j in 0..<m1:
      if i == j:
        continue
      G.addEdge(i, j)
  # path
  for i in (m1-1)..<(m1+m2):
    G.addEdge(i, i + 1)
  # right barbell
  for i in (m1+m2)..<(2*m1+m2):
    for j in (m1+m2)..<(2*m1+m2):
      if i == j:
        continue
      G.addEdge(i, j)
  return G
proc barbellDiGraph*(m1, m2: int): DiGraph =
  ## Returns barbell graph: 2 complete graph connected by a path
  ## 2 <= m1, 0 <= m2
  ## barbell graph contains `2*m1+m2` nodes.
  ## left barbell: `0, ..., m1-1` (clique)
  ## path: `m1, ..., m1+m2-1` (path)
  ## right barbell: `m1+m2, ..., 2*m1+m2` (clique)
  ## if `m2 == 0`, this is just 2 complete graphs jointed together
  if m1 < 2:
    raise newNNError("invalid graph description: 2 <= m1")
  if m2 < 0:
    raise newNNError("invalid graph description: 0 <= m2")

  # left barbell
  let DG = newDiGraph()
  for i in 0..<m1:
    for j in 0..<m1:
      if i == j:
        continue
      DG.addEdge(i, j)
      DG.addEdge(j, i)
  # path
  for i in (m1-1)..<(m1+m2):
    DG.addEdge(i, i + 1)
    DG.addEdge(i + 1, i)
  # right barbell
  for i in (m1+m2)..<(2*m1+m2):
    for j in (m1+m2)..<(2*m1+m2):
      if i == j:
        continue
      DG.addEdge(i, j)
      DG.addEdge(j, i)
  return DG

proc binomialTree*(n: int): Graph =
  let G = newGraph(@[1])
  var N = 1
  for i in 0..<n:
    var edges: seq[Edge] = @[]
    for (u, v) in G.edges():
      edges.add((u + N, v + N))
    G.addEdgesFrom(edges)
    G.addEdge(0, N)
    N *= 2
  return G

proc completeGraph*(n: int): Graph =
  let G = newGraph()
  for i in 0..<n:
    for j in i..<n:
      if i == j:
        continue
      G.addEdge(i, j)
  return G
proc completeDiGraph*(n: int): DiGraph =
  let DG = newDiGraph()
  for i in 0..<n:
    for j in 0..<n:
      if i == j:
        continue
      DG.addEdge(i, j)
  return DG

proc circularLadderGraph*(n: int): Graph =
  let G = newGraph()
  for i in 0..<(n-1):
    G.addEdge(i, i + 1)
    G.addEdge(i + n, i + n + 1)
  for i in 0..<n:
    G.addEdge(i, i + n)
  G.addEdge(0, n - 1)
  G.addEdge(n, 2 * n - 1)
  return G
proc circularLadderDiGraph*(n: int): DiGraph =
  let DG = newDiGraph()
  for i in 0..<(n-1):
    DG.addEdge(i, i + 1)
    DG.addEdge(i + 1, i)
    DG.addEdge(i + n, i + n + 1)
    DG.addEdge(i + n + 1, i + n)
  for i in 0..<n:
    DG.addEdge(i, i + n)
    DG.addEdge(i + n, i)
  DG.addEdge(0, n - 1)
  DG.addEdge(n - 1, 0)
  DG.addEdge(n, 2 * n - 1)
  DG.addEdge(2 * n - 1, n)
  return DG

proc circulantGraph*(n: int, offsets: seq[int]): Graph =
  let G = newGraph()
  for i in 0..<n:
    for j in offsets:
      G.addEdge(i, (i - j + n) mod n)
      G.addEdge(i, (i + j) mod n)
  return G
proc circulantDiGraph*(n: int, offsets: seq[int]): DiGraph =
  let DG = newDiGraph()
  for i in 0..<n:
    for j in offsets:
      DG.addEdge(i, (i - j + n) mod n)
      DG.addEdge((i - j + n) mod n, i)
      DG.addEdge(i, (i + j) mod n)
      DG.addEdge((i + j) mod n, i)
  return DG

proc cycleGraph*(n: int): Graph =
  let G = newGraph()
  if n < 0:
    raise newNNError("n must be greater than or equal to 0")
  if n == 0:
    return G
  for i in 0..<(n-1):
    G.addEdge(i, i + 1)
  G.addEdge(n - 1, 0)
  return G
proc cycleDiGraph*(n: int): DiGraph =
  let DG = newDiGraph()
  if n < 0:
    raise newNNError("n must be greater than or equal to 0")
  if n == 0:
    return DG
  for i in 0..<(n-1):
    DG.addEdge(i, i + 1)
  DG.addEdge(n - 1, 0)
  return DG

proc dorogovtsevGoltsevMendesGraph*(n: int): Graph =
  let G = newGraph()
  G.addEdge(0, 1)
  if n == 0:
    return G
  var newNode = 2
  for i in 1..<(n+1):
    let lastGenerationEdges = G.edges()
    let numberOfEdgesInLastGeneration = len(lastGenerationEdges)
    for j in 0..<numberOfEdgesInLastGeneration:
      G.addEdge(newNode, lastGenerationEdges[j].u)
      G.addEdge(newNode, lastGenerationEdges[j].v)
      newNode += 1
  return G

proc emptyGraph*(n: int): Graph =
  let G = newGraph()
  for i in 0..<n:
    G.addNode(i)
  return G
proc emptyDiGraph*(n: int): DiGraph =
  let DG = newDiGraph()
  for i in 0..<n:
    DG.addNode(i)
  return DG

proc fullRaryTree*(r: int, n: int): Graph =
  let G = newGraph()
  for edge in treeEdges(n, r):
    G.addEdge(edge)
  return G
proc fullRaryDiTree*(r: int, n: int): DiGraph =
  let DG = newDiGraph()
  for edge in treeEdges(n, r):
    DG.addEdge(edge)
  return DG

proc ladderGraph*(n: int): Graph =
  let G = newGraph()
  for i in 0..<(n-1):
    G.addEdge(i, i + 1)
    G.addEdge(i + n, i + n + 1)
  for i in 0..<n:
    G.addEdge(i, i + n)
  return G
proc ladderDiGraph*(n: int): DiGraph =
  let DG = newDiGraph()
  for i in 0..<(n-1):
    DG.addEdge(i, i + 1)
    DG.addEdge(i + 1, i)
    DG.addEdge(i + n, i + n + 1)
    DG.addEdge(i + n + 1, i + n)
  for i in 0..<n:
    DG.addEdge(i, i + n)
    DG.addEdge(i + n, i)
  return DG

proc lollipopGraph*(m: int, n: int): Graph =
  if m < 2:
    raise newNNError("invalid graph description: 2 <= m")
  if n < 0:
    raise newNNError("invalid graph description: 0 <= n")

  let G = newGraph()
  for i in 0..<m:
    for j in (i+1)..<m:
      G.addEdge(i, j)
  for i in (m-1)..<(m+n-1):
    G.addEdge(i, i + 1)
  return G

proc nullGraph*(): Graph =
  return newGraph()
proc nullDiGraph*(): DiGraph =
  return newDiGraph()

proc pathGraph*(n: int): Graph =
  let G = newGraph()
  for i in 0..<(n-1):
    G.addEdge(i, i + 1)
  return G
proc pathDiGraph*(n: int): DiGraph =
  let DG = newDiGraph()
  for i in 0..<(n-1):
    DG.addEdge(i, i + 1)
  return DG

proc starGraph*(n: int): Graph =
  let G = newGraph()
  G.addNode(0)
  for i in 1..n:
    G.addEdge(0, i)
  return G
proc starDiGraph*(n: int): DiGraph =
  let DG = newDiGraph()
  DG.addNode(0)
  for i in 1..n:
    DG.addEdge(0, i)
  return DG

proc trivialGraph*(): Graph =
  return newGraph(@[0])
proc trivialDiGraph*(): DiGraph =
  return newDiGraph(@[0])

proc wheelGraph*(n: int): Graph =
  let G = newGraph()
  for i in 1..<n:
    G.addEdge(0, i)
  for i in 1..<n:
    if i == n - 1:
      G.addEdge(i, 1)
    else:
      G.addEdge(i, i + 1)
  return G
proc wheelDiGraph*(n: int): DiGraph =
  let DG = newDiGraph()
  for i in 1..<n:
    DG.addEdge(0, i)
    DG.addEdge(i, 0)
  for i in 1..<n:
    if i == n - 1:
      DG.addEdge(i, 1)
      DG.addEdge(1, i)
    else:
      DG.addEdge(i, i + 1)
      DG.addEdge(i + 1, i)
  return DG

# -------------------------------------------------------------------
# TODO:
# Expanders
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Lattice
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Small
# TODO: make_small_graph(graph_description[, …])
# TODO: LCF_graph(n, shift_list, repeats[, create_using])
# TODO: desargues_graph([create_using])
# TODO: dodecahedral_graph([create_using])
# TODO: heawood_graph([create_using])
# TODO: moebius_kantor_graph([create_using])
# TODO: pappus_graph()
# -------------------------------------------------------------------

proc bullGraph*(): Graph =
  let G = newGraph()
  G.addEdgesFrom(@[(0, 1), (0, 2), (1, 2), (1, 3), (2, 4)])
  return G

proc chvatalGraph*(): Graph =
  let G = newGraph()
  G.addEdgesFrom(@[(0, 1), (0, 4), (0, 6), (0, 9), (1, 2), (1, 5), (1, 7), (2, 3), (2, 6), (2, 8), (3, 4), (3, 7), (3, 9), (4, 5), (4, 8), (5, 10), (5, 11), (6, 10), (6, 11), (7, 8), (7, 11), (8, 10), (9, 10), (9, 11)])
  return G

proc cubicalGraph*(): Graph =
  let G = newGraph()
  G.addEdgesFrom(@[(0, 1), (0, 3), (0, 4), (1, 2), (1, 7), (2, 3), (2, 6), (3, 5), (4, 5), (4, 7), (5, 6), (6, 7)])
  return G

proc diamondGraph*(): Graph =
  let G = newGraph()
  G.addEdgesFrom(@[(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])
  return G

proc fruchtGraph*(): Graph =
  let G = cycleGraph(7)
  G.addEdgesFrom(@[(0, 7), (1, 7), (2, 8), (3, 9), (4, 9), (5, 10), (6, 10), (7, 11), (8, 11), (8, 9), (10, 11)])
  return G
proc fruchtDiGraph*(): DiGraph =
  let DG = cycleDiGraph(7)
  DG.addEdgesFrom(@[(0, 7), (1, 7), (2, 8), (3, 9), (4, 9), (5, 10), (6, 10), (7, 11), (8, 11), (8, 9), (10, 11)])
  return DG

proc hoffmanSingletonGraph*(): Graph =
  let G = newGraph()
  G.addEdgesFrom(@[(0, 1), (0, 2), (0, 3), (0, 6), (0, 7), (0, 8), (0, 9), (1, 12), (1, 17), (1, 26), (1, 27), (1, 28), (1, 29), (2, 10), (2, 11), (2, 13), (2, 14), (2, 15), (2, 16), (3, 4), (3, 5), (3, 30), (3, 35), (3, 40), (3, 45), (4, 11), (4, 17), (4, 34), (4, 39), (4, 44), (4, 49), (5, 10), (5, 12), (5, 33), (5, 38), (5, 43), (5, 48), (6, 18), (6, 22), (6, 31), (6, 39), (6, 43), (6, 47), (7, 19), (7, 23), (7, 34), (7, 37), (7, 41), (7, 48), (8, 20), (8, 24), (8, 33), (8, 36), (8, 42), (8, 49), (9, 21), (9, 25), (9, 32), (9, 38), (9, 44), (9, 46), (10, 17), (10, 18), (10, 19), (10, 20), (10, 21), (11, 12), (11, 32), (11, 37), (11, 42), (11, 47), (12, 31), (12, 36), (12, 41), (12, 46), (13, 22), (13, 26), (13, 30), (13, 36), (13, 44), (13, 48), (14, 23), (14, 27), (14, 31), (14, 38), (14, 40), (14, 49), (15, 24), (15, 28), (15, 34), (15, 35), (15, 43), (15, 46), (16, 25), (16, 29), (16, 33), (16, 39), (16, 41), (16, 45), (17, 22), (17, 23), (17, 24), (17, 25), (18, 26), (18, 32), (18, 35), (18, 41), (18, 49), (19, 27), (19, 30), (19, 39), (19, 42), (19, 46), (20, 28), (20, 31), (20, 37), (20, 44), (20, 45), (21, 29), (21, 34), (21, 36), (21, 40), (21, 47), (22, 33), (22, 37), (22, 40), (22, 46), (23, 32), (23, 36), (23, 43), (23, 45), (24, 30), (24, 38), (24, 41), (24, 47), (25, 31), (25, 35), (25, 42), (25, 48), (26, 34), (26, 38), (26, 42), (26, 45), (27, 33), (27, 35), (27, 44), (27, 47), (28, 32), (28, 39), (28, 40), (28, 48), (29, 30), (29, 37), (29, 43), (29, 49), (30, 31), (30, 32), (31, 34), (32, 33), (33, 34), (35, 36), (35, 37), (36, 39), (37, 38), (38, 39), (40, 41), (40, 42), (41, 44), (42, 43), (43, 44), (45, 46), (45, 47), (46, 49), (47, 48), (48, 49)])
  return G

proc houseGraph*(): Graph =
  let G = newGraph()
  G.addEdgesFrom(@[(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
  return G

proc houseWithXGraph*(): Graph =
  let G = newGraph()
  G.addEdgesFrom(@[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
  return G

proc icosahedralGraph*(): Graph =
  let G = newGraph()
  G.addEdgesFrom(@[(0, 1), (0, 5), (0, 7), (0, 8), (0, 11), (1, 2), (1, 5), (1, 6), (1, 8), (2, 3), (2, 6), (2, 8), (2, 9), (3, 4), (3, 6), (3, 9), (3, 10), (4, 5), (4, 6), (4, 10), (4, 11), (5, 6), (5, 11), (7, 8), (7, 9), (7, 10), (7, 11), (8, 9), (9, 10), (10, 11)])
  return G

proc krackhardtKiteGraph*(): Graph =
  let G = newGraph()
  G.addEdgesFrom(@[(0, 1), (0, 2), (0, 3), (0, 5), (1, 3), (1, 4), (1, 6), (2, 3), (2, 5), (3, 4), (3, 5), (3, 6), (4, 6), (5, 6), (5, 7), (6, 7), (7, 8), (8, 9)])
  return G

proc octahedralGraph*(): Graph =
  let G = newGraph()
  G.addEdgesFrom(@[(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)])
  return G

proc petersenGraph*(): Graph =
  let G = newGraph()
  G.addEdgesFrom(@[(0, 1), (0, 4), (0, 5), (1, 2), (1, 6), (2, 3), (2, 7), (3, 4), (3, 8), (4, 9), (5, 7), (5, 8), (6, 8), (6, 9), (7, 9)])
  return G

proc sedgewickMazeGraph*(): Graph =
  let G = newGraph()
  G.addEdgesFrom(@[(0, 2), (0, 5), (0, 7), (1, 7), (2, 6), (3, 4), (3, 5), (4, 5), (4, 6), (4, 7)])
  return G
proc sedgewickMazeDiGraph*(): DiGraph =
  let DG = newDiGraph()
  DG.addEdgesFrom(@[(0, 2), (0, 5), (0, 7), (1, 7), (2, 6), (3, 4), (3, 5), (4, 5), (4, 6), (4, 7)])
  return DG

proc tetrahedralGraph*(): Graph =
  let G = newGraph()
  G.addEdgesFrom(@[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
  return G
proc tetrahedralDiGraph*(): DiGraph =
  let DG = newDiGraph()
  DG.addEdgesFrom(@[(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2)])
  return DG

proc truncatedCubeGraph*(): Graph =
  let G = newGraph()
  G.addEdgesFrom(@[(0, 1), (0, 2), (0, 4), (1, 11), (1, 14), (2, 3), (2, 4), (3, 6), (3, 8), (4, 5), (5, 16), (5, 18), (6, 7), (6, 8), (7, 10), (7, 12), (8, 9), (9, 17), (9, 20), (10, 11), (10, 12), (11, 14), (12, 13), (13, 21), (13, 22), (14, 15), (15, 19), (15, 23), (16, 17), (16, 18), (17, 20), (18, 19), (19, 23), (20, 21), (21, 22), (22, 23)])
  return G

proc truncatedTetrahedronGraph*(): Graph =
  let G = newGraph()
  G.addEdgesFrom(@[(0, 1), (0, 2), (0, 9), (1, 2), (1, 6), (2, 3), (3, 4), (3, 11), (4, 5), (4, 11), (5, 6), (5, 7), (6, 7), (7, 8), (8, 9), (8, 10), (9, 10), (10, 11)])
  return G
proc truncatedTetrahedronDiGraph*(): DiGraph =
  let DG = newDiGraph()
  DG.addEdgesFrom(@[(0, 1), (0, 2), (0, 9), (1, 2), (1, 6), (2, 3), (3, 4), (3, 11), (4, 5), (4, 11), (5, 6), (5, 7), (6, 7), (7, 8), (8, 9), (8, 10), (9, 10), (10, 11)])
  return DG

proc tutteGraph*(): Graph =
  let G = newGraph()
  G.addEdgesFrom(@[(0, 1), (0, 2), (0, 3), (1, 4), (1, 26), (2, 10), (2, 11), (3, 18), (3, 19), (4, 5), (4, 33), (5, 6), (5, 29), (6, 7), (6, 27), (7, 8), (7, 14), (8, 9), (8, 38), (9, 10), (9, 37), (10, 39), (11, 12), (11, 39), (12, 13), (12, 35), (13, 14), (13, 15), (14, 34), (15, 16), (15, 22), (16, 17), (16, 44), (17, 18), (17, 43), (18, 45), (19, 20), (19, 45), (20, 21), (20, 41), (21, 22), (21, 23), (22, 40), (23, 24), (23, 27), (24, 25), (24, 32), (25, 26), (25, 31), (26, 33), (27, 28), (28, 29), (28, 32), (29, 30), (30, 31), (30, 33), (31, 32), (34, 35), (34, 38), (35, 36), (36, 37), (36, 39), (37, 38), (40, 41), (40, 44), (41, 42), (42, 43), (42, 45), (43, 44)])
  return G

# -------------------------------------------------------------------
# TODO:
# Random Graph
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Duplication Divergence
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Degree Sequence
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Random Clutered
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Directed
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Geometric
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Line Graph
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Ego Graph
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Stochastic
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Autonmous System Network as Graph
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Intersection
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Social Graph Generator
# -------------------------------------------------------------------

proc karateClubGraph*(): Graph =
  let edges = @[
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (0, 6), (0, 7), (0, 8), (0, 10), (0, 11),
    (0, 12), (0, 13), (0, 17), (0, 19), (0, 21),
    (0, 31), (1, 2), (1, 3), (1, 7), (1, 13),
    (1, 17), (1, 19), (1, 21), (1, 30), (2, 3),
    (2, 7), (2, 8), (2, 9), (2, 13), (2, 27),
    (2, 28), (2, 32), (3, 7), (3, 12), (3, 13),
    (4, 6), (4, 10), (5, 6), (5, 10), (5, 16),
    (6, 16), (8, 30), (8, 32), (8, 33), (9, 33),
    (13, 33), (14, 32), (14, 33), (15, 32),
    (15, 33), (18, 32), (18, 33), (19, 33),
    (20, 32), (20, 33), (22, 32), (22, 33),
    (23, 25), (23, 27), (23, 29), (23, 32),
    (23, 33), (24, 25), (24, 27), (24, 31),
    (25, 31), (26, 29), (26, 33), (27, 33),
    (28, 31), (28, 33), (29, 32), (29, 33),
    (30, 32), (30, 33), (31, 32), (31, 33),
    (32, 33)
  ]
  var ret = newGraph()
  ret.addEdgesFrom(edges)
  return ret

proc davisSouthernWomenGraph*(): Graph =
  let edges = @[
    (0, 18), (0, 19), (0, 20), (0, 21), (0, 22),
    (0, 23), (0, 25), (0, 26), (1, 18), (1, 19),
    (1, 20), (1, 22), (1, 23), (1, 24), (1, 25),
    (2, 19), (2, 20), (2, 21), (2, 22), (2, 23),
    (2, 24), (2, 25), (2, 26), (3, 18), (3, 20),
    (3, 21), (3, 22), (3, 23), (3, 24), (3, 25),
    (4, 20), (4, 21), (4, 22), (4, 24), (5, 20),
    (5, 22), (5, 23), (5, 25), (6, 22), (6, 23),
    (6, 24), (6, 25), (7, 23), (7, 25), (7, 26),
    (8, 22), (8, 24), (8, 25), (8, 26), (9, 24),
    (9, 25), (9, 26), (9, 29), (10, 25), (10, 26),
    (10, 27), (10, 29), (11, 25), (11, 26),
    (11, 27), (11, 29), (11, 30), (11, 31),
    (12, 24), (12, 25), (12, 26), (12, 27),
    (12, 29), (12, 30), (12, 31), (13, 23),
    (13, 24), (13, 26), (13, 27), (13, 28),
    (13, 29), (13, 30), (13, 31), (14, 24),
    (14, 25), (14, 27), (14, 28), (14, 29),
    (15, 25), (15, 26), (16, 26), (16, 28),
    (17, 26), (17, 28)
  ]
  var ret = newGraph()
  ret.addEdgesFrom(edges)
  return ret

proc florentineFamiliesGraph*(): Graph =
  let edges = @[
    (0, 1), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
    (2, 3), (2, 4), (2, 5), (3, 4), (3, 11), (4, 6),
    (4, 11), (6, 7), (7, 12), (8, 12), (8, 13),
    (9, 10), (11, 12), (12, 14)
  ]
  var ret = newGraph()
  ret.addEdgesFrom(edges)
  return ret

proc lesMiserablesGraph*(): Graph =
  let edges = @[
    (0, 1), (1, 2), (1, 3), (1, 4), (1, 5),
    (1, 6), (1, 7), (1, 8), (1, 9), (1, 10),
    (2, 3), (2, 10), (3, 10), (10, 11),
    (10, 12), (10, 13), (10, 14), (10, 15),
    (10, 23), (10, 24), (10, 25), (10, 26),
    (10, 27), (10, 28), (10, 29), (10, 31),
    (10, 32), (10, 33), (10, 34), (10, 35),
    (10, 36), (10, 37), (10, 38), (10, 43),
    (10, 44), (10, 48), (10, 49), (10, 51),
    (10, 55), (10, 58), (10, 64), (10, 68),
    (10, 69), (10, 70), (10, 71), (10, 72),
    (12, 23), (16, 17), (16, 18), (16, 19),
    (16, 20), (16, 21), (16, 22), (16, 23),
    (17, 18), (17, 19), (17, 20), (17, 21),
    (17, 22), (17, 23), (17, 26), (17, 55),
    (18, 19), (18, 20), (18, 21), (18, 22),
    (18, 23), (19, 20), (19, 21), (19, 22),
    (19, 23), (20, 21), (20, 22), (20, 23),
    (21, 22), (21, 23), (22, 23), (23, 24),
    (23, 25), (23, 27), (23, 29), (23, 30),
    (23, 31), (24, 25), (24, 26), (24, 27),
    (24, 41), (24, 42), (24, 50), (24, 68),
    (24, 69), (24, 70), (25, 26), (25, 27),
    (25, 39), (25, 40), (25, 41), (25, 42),
    (25, 48), (25, 55), (25, 68), (25, 69),
    (25, 70), (25, 71), (25, 75), (26, 27),
    (26, 43), (26, 49), (26, 51), (26, 54),
    (26, 55), (26, 72), (27, 28), (27, 29),
    (27, 31), (27, 33), (27, 43), (27, 48),
    (27, 58), (27, 68), (27, 69), (27, 70),
    (27, 71), (27, 72), (28, 44), (28, 45),
    (29, 34), (29, 35), (29, 36), (29, 37),
    (29, 38), (30, 31), (34, 35), (34, 36),
    (34, 37), (34, 38), (35, 36), (35, 37),
    (35, 38), (36, 37), (36, 38), (37, 38),
    (39, 52), (39, 55), (41, 42), (41, 55),
    (41, 57), (41, 62), (41, 68), (41, 69),
    (41, 70), (41, 71), (41, 75), (46, 47),
    (46, 48), (48, 55), (48, 57), (48, 58),
    (48, 59), (48, 60), (48, 61), (48, 62),
    (48, 63), (48, 64), (48, 65), (48, 66),
    (48, 68), (48, 69), (48, 71), (48, 73),
    (48, 74), (48, 75), (48, 76), (49, 50),
    (49, 51), (49, 54), (49, 55), (49, 56),
    (51, 52), (51, 53), (51, 54), (51, 55),
    (54, 55), (55, 56), (55, 57), (55, 58),
    (55, 59), (55, 61), (55, 62), (55, 63),
    (55, 64), (55, 65), (57, 58), (57, 59),
    (57, 61), (57, 62), (57, 63), (57, 64),
    (57, 65), (57, 67), (58, 59), (58, 60),
    (58, 61), (58, 62), (58, 63), (58, 64),
    (58, 65), (58, 66), (58, 70), (58, 76),
    (59, 60), (59, 61), (59, 62), (59, 63),
    (59, 64), (59, 65), (59, 66), (60, 61),
    (60, 62), (60, 63), (60, 64), (60, 65),
    (60, 66), (61, 62), (61, 63), (61, 64),
    (61, 65), (61, 66), (62, 63), (62, 64),
    (62, 65), (62, 66), (62, 76), (63, 64),
    (63, 65), (63, 66), (63, 76), (64, 65),
    (64, 66), (64, 76), (65, 66), (65, 76),
    (66, 76), (68, 69), (68, 70), (68, 71),
    (68, 75), (69, 70), (69, 71), (69, 75),
    (70, 71), (70, 75), (71, 75), (73, 74)
  ]
  var ret = newGraph()
  ret.addEdgesFrom(edges)
  return ret

# -------------------------------------------------------------------
# TODO:
# Community
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Spectral
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Trees
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Non-isomorphic Trees
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Triads
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Joint Degre Sequence
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Mycielshi
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Harary Graph
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Cographs
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Interval Graph
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Sudoku
# -------------------------------------------------------------------