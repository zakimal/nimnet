import tables
import strformat
import sequtils
import sets
import math
import heapqueue
import algorithm
import deques

import ../nimnet

# -------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------

proc groups(manyToOne: Table[Node, int]): Table[int, HashSet[Node]] =
  # {1: 1, 2: 1, 3: 2, 4: 3, 5: 3} -> {1: {1, 2}, 2: {3}, 3: {4, 5}}
  var oneToMany = initTable[int, HashSet[Node]]()
  for (k, v) in manyToOne.pairs():
    if v notin oneToMany:
      oneToMany[v] = initHashSet[Node]()
    oneToMany[v].incl(k)
  return oneToMany

type UnionFind = ref object of RootObj
  elements: HashSet[Node]
  parents: Table[Node, Node]
  weights: Table[Node, int]
proc newUnionFind(): UnionFind =
  var uf = UnionFind()
  uf.elements = initHashSet[Node]()
  uf.parents = initTable[Node, Node]()
  uf.weights = initTable[Node, int]()
  return uf
proc newUnionFind(elements: HashSet[Node]): UnionFind =
  var uf = UnionFind()
  uf.elements = elements
  uf.parents = initTable[Node, Node]()
  uf.weights = initTable[Node, int]()
  for x in elements:
    uf.weights[x] = 1
    uf.parents[x] = x
  return uf
proc `[]`(uf: UnionFind, node: Node): Node =
  if node notin uf.parents:
    uf.parents[node] = node
    uf.weights[node] = 1
    return node
  var path = @[node]
  var root = uf.parents[node]
  while root != path[^1]:
    path.add(root)
    root = uf.parents[root]
  for ancestor in path:
    uf.parents[ancestor] = root
  return root
iterator toSets(uf: UnionFind): HashSet[Node] =
  for x in uf.parents.keys():
    discard uf[x]
  for g in groups(uf.parents).values():
    yield g
proc union(uf: UnionFind, nodes: HashSet[Node]) =
  var rootsWithWeight: seq[tuple[node: Node, weight: int]] = @[]
  for x in nodes:
    rootsWithWeight.add((-uf.weights[uf[x]], uf[x]))
  rootsWithWeight = rootsWithWeight.toHashSet.toSeq()
  rootsWithWeight.sort()

  var roots: seq[Node] = @[]
  for (_, root) in rootsWithWeight:
    roots.add(root)

  if len(roots) == 0:
    return
  var head = roots[0]
  for r in roots:
    uf.weights[head] += uf.weights[r]
    uf.parents[r] = head

iterator pairwise(nodes: seq[Node], cyclic: bool = false): tuple[u, v: Node] =
  if cyclic:
    var i = 0
    let N = len(nodes)
    while true:
      yield (nodes[i mod N], nodes[(i+1) mod N])
  else:
    for i in 0..<(len(nodes) - 1):
      yield (nodes[i], nodes[i+1])

# -------------------------------------------------------------------
# TODO:
# Approximations and Heuristics
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Assortativity
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Asteroidal
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Bipartite
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Boundary
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Chordal
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Clique
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Clutering
# -------------------------------------------------------------------

iterator trianglesAndDegree(G: Graph, nodes: seq[Node] = @[]): tuple[v: Node, len: int, nTri: int, genDegree: Table[int, int]] =
  var nodesNbrs: Table[Node, HashSet[Node]]
  if len(nodes) == 0:
    nodesNbrs = G.adj
  else:
    for n in nodes:
      nodesNbrs[n] = G.adj[n]

  for (v, vNbrs) in nodesNbrs.pairs():
    var vSet = initHashSet[Node]()
    vSet.incl(v)
    let vs = vNbrs - vSet
    var genDegree = initTable[int, int]()
    for w in vs:
      var wSet = initHashSet[Node]()
      wSet.incl(w)
      if len(vs * (G.adj[w] - wSet)) notin genDegree:
        genDegree[len(vs * (G.adj[w] - wSet))] = 1
      else:
        genDegree[len(vs * (G.adj[w] - wSet))] += 1
    var nTriangles = 0
    for (k, val) in genDegree.pairs():
      nTriangles += (k * val)
    yield (v, len(vs), nTriangles, genDegree)
iterator trianglesAndDegree(DG: DiGraph, nodes: seq[Node] = @[]): tuple[v: Node, len: int, nTri: int, genDegree: Table[int, int]] =
  var nodesNbrs: Table[Node, HashSet[Node]]
  if len(nodes) == 0:
    nodesNbrs = DG.succ
  else:
    for n in nodes:
      nodesNbrs[n] = DG.succ[n]

  for (v, vNbrs) in nodesNbrs.pairs():
    var vSet = initHashSet[Node]()
    vSet.incl(v)
    let vs = vNbrs - vSet
    var genDegree = initTable[int, int]()
    for w in vs:
      var wSet = initHashSet[Node]()
      wSet.incl(w)
      if len(vs * (DG.succ[w] - wSet)) notin genDegree:
        genDegree[len(vs * (DG.succ[w] - wSet))] = 1
      else:
        genDegree[len(vs * (DG.succ[w] - wSet))] += 1
    var nTriangles = 0
    for (k, val) in genDegree.pairs():
      nTriangles += (k * val)
    yield (v, len(vs), nTriangles, genDegree)

proc triangles*(
  G: Graph,
  nodes: seq[Node] = @[]
): Table[Node, int] =
  if len(nodes) == 1 and nodes[0] in G.nodesSet():
    return {nodes[0]: trianglesAndDegree(G, nodes).toSeq()[0].nTri div 2}.toTable()
  var ret = initTable[Node, int]()
  for (v, d, t, _) in G.trianglesAndDegree(nodes):
    ret[v] = t div 2
  return ret

proc transitivity*(G: Graph): float =
  var trianglesContri: seq[tuple[t: int, d: int]] = @[]
  for (v, d, t, _) in trianglesAndDegree(G):
    trianglesContri.add((t, d * (d - 1)))
  if len(trianglesContri) == 0:
    return 0.0
  var triangles = 0
  var contri = 0
  for i in 0..<len(trianglesContri):
    triangles += trianglesContri[i].t
    contri += trianglesContri[i].d
  if triangles == 0:
    return 0.0
  return triangles.float / contri.float
proc transitivity*(DG: DiGraph): float =
  var trianglesContri: seq[tuple[t: int, d: int]] = @[]
  for (v, d, t, _) in trianglesAndDegree(DG):
    trianglesContri.add((t, d * (d - 1)))
  if len(trianglesContri) == 0:
    return 0.0
  var triangles = 0
  var contri = 0
  for i in 0..<len(trianglesContri):
    triangles += trianglesContri[i].t
    contri += trianglesContri[i].d
  if triangles == 0:
    return 0.0
  return triangles.float / contri.float

# -------------------------------------------------------------------
# TODO:
# Coloring
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Communicability
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Communities
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Components
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Cores
# -------------------------------------------------------------------

proc coreNumber*(g: Graph): Table[Node, int] =
  if 0 < g.numberOfSelfloop():
    raise newNNError("input graph has self loops which is not permitted")
  var degrees = g.degree()
  var cmpDegree = proc (x, y: Node): int =
    result = system.cmp(degrees[x], degrees[y])
  var nodes = g.nodes()
  nodes.sort(cmpDegree)
  var binBoundaries = @[0]
  var currDegree = 0
  for i, v in nodes:
    if currDegree < degrees[v] :
      for d in 0..<(degrees[v] - currDegree):
        binBoundaries.add(i)
      currDegree = degrees[v]
  var nodePos: Table[Node, int] = initTable[Node, int]()
  for pos, v in nodes:
    nodePos[v] = pos
  var core = degrees
  var nbrs: Table[Node, seq[Node]] = initTable[Node, seq[Node]]()
  for v in g.nodes():
    nbrs[v] = g.neighbors(v)
  for v in nodes:
    for u in nbrs[v]:
      if core[v] < core[u]:
        nbrs[u].delete(nbrs[u].find(v))
        var pos = nodePos[u]
        var binStart = binBoundaries[core[u]]
        nodePos[u] = binStart
        nodePos[nodes[binStart]] = pos
        swap(nodes[binStart], nodes[pos])
        binBoundaries[core[u]] += 1
        core[u] -= 1
  return core
proc coreNumber*(dg: DiGraph): Table[Node, int] =
  if 0 < dg.numberOfSelfloop():
    raise newNNError("input graph has self loops which is not permitted")
  var degrees = dg.degree()
  var cmpDegree = proc (x, y: Node): int =
    result = system.cmp(degrees[x], degrees[y])
  var nodes = dg.nodes()
  nodes.sort(cmpDegree)
  var binBoundaries = @[0]
  var currDegree = 0
  for i, v in nodes:
    if currDegree < degrees[v] :
      for d in 0..<(degrees[v] - currDegree):
        binBoundaries.add(i)
      currDegree = degrees[v]
  var nodePos: Table[Node, int] = initTable[Node, int]()
  for pos, v in nodes:
    nodePos[v] = pos
  var core = degrees
  var nbrs: Table[Node, seq[Node]] = initTable[Node, seq[Node]]()
  for v in dg.nodes():
    nbrs[v] = dg.neighbors(v)
  for v in nodes:
    for u in nbrs[v]:
      if core[v] < core[u]:
        nbrs[u].delete(nbrs[u].find(v))
        var pos = nodePos[u]
        var binStart = binBoundaries[core[u]]
        nodePos[u] = binStart
        nodePos[nodes[binStart]] = pos
        swap(nodes[binStart], nodes[pos])
        binBoundaries[core[u]] += 1
        core[u] -= 1
  return core

proc coreSubgraph(
  g: Graph,
  kFilter: proc(node: Node, cutoff: int, core: Table[Node, int]): bool,
  k : int = -1,
  core: Table[Node, int] = initTable[Node, int]()
): Graph =
  var coreUsing = core
  if len(core.keys().toSeq()) == 0:
    coreUsing = g.coreNumber()
  var kUsing = k
  if k == -1:
    kUsing = max(coreUsing.values().toSeq())
  var nodes: HashSet[Node] = initHashSet[Node]()
  for v in coreUsing.keys():
    if kFilter(v, kUsing, coreUsing):
      nodes.incl(v)
  return g.subgraph(nodes)
proc coreSubgraph(
  dg: DiGraph,
  kFilter: proc(node: Node, cutoff: int, core: Table[Node, int]): bool,
  k : int = -1,
  core: Table[Node, int] = initTable[Node, int]()
): DiGraph =
  var coreUsing = core
  if len(core.keys().toSeq()) == 0:
    coreUsing = dg.coreNumber()
  var kUsing = k
  if k == -1:
    kUsing = max(coreUsing.values().toSeq())
  var nodes: HashSet[Node] = initHashSet[Node]()
  for v in coreUsing.keys():
    if kFilter(v, kUsing, coreUsing):
      nodes.incl(v)
  return dg.subgraph(nodes)

proc kCore*(g: Graph, k: int = -1, coreNumber: Table[Node, int] = initTable[Node, int]()): Graph =
  let kFilter: proc(node: Node, cutoff: int, core: Table[Node, int]): bool =
    proc(node: Node, cutoff: int, core: Table[Node, int]): bool =
      return cutoff <= core[node]
  return coreSubgraph(g, kFilter, k, coreNumber)
proc kCore*(dg: DiGraph, k: int = -1, coreNumber: Table[Node, int] = initTable[Node, int]()): DiGraph =
  let kFilter: proc(node: Node, cutoff: int, core: Table[Node, int]): bool =
    proc(node: Node, cutoff: int, core: Table[Node, int]): bool =
      return cutoff <= core[node]
  return coreSubgraph(dg, kFilter, k, coreNumber)

proc kShell*(g: Graph, k: int = -1, coreNumber: Table[Node, int] = initTable[Node, int]()): Graph =
  let kFilter: proc(node: Node, cutoff: int, core: Table[Node, int]): bool =
    proc(node: Node, cutoff: int, core: Table[Node, int]): bool =
      return cutoff == core[node]
  return coreSubgraph(g, kFilter, k, coreNumber)
proc kShell*(dg: DiGraph, k: int = -1, coreNumber: Table[Node, int] = initTable[Node, int]()): DiGraph =
  let kFilter: proc(node: Node, cutoff: int, core: Table[Node, int]): bool =
    proc(node: Node, cutoff: int, core: Table[Node, int]): bool =
      return cutoff == core[node]
  return coreSubgraph(dg, kFilter, k, coreNumber)

proc kCrust*(g: Graph, k: int = -1, coreNumber: Table[Node, int] = initTable[Node, int]()): Graph =
  var coreNumberUsing = coreNumber
  if len(coreNumber.keys().toSeq()) == 0:
    coreNumberUsing = g.coreNumber()
  var kUsing = k
  if k == -1:
    kUsing = max(coreNumberUsing.values().toSeq()) - 1
  var nodes: HashSet[Node] = initHashSet[Node]()
  for (k, v) in coreNumberUsing.pairs():
    if v <= kUsing:
      nodes.incl(k)
  return g.subgraph(nodes)
proc kCrust*(dg: DiGraph, k: int = -1, coreNumber: Table[Node, int] = initTable[Node, int]()): DiGraph =
  var coreNumberUsing = coreNumber
  if len(coreNumber.keys().toSeq()) == 0:
    coreNumberUsing = dg.coreNumber()
  var kUsing = k
  if k == -1:
    kUsing = max(coreNumberUsing.values().toSeq()) - 1
  var nodes: HashSet[Node] = initHashSet[Node]()
  for (k, v) in coreNumberUsing.pairs():
    if v <= kUsing:
      nodes.incl(k)
  return dg.subgraph(nodes)

proc kCorona*(g: Graph, k: int = -1, coreNumber: Table[Node, int] = initTable[Node, int]()): Graph =
  let kFilter: proc(node: Node, cutoff: int, core: Table[Node, int]): bool =
    proc(node: Node, cutoff: int, core: Table[Node, int]): bool =
      var s = 0
      for w in g.neighbors(node):
        if core[w] >= cutoff:
          s += 1
      return cutoff == core[node] and cutoff == s
  return coreSubgraph(g, kFilter, k, coreNumber)
proc kCorona*(dg: DiGraph, k: int = -1, coreNumber: Table[Node, int] = initTable[Node, int]()): DiGraph =
  let kFilter: proc(node: Node, cutoff: int, core: Table[Node, int]): bool =
    proc(node: Node, cutoff: int, core: Table[Node, int]): bool =
      var s = 0
      for w in dg.neighbors(node):
        if core[w] >= cutoff:
          s += 1
      return cutoff == core[node] and cutoff == s
  return coreSubgraph(dg, kFilter, k, coreNumber)

proc kTruss*(g: Graph, k: int): Graph =
  var h = g.copyAsGraph()

  var nDropped = 1
  while 0 < nDropped:
    nDropped = 0
    var toDrop: seq[Edge] = @[]
    var seen = initHashSet[Node]()
    for u in h.nodes():
      var nbrsU = h.neighborsSet(u)
      seen.incl(u)
      var newNbrs: seq[Node] = @[]
      for v in nbrsU:
        if v notin seen:
          newNbrs.add(v)
      for v in newNbrs:
        if len(nbrsU * h.neighborsSet(v)) < (k - 2):
          toDrop.add((u, v))
    h.removeEdgesFrom(toDrop)
    nDropped = len(toDrop)
    var isolatedNodes: seq[Node] = @[]
    for v in h.nodes():
      if h.degree(v) == 0:
        isolatedNodes.add(v)
    h.removeNodesFrom(isolatedNodes)
  return h

proc onionLayers*(g: Graph): Table[Node, int] =
  var h = g.copyAsGraph()
  if 0 < g.numberOfSelfloop():
    raise newNNError("input graph contains self loops which is not permitted")
  var odLayers: Table[Node, int] = initTable[Node, int]()
  var neighbors: Table[Node, seq[Node]] = initTable[Node, seq[Node]]()
  for v in g.nodes():
    neighbors[v] = g.neighbors(v)
  var degrees = g.degree()
  var currentCore = 1
  var currentLayer = 1
  var isolatedNodes: seq[Node] = @[]
  for v in h.nodes():
    if h.degree(v) == 0:
      isolatedNodes.add(v)
  if 0 < len(isolatedNodes):
    for v in isolatedNodes:
      odLayers[v] = currentLayer
      degrees.del(v)
    currentLayer = 2
  while 0 < len(degrees):
    var nodes = degrees.keys().toSeq()
    var cmpDegree = proc (x, y: Node): int =
      result = system.cmp(degrees[x], degrees[y])
    nodes.sort(cmpDegree)
    var minDegree = degrees[nodes[0]]
    if currentCore < minDegree:
      currentCore = minDegree
    var thisLayer: seq[Node] = @[]
    for n in nodes:
      if currentCore < degrees[n]:
        break
      thisLayer.add(n)
    for v in thisLayer:
      odLayers[v] = currentLayer
      for n in neighbors[v]:
        neighbors[n].delete(neighbors[n].find(v))
        degrees[n] -= 1
      degrees.del(v)
    currentLayer += 1
  return odLayers

# -------------------------------------------------------------------
# TODO:
# Covering
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Cycle
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Cuts
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Distance Measures
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Distance-Regular Graphs
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Dominance
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Dominating Sets
# -------------------------------------------------------------------

proc dominatingSet*(G: Graph, startWith: Node = None): HashSet[Node] =
  let allNodes = G.nodesSet()
  var startWithUsing = startWith
  if startWith == None:
    startWithUsing = G.nodes()[0]
  if startWithUsing notin allNodes:
    raise newNNError(fmt"node {startWithUsing} not in graph")
  var dominatingSet = initHashSet[Node]()
  dominatingSet.incl(startWithUsing)
  var dominatedNodes = G.neighborsSet(startWithUsing)
  var remainingNodes = allNodes - dominatedNodes - dominatingSet
  while len(remainingNodes) != 0:
    var tmp = remainingNodes.toSeq()
    tmp.sort()
    var v = tmp[0]
    remainingNodes.excl(v)
    var undominatedNeighbors = G.neighborsSet(v) - dominatingSet
    dominatingSet.incl(v)
    dominatedNodes = dominatedNodes + undominatedNeighbors
    remainingNodes = remainingNodes - undominatedNeighbors
  return dominatingSet
proc dominatingSet*(DG: DiGraph, startWith: Node = None): HashSet[Node] =
  let allNodes = DG.nodesSet()
  var startWithUsing = startWith
  if startWith == None:
    startWithUsing = DG.nodes()[0]
  if startWithUsing notin allNodes:
    raise newNNError(fmt"node {startWithUsing} not in graph")
  var dominatingSet = initHashSet[Node]()
  dominatingSet.incl(startWithUsing)
  var dominatedNodes = DG.successorsSet(startWithUsing)
  var remainingNodes = allNodes - dominatedNodes - dominatingSet
  while len(remainingNodes) != 0:
    var tmp = remainingNodes.toSeq()
    tmp.sort()
    var v = tmp[0]
    remainingNodes.excl(v)
    var undominatedNeighbors = DG.successorsSet(v) - dominatingSet
    dominatingSet.incl(v)
    dominatedNodes = dominatedNodes + undominatedNeighbors
    remainingNodes = remainingNodes - undominatedNeighbors
  return dominatingSet

proc isDominatingSet*(G: Graph, nbunch: seq[Node]): bool =
  var testset = initHashSet[Node]()
  for node in nbunch:
    if node in G.nodesSet():
      testset.incl(node)
  var nbrs = initHashSet[Node]()
  for node in testset:
    for nbr in G.neighbors(node):
      nbrs.incl(nbr)
  return len(G.nodesSet() - testset - nbrs) == 0
proc isDominatingSet*(G: Graph, nbunch: HashSet[Node]): bool =
  var testset = initHashSet[Node]()
  for node in nbunch:
    if node in G.nodesSet():
      testset.incl(node)
  var nbrs = initHashSet[Node]()
  for node in testset:
    for nbr in G.neighbors(node):
      nbrs.incl(nbr)
  return len(G.nodesSet() - testset - nbrs) == 0
proc isDominatingSet*(DG: DiGraph, nbunch: seq[Node]): bool =
  var testset = initHashSet[Node]()
  for node in nbunch:
    if node in DG.nodesSet():
      testset.incl(node)
  var nbrs = initHashSet[Node]()
  for node in testset:
    for nbr in DG.successors(node):
      nbrs.incl(nbr)
  return len(DG.nodesSet() - testset - nbrs) == 0
proc isDominatingSet*(DG: DiGraph, nbunch: HashSet[Node]): bool =
  var testset = initHashSet[Node]()
  for node in nbunch:
    if node in DG.nodesSet():
      testset.incl(node)
  var nbrs = initHashSet[Node]()
  for node in testset:
    for nbr in DG.successors(node):
      nbrs.incl(nbr)
  return len(DG.nodesSet() - testset - nbrs) == 0

# -------------------------------------------------------------------
# TODO:
# Efficiency
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Flows
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Graph Hashing
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Graphical Degree Sequence
# -------------------------------------------------------------------

proc basicGraphicalTests(sequence: seq[int]): tuple[dmax: int, dmin: int, dsum: int, n: int, numDegs: seq[int]] =
  var degSequence = sequence
  var p = len(degSequence)
  var numDegs: seq[int] = newSeq[int]()
  for i in 0..<p:
      numDegs.add(0)
  var dmax = 0
  var dmin = p
  var dsum = 0
  var n = 0
  for d in degSequence:
    if d < 0 or d >= p:
      raise newNNUnfeasible("d < 0 or d >= p")
    elif d > 0:
      dmax = max(dmax, d)
      dmin = min(dmin, d)
      dsum += d
      n += 1
      numDegs[d] += 1
  if dsum mod 2 == 1 or dsum > n * (n - 1):
    raise newNNUnfeasible("dsum mod 2 == 1 or dsum > n * (n - 1)")
  return (dmax, dmin, dsum, n, numDegs)

proc isValidDegreeSequenceHavelHakimi*(sequence: seq[int]): bool =
  var dmax: int
  var dmin: int
  var dsum: int
  var n: int
  var numDegs: seq[int]
  try:
    (dmax, dmin, dsum, n, numDegs) = basicGraphicalTests(sequence)
  except NNUnfeasible:
    return false

  if n == 0 or dmin * n >= (dmax + dmin + 1) * (dmax + dmin + 1):
    return true

  var modstubs: seq[int] = @[]
  for i in 0..<(dmax + 1):
    modstubs.add(0)

  while 0 < n:
    while numDegs[dmax] == 0:
      dmax -= 1
    if n - 1 < dmax:
      return false
    numDegs[dmax] = numDegs[dmax] - 1
    n -= 1
    var mslen = 0
    var k = dmax
    for i in 0..<dmax:
      while numDegs[k] == 0:
        k -= 1
      numDegs[k] -= 1
      n -= 1
      if 1 < k:
        modstubs[mslen] = k - 1
        mslen += 1
    for i in 0..<mslen:
      var stub = modstubs[i]
      numDegs[stub] += 1
      n += 1
  return true

proc isValidDegreeSequenceErodsGallai*(sequence: seq[int]): bool =
  var dmax: int
  var dmin: int
  var dsum: int
  var n: int
  var numDegs: seq[int]
  try:
    (dmax, dmin, dsum, n, numDegs) = basicGraphicalTests(sequence)
  except NNUnfeasible:
    return false
  if n == 0 or 4 * dmin * n >= (dmax + dmin + 1) * (dmax + dmin + 1):
    return true

  var k: int = 0
  var sumDeg: int = 0
  var sumNj: int = 0
  var sumJnj: int = 0

  for dk in countdown(dmax, dmin):
    if dk < k + 1:
      return true
    if 0 < numDegs[dk]:
      var runSize = numDegs[dk]
      if dk < k + runSize:
        runSize = dk - k
      sumDeg += runSize * dk
      for v in 0..<runSize:
        sumNj += numDegs[k + v]
        sumJnj += (k + v) * numDegs[k + v]
      k += runSize
      if sumDeg > k * (n - 1) - k * sumNj + sumJnj:
        return false
  return true

proc isMultiGraphical*(sequence: seq[int]): bool =
  var degSequence: seq[int] = sequence
  var dsum = 0
  var dmax = 0
  for d in degSequence:
    if d < 0:
      return false
    dsum += d
    dmax = max(dmax, d)
  if dsum mod 2 == 1 or dsum < 2 * dmax:
    return false
  return true

proc isPseudoGraphical*(sequence: seq[int]): bool =
  var s = 0
  for d in sequence:
    s += d
  return s mod 2 == 0 and min(sequence) >= 0

proc isDiGraphical*(inSequence: seq[int], outSequence: seq[int]): bool =
  var sumin = 0
  var sumout = 0
  var nin = len(inSequence)
  var nout = len(outSequence)
  var maxn = max(nin, nout)
  var maxin = 0
  if maxn == 0:
    return true
  var stubheap: HeapQueue[tuple[outDeg: int, inDeg: int]] = initHeapQueue[tuple[outDeg: int, inDeg: int]]()
  var zeroheap: HeapQueue[int] = initHeapQueue[int]()
  for n in 0..<maxn:
    var inDeg = 0
    var outDeg = 0
    if n < nout:
      outDeg = outSequence[n]
    if n < nin:
      inDeg = inSequence[n]
    if inDeg < 0 or outDeg < 0:
      return false
    sumin += inDeg
    sumout += outDeg
    maxin = max(maxin, inDeg)
    if inDeg > 0:
      stubheap.push((-1 * outDeg, -1 * inDeg))
    elif outDeg > 0:
      zeroheap.push(-1 * outDeg)
  if sumin != sumout:
    return false
  var modstubs: seq[tuple[outDeg: int, inDeg: int]] = @[]
  for i in 0..maxin:
    modstubs.add((0, 0))
  while len(stubheap) != 0:
    var (freeout, freein) = stubheap.pop()
    freein *= -1
    if freein > len(stubheap) + len(zeroheap):
      return false
    var mslen = 0
    for i in 0..<freein:
      var stubout: int
      var stubin: int
      if len(zeroheap) != 0 and (len(stubheap) == 0 or stubheap[0][0] > zeroheap[0]):
        stubout = zeroheap.pop()
        stubin = 0
      else:
        (stubout, stubin) = stubheap.pop()
      if stubout == 0:
        return false
      if stubout + 1 < 0 or stubin < 0:
        modstubs[mslen] = (stubout + 1, stubin)
        mslen += 1
    for i in 0..<mslen:
      var stub: tuple[outDeg: int, inDeg: int] = modstubs[i]
      if stub[1] < 0:
        stubheap.push(stub)
      else:
        zeroheap.push(stub[0])
    if freeout < 0:
      zeroheap.push(freeout)
  return true

proc isGraphical*(sequence: seq[int], methodName: string = "eg"): bool =
  if methodName == "eg":
    return isValidDegreeSequenceErodsGallai(sequence)
  elif methodName == "hh":
    return isValidDegreeSequenceHavelHakimi(sequence)
  raise newNNException("method must be 'eg' or 'hh'")

# -------------------------------------------------------------------
# TODO:
# Hierarchy
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Hybrid
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Isolates
# -------------------------------------------------------------------

proc isIsolate*(G: Graph, n: Node): bool =
  return G.degree(n) == 0
proc isIsolate*(DG: DiGraph, n: Node): bool =
  return DG.degree(n) == 0

proc isolates*(G: Graph): seq[Node] =
  var ret: seq[Node] = @[]
  for n in G.nodes():
    if G.isIsolate(n):
      ret.add(n)
  return ret
proc isolatesSet*(G: Graph): HashSet[Node] =
  return G.isolates().toHashSet()
proc isolatesSeq*(G: Graph): seq[Node] =
  return G.isolates()
proc isolates*(DG: DiGraph): seq[Node] =
  var ret: seq[Node] = @[]
  for n in DG.nodes():
    if DG.isIsolate(n):
      ret.add(n)
  return ret
proc isolatesSet*(DG: DiGraph): HashSet[Node] =
  return DG.isolates().toHashSet()
proc isolatesSeq*(DG: DiGraph): seq[Node] =
  return DG.isolates()

proc numberOfIsolates*(G: Graph): int =
  return len(G.isolates())
proc numberOfIsolates*(DG: DiGraph): int =
  return len(DG.isolates())

# -------------------------------------------------------------------
# TODO:
# Isomorphism
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Link Analysis
# -------------------------------------------------------------------

proc pagerankNim(
  DG: DiGraph,
  alpha: float = 0.85,
  personalization: TableRef[Node, float] = nil,
  maxIter: int = 100,
  tol: float = 1.0e-6,
  nstart: TableRef[Node, float] = nil,
  weight: TableRef[Edge, float] = nil,
  dangling: TableRef[Node, float] = nil
): Table[Node, float] =
  if len(DG) == 0:
    return initTable[Node, float]()
  let N = DG.numberOfNodes()

  var x: Table[Node, float]
  if nstart == nil:
    for node in DG.nodes():
      x[node] = 1.0 / N.float
  else:
    var s = 0.0
    for val in nstart.values():
      s += val
    for (k, v) in nstart.pairs():
      x[k] = v / s

  var p: Table[Node, float]
  if personalization == nil:
    for node in DG.nodes():
      p[node] = 1.0 / N.float
  else:
    var s = 0.0
    for val in personalization.values():
      s += val
    for (k, v) in personalization.pairs():
      p[k] = v / s

  var danglingWeights: Table[Node, float]
  if dangling == nil:
    danglingWeights = p
  else:
    var s = 0.0
    for val in dangling.values():
      s += val
    for (k, v) in dangling.pairs():
      danglingWeights[k] = v / s

  var danglingNodes: seq[Node] = @[]
  for node in DG.nodes():
    if DG.outDegree(node) == 0:
      danglingNodes.add(node)

  for _ in 0..<maxIter:
    var xlast = x
    x = initTable[Node, float]()
    for node in x.keys():
      x[node] = 0.0
    var dangleSum = 0.0
    for node in danglingNodes:
      dangleSum += xlast[node]
    dangleSum *= alpha
    for n in x.keys():
      for nbr in DG.successors(n):
        let wt = weight.getOrDefault((n, nbr), 1.0 / DG.outDegree(n).float)
        x[nbr] += alpha * xlast[n] * wt
      x[n] += dangleSum * danglingWeights.getOrDefault(n, 0.0) + (1.0 - alpha) * p.getOrDefault(n, 0.0)
    var err = 0.0
    for n in x.keys():
      err += abs(x[n] - xlast[n])
    if err < N.float * tol:
      return x
  raise newNNPowerIterationFailedConvergence(maxIter)

proc pagerank*(
  G: Graph,
  alpha: float = 0.85,
  personalization: TableRef[Node, float] = nil,
  maxIter: int = 100,
  tol: float = 1.0e-6,
  nstart: TableRef[Node, float] = nil,
  weight: TableRef[Edge, float] = nil,
  dangling: TableRef[Node, float] = nil
): Table[Node, float] =
  return pagerankNim(G.toDirected(), alpha, personalization, maxIter, tol, nstart, weight, dangling)
proc pagerank*(
  DG: DiGraph,
  alpha: float = 0.85,
  personalization: TableRef[Node, float] = nil,
  maxIter: int = 100,
  tol: float = 1.0e-6,
  nstart: TableRef[Node, float] = nil,
  weight: TableRef[Edge, float] = nil,
  dangling: TableRef[Node, float] = nil
): Table[Node, float] =
  return pagerankNim(DG, alpha, personalization, maxIter, tol, nstart, weight, dangling)

proc hitsNim(
  G: Graph,
  maxIter: int = 100,
  tol: float = 1.0e-8,
  nstart: TableRef[Node, float] = nil,
  weight: TableRef[Edge, float] = nil,
  normalized: bool = true
): tuple[hubs: Table[Node, float], authorities: Table[Node, float]] =
  if len(G) == 0:
    return (initTable[Node, float](), initTable[Node, float]())

  let N = G.numberOfNodes()

  var h: Table[Node, float]
  if nstart == nil:
    for node in G.nodes():
      h[node] = 1.0 / N.float
  else:
    var s = sum(nstart.values().toSeq())
    for (k, v) in nstart.pairs():
      h[k] = v / s

  var a = initTable[Node, float]()
  for node in h.keys():
    a[node] = 0.0

  var converged = false
  for _ in 0..<maxIter:
    var hlast = h
    for node in h.keys():
      h[node] = 0.0
    for n in h.keys():
      for nbr in G.neighbors(n):
        if weight != nil:
          a[nbr] += hlast[n] * weight.getOrDefault((n, nbr), 1.0)
        else:
          a[nbr] += hlast[n]
    for n in a.keys():
      for nbr in G.neighbors(n):
        if weight != nil:
          h[n] += a[nbr] * weight.getOrDefault((n, nbr), 1.0)
        else:
          h[n] += a[nbr]

    var s = 1.0 / max(h.values().toSeq())
    for n in h.keys():
      h[n] *= s

    s = 1.0 / max(a.values().toSeq())
    for n in a.keys():
      a[n] *= s

    var err = 0.0
    for n in h.keys():
      err += abs(h[n] - hlast[n])
    if err < tol:
      converged = true
      break

  if not converged:
    raise newNNPowerIterationFailedConvergence(maxIter)

  if normalized:
    var s = 1.0 / sum(a.values().toSeq())
    for n in a.keys():
      a[n] *= s
    s = 1.0 / sum(h.values().toSeq())
    for n in h.keys():
      h[n] *= s
  return (h, a)

proc hitsNim(
  DG: DiGraph,
  maxIter: int = 100,
  tol: float = 1.0e-8,
  nstart: TableRef[Node, float] = nil,
  weight: TableRef[Edge, float] = nil,
  normalized: bool = true
): tuple[hubs: Table[Node, float], authorities: Table[Node, float]] =
  if len(DG) == 0:
    return (initTable[Node, float](), initTable[Node, float]())

  let N = DG.numberOfNodes()

  var h: Table[Node, float]
  if nstart == nil:
    for node in DG.nodes():
      h[node] = 1.0 / N.float
  else:
    var s = sum(nstart.values().toSeq())
    for (k, v) in nstart.pairs():
      h[k] = v / s

  var a = initTable[Node, float]()
  for node in h.keys():
    a[node] = 0.0

  var converged = false
  for _ in 0..<maxIter:
    var hlast = h
    for node in h.keys():
      h[node] = 0.0
    for n in h.keys():
      for succ in DG.successors(n):
        if weight != nil:
          a[succ] += hlast[n] * weight.getOrDefault((n, succ), 1.0)
        else:
          a[succ] += hlast[n]
    for n in a.keys():
      for succ in DG.successors(n):
        if weight != nil:
          h[n] += a[succ] * weight.getOrDefault((n, succ), 1.0)
        else:
          h[n] += a[succ]

    var s = 1.0 / max(h.values().toSeq())
    for n in h.keys():
      h[n] *= s

    s = 1.0 / max(a.values().toSeq())
    for n in a.keys():
      a[n] *= s

    var err = 0.0
    for n in h.keys():
      err += abs(h[n] - hlast[n])
    if err < tol:
      converged = true
      break

  if not converged:
    raise newNNPowerIterationFailedConvergence(maxIter)

  if normalized:
    var s = 1.0 / sum(a.values().toSeq())
    for n in a.keys():
      a[n] *= s
    s = 1.0 / sum(h.values().toSeq())
    for n in h.keys():
      h[n] *= s
  return (h, a)

proc hits*(
  G: Graph,
  maxIter: int = 100,
  tol: float = 1.0e-8,
  nstart: TableRef[Node, float] = nil,
  weight: TableRef[Edge, float] = nil,
  normalized: bool = true
): tuple[hubs: Table[Node, float], authorities: Table[Node, float]] =
  return hitsNim(G, maxIter, tol, nstart, weight, normalized)
proc hits*(
  DG: DiGraph,
  maxIter: int = 100,
  tol: float = 1.0e-8,
  nstart: TableRef[Node, float] = nil,
  weight: TableRef[Edge, float] = nil,
  normalized: bool = true
): tuple[hubs: Table[Node, float], authorities: Table[Node, float]] =
  return hitsNim(DG, maxIter, tol, nstart, weight, normalized)

# -------------------------------------------------------------------
# TODO:
# Lowest Common Ancestor
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Matching
# -------------------------------------------------------------------

proc maximalMatching*(G: Graph): HashSet[tuple[u: Node, v: Node]] =
  var matching = initHashSet[tuple[u: Node, v: Node]]()
  var nodes = initHashSet[Node]()
  for (u, v) in G.edges():
    if u notin nodes and v notin nodes and u != v:
      matching.incl((u, v))
      nodes.incl(u)
      nodes.incl(v)
  return matching

proc matchingTableToSet(matching: Table[Node, Node]): HashSet[tuple[u: Node, v: Node]] =
  var ret = initHashSet[tuple[u: Node, v: Node]]()
  for (k, v) in matching.pairs():
    ret.incl((k, v))
  return ret

# -------------------------------------------------------------------
# TODO:
# Minors
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Maximal Independent Set
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Non-randomness
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Moral
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Node Classification
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Operators
# -------------------------------------------------------------------

proc complement*(G: Graph): Graph =
  let R = newGraph()
  R.addNodesFrom(G.nodes())
  var edges: seq[Edge] = @[]
  for (n, nbrs) in G.adj.pairs():
    for n2 in G.nodes():
      if n != n2:
        if n2 notin nbrs:
          edges.add((n, n2))
  R.addEdgesFrom(edges)
  return R
proc complement*(DG: DiGraph): DiGraph =
  let R = newDiGraph()
  R.addNodesFrom(DG.nodes())
  var edges: seq[Edge] = @[]
  for (n, nbrs) in DG.succ.pairs():
    for n2 in DG.nodes():
      if n != n2:
        if n2 notin nbrs:
          edges.add((n, n2))
  R.addEdgesFrom(edges)
  return R

proc reverse*(DG: DiGraph): DiGraph =
  return DG.reversed()
proc reverseInplace*(DG: var DiGraph) =
  DG = DG.reverse()

proc compose*(G: Graph, H: Graph): Graph =
  let R = newGraph()
  R.addNodesFrom(G.nodes())
  R.addNodesFrom(H.nodes())
  R.addEdgesFrom(G.edges())
  R.addEdgesFrom(H.edges())
  return R
proc compose*(DG: DiGraph, DH: DiGraph): DiGraph =
  let DR = newDiGraph()
  DR.addNodesFrom(DG.nodes())
  DR.addNodesFrom(DH.nodes())
  DR.addEdgesFrom(DG.edges())
  DR.addEdgesFrom(DH.edges())
  return DR

proc union*(G: Graph, H: Graph): Graph =
  if len(G.nodesSet() * H.nodesSet()) != 0:
    raise newNNError("nodes sets of G and H are not disjoint")
  let R = newGraph()
  R.addNodesFrom(G.nodes())
  R.addNodesFrom(H.nodes())
  R.addEdgesFrom(G.edges())
  R.addEdgesFrom(H.edges())
  return R
proc union*(DG: DiGraph, DH: DiGraph): DiGraph =
  if len(DG.nodesSet() * DH.nodesSet()) != 0:
    raise newNNError("nodes sets of DG and DH are not disjoint")
  let DR = newDiGraph()
  DR.addNodesFrom(DG.nodes())
  DR.addNodesFrom(DH.nodes())
  DR.addEdgesFrom(DG.edges())
  DR.addEdgesFrom(DH.edges())
  return DR

proc disjointUnion*(G: Graph, H: Graph): Graph =
  var originalToNew = initTable[tuple[which: string, node: Node], int]()
  var newLabel = 0
  for node in G.nodes():
    originalToNew[("G", node)] = newLabel
    newLabel += 1
  for node in H.nodes():
    originalToNew[("H", node)] = newLabel
    newLabel += 1
  let R = newGraph()
  for node in G.nodes():
    R.addNode(originalToNew[("G", node)])
  for edge in G.edges():
    R.addEdge((originalToNew[("G", edge.u)], originalToNew[("G", edge.v)]))
  for node in H.nodes():
    R.addNode(originalToNew[("H", node)])
  for edge in H.edges():
    R.addEdge((originalToNew[("H", edge.u)], originalToNew[("H", edge.v)]))
  return R
proc disjointUnion*(DG: DiGraph, DH: DiGraph): DiGraph =
  var originalToNew = initTable[tuple[which: string, node: Node], int]()
  var newLabel = 0
  for node in DG.nodes():
    originalToNew[("DG", node)] = newLabel
    newLabel += 1
  for node in DH.nodes():
    originalToNew[("DH", node)] = newLabel
    newLabel += 1
  let DR = newDiGraph()
  for node in DG.nodes():
    DR.addNode(originalToNew[("DG", node)])
  for edge in DG.edges():
    DR.addEdge((originalToNew[("DG", edge.u)], originalToNew[("DG", edge.v)]))
  for node in DH.nodes():
    DR.addNode(originalToNew[("DH", node)])
  for edge in DH.edges():
    DR.addEdge((originalToNew[("DH", edge.u)], originalToNew[("DH", edge.v)]))
  return DR

proc intersection*(G: Graph, H: Graph): Graph =
  let R = newGraph()
  R.addNodesFrom(G.nodesSet() * H.nodesSet())
  if len(G.edges()) <= len(H.edges()):
    for edge in G.edges():
      if H.hasEdge(edge):
        R.addEdge(edge)
  else:
    for edge in H.edges():
      if G.hasEdge(edge):
        R.addEdge(edge)
  return R
proc intersection*(DG: DiGraph, DH: DiGraph): DiGraph =
  let DR = newDiGraph()
  DR.addNodesFrom(DG.nodesSet() * DH.nodesSet())
  if len(DG.edges()) <= len(DH.edges()):
    for edge in DG.edges():
      if DH.hasEdge(edge):
        DR.addEdge(edge)
  else:
    for edge in DH.edges():
      if DG.hasEdge(edge):
        DR.addEdge(edge)
  return DR

proc difference*(G: Graph, H: Graph): Graph =
  if G.nodesSet() != H.nodesSet():
    raise newNNError("node sets of graphs not equal")
  let R = newGraph()
  for edge in G.edges():
    if not H.hasEdge(edge):
      R.addEdge(edge)
  return R
proc difference*(DG: DiGraph, DH: DiGraph): DiGraph =
  if DG.nodesSet() != DH.nodesSet():
    raise newNNError("node sets of directed graphs not equal")
  let DR = newDiGraph()
  for edge in DG.edges():
    if not DH.hasEdge(edge):
      DR.addEdge(edge)
  return DR

proc symmetricDifference*(G: Graph, H: Graph): Graph =
  if G.nodesSet() != H.nodesSet():
    raise newNNError("node sets of graphs not equal")
  let R = newGraph()
  for edge in G.edges():
    if not H.hasEdge(edge):
      R.addEdge(edge)
  for edge in H.edges():
    if not G.hasEdge(edge):
      R.addEdge(edge)
  return R
proc symmetricDifference*(DG: DiGraph, DH: DiGraph): DiGraph =
  if DG.nodesSet() != DH.nodesSet():
    raise newNNError("node sets of directed graphs not equal")
  let DR = newDiGraph()
  for edge in DG.edges():
    if not DH.hasEdge(edge):
      DR.addEdge(edge)
  for edge in DH.edges():
    if not DG.hasEdge(edge):
      DR.addEdge(edge)
  return DR

proc fullJoin*(G: Graph, H: Graph): Graph =
  let R = union(G, H)
  for i in G.nodes():
    for j in H.nodes():
      R.addEdge(i, j)
  return R
proc fullJoin*(DG: DiGraph, DH: DiGraph): DiGraph =
  let DR = union(DG, DH)
  for i in DG.nodes():
    for j in DH.nodes():
      DR.addEdge(i, j)
      DR.addEdge(j, i)
  return DR

proc composeAll*(Gs: seq[Graph]): Graph =
  if len(Gs) == 0:
    raise newNNError("cannot apply composeAll to empty graph sequence")
  var R = Gs[0]
  for i in 1..<len(Gs):
    R = compose(R, Gs[i])
  return R
proc composeAll*(DGs: seq[DiGraph]): DiGraph =
  if len(DGs) == 0:
    raise newNNError("cannot apply composeAll to empty directed graph sequence")
  var DR = DGs[0]
  for i in 1..<len(DGs):
    DR = compose(DR, DGs[i])
  return DR

proc unionAll*(Gs: seq[Graph]): Graph =
  if len(Gs) == 0:
    raise newNNError("cannot apply unionAll to empty graph sequence")
  var R = Gs[0]
  for i in 1..<len(Gs):
    R = union(R, Gs[i])
  return R
proc unionAll*(DGs: seq[DiGraph]): DiGraph =
  if len(DGs) == 0:
    raise newNNError("cannot apply unionAll to empty directed graph sequence")
  var DR = DGs[0]
  for i in 1..<len(DGs):
    DR = union(DR, DGs[i])
  return DR

proc disjointUnionAll*(Gs: seq[Graph]): Graph =
  if len(Gs) == 0:
    raise newNNError("cannot apply disjointUnionAll to empty graph sequence")
  var R = Gs[0]
  for i in 1..<len(Gs):
    R = disjointUnion(R, Gs[i])
  return R
proc disjointUnionAll*(DGs: seq[DiGraph]): DiGraph =
  if len(DGs) == 0:
    raise newNNError("cannot apply disjointUnionAll to empty directed graph sequence")
  var DR = DGs[0]
  for i in 1..<len(DGs):
    DR = disjointUnion(DR, DGs[i])
  return DR

proc intersectionAll*(Gs: seq[Graph]): Graph =
  if len(Gs) == 0:
    raise newNNError("cannot apply intersectionAll to empty graph sequence")
  var R = Gs[0]
  for i in 1..<len(Gs):
    R = intersection(R, Gs[i])
  return R
proc intersectionAll*(DGs: seq[DiGraph]): DiGraph =
  if len(DGs) == 0:
    raise newNNError("cannot apply intersectionAll to empty directed graph sequence")
  var DR = DGs[0]
  for i in 1..<len(DGs):
    DR = intersection(DR, DGs[i])
  return DR

proc power*(G: Graph, k: int): Graph =
  if k <= 0:
    raise newNNError("k must be greater than zero")
  let H = newGraph(G.nodes())
  for n in G.nodes():
    var seen = initTable[Node, int]()
    var level = 1
    var nextLevel = G.neighborsSet(n)
    while len(nextLevel) != 0:
      let thisLevel = nextLevel
      nextLevel = initHashSet[Node]()
      for v in thisLevel:
        if v == n:
          continue # ignore selfloop
        if v notin seen:
          seen[v] = level
          nextLevel = nextLevel + G.neighborsSet(v)
      if k <= level:
        break
      level += 1
    for nbr in seen.keys():
      H.addEdge((n, nbr))
  return H

# -------------------------------------------------------------------
# TODO:
# Planarity
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Planar Drawing
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Reciprocity
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Regular
# -------------------------------------------------------------------

proc isRegular*(G: Graph): bool =
  let n1 = G.nodes()[0]
  let d1 = G.degree(n1)
  var tmp: seq[bool] = @[]
  for d in G.degree().values():
    tmp.add(d1 == d)
  return all(tmp, proc(b: bool): bool = return b)
proc isRegular*(DG: DiGraph): bool =
  let n1 = DG.nodes()[0]
  let dIn = DG.inDegree(n1)
  var tmp: seq[bool] = @[]
  for d in DG.inDegree().values():
    tmp.add(dIn == d)
  let inRegular = all(tmp, proc(b: bool): bool = return b)

  let dOut = DG.outDegree(n1)
  tmp = @[]
  for d in DG.outDegree().values():
    tmp.add(dOut == d)
  let outRegular = all(tmp, proc(b: bool): bool = return b)
  return inRegular and outRegular

proc isKRegular*(G: Graph, k: int): bool =
  var tmp: seq[bool] = @[]
  for d in G.degree().values():
    tmp.add(d == k)
  return all(tmp, proc(b: bool): bool = return b)

# -------------------------------------------------------------------
# TODO:
# Rich Club
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Shortest Paths
# -------------------------------------------------------------------

iterator singleShortestPathLength(adj: Table[Node, HashSet[Node]], firstLevel: Table[Node, int], cutoff: float): tuple[node: Node, level: int] =
  var seen = initTable[Node, int]()
  var level = 0
  var nextLevel = firstLevel.keys().toSeq().toHashSet()
  let N = len(adj)
  while len(nextLevel) != 0 and cutoff >= level.float:
    var thisLevel = nextLevel
    nextLevel = initHashSet[Node]()
    var found: seq[Node] = @[]
    for v in thisLevel:
      if v notin seen:
        seen[v] = level
        found.add(v)
        yield (v, level)
    if len(seen) == N:
      break
    for v in found:
      for adjNode in adj[v]:
        nextLevel.incl(adjNode)
    level += 1
  seen.clear()

proc singleSourceShortestPathLength*(G: Graph, source: Node, cutoff: int = -1): Table[Node, int] =
  if source notin G.nodesSet():
    raise newNNNodeNotFound(source)
  var cutoffUsing: float = cutoff.float
  if cutoff == -1:
    cutoffUsing = Inf
  let nextLevel: Table[Node, int] = {source: 1}.toTable()
  var ret = initTable[Node, int]()
  for (target, length) in singleShortestPathLength(G.adj, nextLevel, cutoffUsing):
    ret[target] = length
  return ret
proc singleSourceShortestPathLength*(DG: DiGraph, source: Node, cutoff: int = -1): Table[Node, int] =
  if source notin DG.nodesSet():
    raise newNNNodeNotFound(source)
  var cutoffUsing: float = cutoff.float
  if cutoff == -1:
    cutoffUsing = Inf
  let nextLevel: Table[Node, int] = {source: 1}.toTable()
  var ret = initTable[Node, int]()
  for (target, length) in singleShortestPathLength(DG.succ, nextLevel, cutoffUsing):
    ret[target] = length
  return ret

proc singleTargetShortestPathLength*(G: Graph, target: Node, cutoff: int = -1): Table[Node, int] =
  if target notin G.nodesSet():
    raise newNNNodeNotFound(target)
  var cutoffUsing: float = cutoff.float
  if cutoff == -1:
    cutoffUsing = Inf
  let nextLevel = {target: 1}.toTable()
  var ret = initTable[Node, int]()
  for (source, shortestPathLength) in singleShortestPathLength(G.adj, nextLevel, cutoffUsing):
    ret[source] = shortestPathLength
  return ret
proc singleTargetShortestPathLength*(DG: DiGraph, target: Node, cutoff: int = -1): Table[Node, int] =
  if target notin DG.nodesSet():
    raise newNNNodeNotFound(target)
  var cutoffUsing: float = cutoff.float
  if cutoff == -1:
    cutoffUsing = Inf
  let nextLevel = {target: 1}.toTable()
  var ret = initTable[Node, int]()
  for (source, shortestPathLength) in singleShortestPathLength(DG.pred, nextLevel, cutoffUsing):
    ret[source] = shortestPathLength
  return ret

iterator allPairsShortestPathLength*(G: Graph, cutoff: int = -1): tuple[source: Node, shortestPathLengthTable: Table[Node, int]] =
  for n in G.nodes():
    yield (n, singleSourceShortestPathLength(G, n, cutoff))
iterator allPairsShortestPathLength*(DG: DiGraph, cutoff: int = -1): tuple[source: Node, shortestPathLengthTable: Table[Node, int]] =
  for n in DG.nodes():
    yield (n, singleSourceShortestPathLength(DG, n, cutoff))

proc bidirectionalPredSucc(G: Graph, source: Node, target: Node): tuple[pred: Table[Node, Node], succ: Table[Node, Node], w: Node] =
  if target == source:
    return ({target: None}.toTable(), {source: None}.toTable(), source)
  var pred = {source: None}.toTable()
  var succ = {target: None}.toTable()
  var forwardFringe = @[source]
  var reverseFringe = @[target]
  while len(forwardFringe) != 0 and len(reverseFringe) != 0:
    if len(forwardFringe) <= len(reverseFringe):
      var thisLevel = forwardFringe
      forwardFringe = @[]
      for v in thisLevel:
        for w in G.neighbors(v):
          if w notin pred:
            forwardFringe.add(w)
            pred[w] = v
          if w in succ:
            return (pred, succ, w)
    else:
      var thisLevel = reverseFringe
      reverseFringe = @[]
      for v in thisLevel:
        for w in G.neighbors(v):
          if w notin succ:
            succ[w] = v
            reverseFringe.add(w)
          if w in pred:
            return (pred, succ, w)
  raise newNNNoPath(fmt"no paths between {source} and {target}")
proc bidirectionalPredSucc(DG: DiGraph, source: Node, target: Node): tuple[pred: Table[Node, Node], succ: Table[Node, Node], w: Node] =
  if target == source:
    return ({target: None}.toTable(), {source: None}.toTable(), source)
  var pred = {source: None}.toTable()
  var succ = {target: None}.toTable()
  var forwardFringe = @[source]
  var reverseFringe = @[target]
  while len(forwardFringe) != 0 and len(reverseFringe) != 0:
    if len(forwardFringe) <= len(reverseFringe):
      var thisLevel = forwardFringe
      forwardFringe = @[]
      for v in thisLevel:
        for w in DG.successors(v):
          if w notin pred:
            forwardFringe.add(w)
            pred[w] = v
          if w in succ:
            return (pred, succ, w)
    else:
      var thisLevel = reverseFringe
      reverseFringe = @[]
      for v in thisLevel:
        for w in DG.predecessors(v):
          if w notin succ:
            succ[w] = v
            reverseFringe.add(w)
          if w in pred:
            return (pred, succ, w)
  raise newNNNoPath(fmt"no paths between {source} and {target}")

proc bidirectionalShortestPath*(G: Graph, source: Node, target: Node): seq[Node] =
  if source notin G.nodesSet():
    raise newNNNodeNotFound(source)
  if target notin G.nodesSet():
    raise newNNNodeNotFound(target)
  var (pred, succ, w) = bidirectionalPredSucc(G, source, target)
  var path: seq[Node] = @[]
  while w != None:
    path.add(w)
    w = pred[w]
  path.reverse()
  w = succ[path[^1]]
  while w != None:
    path.add(w)
    w = succ[w]
  return path
proc bidirectionalShortestPath*(DG: DiGraph, source: Node, target: Node): seq[Node] =
  if source notin DG.nodesSet():
    raise newNNNodeNotFound(source)
  if target notin DG.nodesSet():
    raise newNNNodeNotFound(target)
  var (pred, succ, w) = bidirectionalPredSucc(DG, source, target)
  var path: seq[Node] = @[]
  while w != None:
    path.add(w)
    w = pred[w]
  path.reverse()
  w = succ[path[^1]]
  while w != None:
    path.add(w)
    w = succ[w]
  return path

proc singleShortestPath(adj: Table[Node, HashSet[Node]], firstLevel: Table[Node, int], paths: var Table[Node, seq[Node]], cutoff: float, join: proc(p1: seq[Node], p2: seq[Node]): seq[Node]): Table[Node, seq[Node]] =
  var level = 0
  var nextLevel = firstLevel
  while len(nextLevel) != 0 and cutoff > level.float:
    var thisLevel = nextLevel
    nextLevel = initTable[Node, int]()
    for v in thisLevel.keys():
      for w in sorted(adj[v].toSeq()):
        if w notin paths:
          paths[w] = join(paths[v], @[w])
          nextLevel[w] = 1
    level += 1
  return paths
proc singleSourceShortestPath*(G: Graph, source: Node, cutoff: int = -1): Table[Node, seq[Node]] =
  if source notin G.nodesSet():
    raise newNNNodeNotFound(source)
  let join = proc(p1, p2: seq[Node]): seq[Node] = return p1 & p2
  var cutoffUsing: float = cutoff.float
  if cutoff == -1:
    cutoffUsing = Inf
  let nextLevel = {source: 1}.toTable()
  var paths = {source: @[source]}.toTable()
  return singleShortestPath(G.adj, nextLevel, paths, cutoffUsing, join)

proc singleSourceShortestPath*(DG: DiGraph, source: Node, cutoff: int = -1): Table[Node, seq[Node]] =
  if source notin DG.nodesSet():
    raise newNNNodeNotFound(source)
  let join = proc(p1, p2: seq[Node]): seq[Node] = return p1 & p2
  var cutoffUsing: float = cutoff.float
  if cutoff == -1:
    cutoffUsing = Inf
  let nextLevel = {source: 1}.toTable()
  var paths = {source: @[source]}.toTable()
  return singleShortestPath(DG.succ, nextLevel, paths, cutoffUsing, join)
proc singleTargetShortestPath*(G: Graph, target: Node, cutoff: int = -1): Table[Node, seq[Node]] =
  if target notin G.nodesSet():
    raise newNNNodeNotFound(target)
  let join = proc(p1, p2: seq[Node]): seq[Node] = return p2 & p1
  var cutoffUsing: float = cutoff.float
  if cutoff == -1:
    cutoffUsing = Inf
  let nextLevel = {target: 1}.toTable()
  var paths = {target: @[target]}.toTable()
  return singleShortestPath(G.adj, nextLevel, paths, cutoffUsing, join)

proc singleTargetShortestPath*(DG: DiGraph, target: Node, cutoff: int = -1): Table[Node, seq[Node]] =
  if target notin DG.nodesSet():
    raise newNNNodeNotFound(target)
  let join = proc(p1, p2: seq[Node]): seq[Node] = return p2 & p1
  var cutoffUsing: float = cutoff.float
  if cutoff == -1:
    cutoffUsing = Inf
  let nextLevel = {target: 1}.toTable()
  var paths = {target: @[target]}.toTable()
  return singleShortestPath(DG.pred, nextLevel, paths, cutoffUsing, join)

iterator allPairsShortestPath*(G: Graph, cutoff: int = -1): tuple[node: Node, shortestPaths: Table[Node, seq[Node]]] =
  for n in G.nodes():
    yield (n, singleSourceShortestPath(G, n, cutoff))
iterator allPairsShortestPath*(DG: DiGraph, cutoff: int = -1): tuple[node: Node, shortestPaths: Table[Node, seq[Node]]] =
  for n in DG.nodes():
    yield (n, singleSourceShortestPath(DG, n, cutoff))

proc predecessor*(G: Graph, source: Node, target: Node = None, cutoff: int = -1): Table[Node, seq[Node]] =
  if source notin G.nodesSet():
    raise newNNNodeNotFound(source)
  var level = 0
  var nextLevel = @[source]
  var seen = {source: level}.toTable()
  var pred = initTable[Node, seq[Node]]()
  pred[source] = @[]
  while len(nextLevel) != 0:
    level += 1
    var thisLevel = nextLevel
    nextLevel = @[]
    for v in thisLevel:
      for w in G.neighbors(v):
        if w notin seen:
          pred[w] = @[v]
          seen[w] = level
          nextLevel.add(w)
        elif seen[w] == level:
          pred[w].add(v)
    if cutoff != -1 and cutoff <= level:
      break
  if target != None:
    if target notin pred:
      var ret = initTable[Node, seq[Node]]()
      ret[target] = @[]
      return ret
    return {target: pred[target]}.toTable()
  return pred
proc predecessor*(DG: DiGraph, source: Node, target: Node = None, cutoff: int = -1): Table[Node, seq[Node]] =
  if source notin DG.nodesSet():
    raise newNNNodeNotFound(source)
  var level = 0
  var nextLevel = @[source]
  var seen = {source: level}.toTable()
  var pred = initTable[Node, seq[Node]]()
  pred[source] = @[]
  while len(nextLevel) != 0:
    level += 1
    var thisLevel = nextLevel
    nextLevel = @[]
    for v in thisLevel:
      for w in DG.successors(v):
        if w notin seen:
          pred[w] = @[v]
          seen[w] = level
          nextLevel.add(w)
        elif seen[w] == level:
          pred[w].add(v)
    if cutoff != -1 and cutoff <= level:
      break
  if target != None:
    if target notin pred:
      var ret = initTable[Node, seq[Node]]()
      ret[target] = @[]
      return ret
    return {target: pred[target]}.toTable()
  return pred

proc predecessorAndSeen*(G: Graph, source: Node, target: Node = None, cutoff: int = -1): tuple[pred: Table[Node, seq[Node]], seen: Table[Node, int]] =
  if source notin G.nodesSet():
    raise newNNNodeNotFound(source)
  var level = 0
  var nextLevel = @[source]
  var seen = {source: level}.toTable()
  var pred = initTable[Node, seq[Node]]()
  pred[source] = @[]
  while len(nextLevel) != 0:
    level += 1
    var thisLevel = nextLevel
    nextLevel = @[]
    for v in thisLevel:
      for w in G.neighbors(v):
        if w notin seen:
          pred[w] = @[v]
          seen[w] = level
          nextLevel.add(w)
        elif seen[w] == level:
          pred[w].add(v)
    if cutoff != -1 and cutoff <= level:
      break
  if target != None:
    if target notin pred:
      var retPred = initTable[Node, seq[Node]]()
      retPred[target] = @[]
      var retSeen = initTable[Node, int]()
      retSeen[target] = -1
      return (retPred, retSeen)
    return ({target: pred[target]}.toTable(), {target: seen[target]}.toTable())
  return (pred, seen)
proc predecessorAndSeen*(DG: DiGraph, source: Node, target: Node = None, cutoff: int = -1): tuple[pred: Table[Node, seq[Node]], seen: Table[Node, int]] =
  if source notin DG.nodesSet():
    raise newNNNodeNotFound(source)
  var level = 0
  var nextLevel = @[source]
  var seen = {source: level}.toTable()
  var pred = initTable[Node, seq[Node]]()
  pred[source] = @[]
  while len(nextLevel) != 0:
    level += 1
    var thisLevel = nextLevel
    nextLevel = @[]
    for v in thisLevel:
      for w in DG.successors(v):
        if w notin seen:
          pred[w] = @[v]
          seen[w] = level
          nextLevel.add(w)
        elif seen[w] == level:
          pred[w].add(v)
    if cutoff != -1 and cutoff <= level:
      break
  if target != None:
    if target notin pred:
      var retPred = initTable[Node, seq[Node]]()
      retPred[target] = @[]
      var retSeen = initTable[Node, int]()
      retSeen[target] = -1
      return (retPred, retSeen)
    return ({target: pred[target]}.toTable(), {target: seen[target]}.toTable())
  return (pred, seen)

proc dijkstraMultiSource(
  G: Graph,
  sources: seq[Node],
  weight: TableRef[Edge, float],
  pred: TableRef[Node, seq[Node]] = nil,
  paths: TableRef[Node, seq[Node]] = nil,
  cutoff: float = NaN,
  target: Node = None
): Table[Node, float] =
  if len(G.edges()) != len(weight):
    raise newNNError("all edge weights are needed")
  var dist = initTable[Node, float]()
  var seen = initTable[Node, float]()
  var c = -1
  var fringe = initHeapQueue[tuple[dist: float, cnt: int, node: Node]]()
  for source in sources:
    seen[source] = 0.0
    c += 1
    fringe.push((0.0, c, source))
  while len(fringe) != 0:
    var (d, _, v) = fringe.pop()
    if v in dist:
      continue
    dist[v] = d
    if v == target:
      break
    for u in G.neighbors(v):
      var cost: float
      if weight != nil:
        cost = weight.getOrDefault((v, u), weight.getOrDefault((u, v), 1.0))
      else:
        cost = 1.0
      var vuDist = dist[v] + cost
      if cutoff != NaN:
        if vuDist > cutoff:
          continue
      if u in dist:
        var uDist = dist[u]
        if vuDist < uDist:
          raise newNNError("contradictory path found: negative weights?")
        elif pred != nil and vuDist == uDist:
          pred[u].add(v)
      elif u notin seen or vuDist < seen[u]:
        seen[u] = vuDist
        c += 1
        fringe.push((vuDist, c, u))
        if paths != nil:
          paths[u] = paths[v] & @[u]
        if pred != nil:
          pred[u] = @[v]
      elif vuDist == seen[u]:
        if pred != nil:
          pred[u].add(v)
  return dist
proc dijkstraMultiSource(
  DG: DiGraph,
  sources: seq[Node],
  weight: TableRef[Edge, float],
  pred: TableRef[Node, seq[Node]] = nil,
  paths: TableRef[Node, seq[Node]] = nil,
  cutoff: float = NaN,
  target: Node = None
): Table[Node, float] =
  if len(DG.edges()) != len(weight):
    raise newNNError("all edge weights are needed")
  var dist = initTable[Node, float]()
  var seen = initTable[Node, float]()
  var c = -1
  var fringe = initHeapQueue[tuple[dist: float, cnt: int, node: Node]]()
  for source in sources:
    seen[source] = 0.0
    c += 1
    fringe.push((0.0, c, source))
  while len(fringe) != 0:
    var (d, _, v) = fringe.pop()
    if v in dist:
      continue
    dist[v] = d
    if v == target:
      break
    for u in DG.successors(v):
      var cost: float
      if weight != nil:
        cost = weight[(v, u)]
      else:
        cost = 1.0
      var vuDist = dist[v] + cost
      if cutoff != NaN:
        if vuDist > cutoff:
          continue
      if u in dist:
        var uDist = dist[u]
        if vuDist < uDist:
          raise newNNError("contradictory path found: negative weights?")
        elif pred != nil and vuDist == uDist:
          pred[u].add(v)
      elif u notin seen or vuDist < seen[u]:
        seen[u] = vuDist
        c += 1
        fringe.push((vuDist, c, u))
        if paths != nil:
          paths[u] = paths[v] & @[u]
        if pred != nil:
          pred[u] = @[v]
      elif vuDist == seen[u]:
        if pred != nil:
          pred[u].add(v)
  return dist
proc dijkstra(
  G: Graph,
  source: Node,
  weight: TableRef[Edge, float],
  pred: TableRef[Node, seq[Node]] = nil,
  paths: TableRef[Node, seq[Node]] = nil,
  cutoff: float = NaN,
  target: Node = None
): Table[Node, float] =
  return dijkstraMultiSource(G, @[source], weight, pred, paths, cutoff, target)
proc dijkstra(
  DG: DiGraph,
  source: Node,
  weight: TableRef[Edge, float],
  pred: TableRef[Node, seq[Node]] = nil,
  paths: TableRef[Node, seq[Node]] = nil,
  cutoff: float = NaN,
  target: Node = None
): Table[Node, float] =
  return dijkstraMultiSource(DG, @[source], weight, pred, paths, cutoff, target)

proc multiSourceDijkstra*(
  G: Graph,
  sources: seq[Node],
  target: Node = None,
  cutoff: float = NaN,
  weight: TableRef[Edge, float],
): tuple[dist: Table[Node, float], paths: Table[Node, seq[Node]]] =
  if len(sources) == 0:
    raise newNNError("sources must not be empty")
  for s in sources:
    if s notin G.nodesSet():
      raise newNNNodeNotFound(s)
  if target in sources.toHashSet():
    return ({target: 0.0}.toTable(), {target: newSeq[Node]()}.toTable())
  var paths = newTable[Node, seq[Node]]()
  for source in sources:
    paths[source] = @[source]
  var dist = dijkstraMultiSource(G, sources, weight=weight, pred=nil, paths=paths, cutoff=cutoff, target=target)
  if target == None:
    return (dist, paths[])
  var retDist = initTable[Node, float]()
  var retPath = initTable[Node, seq[Node]]()
  retDist[target] = dist[target]
  retPath[target] = paths[target]
  return (retDist, retPath)
proc multiSourceDijkstra*(
  DG: DiGraph,
  sources: seq[Node],
  target: Node = None,
  cutoff: float = NaN,
  weight: TableRef[Edge, float],
): tuple[dist: Table[Node, float], paths: Table[Node, seq[Node]]] =
  if len(sources) == 0:
    raise newNNError("sources must not be empty")
  for s in sources:
    if s notin DG.nodesSet():
      raise newNNNodeNotFound(s)
  if target in sources.toHashSet():
    return ({target: 0.0}.toTable(), {target: newSeq[Node]()}.toTable())
  var paths = newTable[Node, seq[Node]]()
  for source in sources:
    paths[source] = @[source]
  var dist = dijkstraMultiSource(DG, sources, weight=weight, pred=nil, paths=paths, cutoff=cutoff, target=target)
  if target == None:
    return (dist, paths[])
  var retDist = initTable[Node, float]()
  var retPath = initTable[Node, seq[Node]]()
  retDist[target] = dist[target]
  retPath[target] = paths[target]
  return (retDist, retPath)

proc multiSourceDijkstraPathLength*(
  G: Graph,
  sources: seq[Node],
  cutoff: float = NaN,
  weight: TableRef[Edge, float],
): Table[Node, float] =
  if len(sources) == 0:
    raise newNNError("sources must not be empty")
  for s in sources:
    if s notin G.nodesSet():
      raise newNNNodeNotFound(s)
  return dijkstraMultiSource(G, sources, weight=weight, cutoff=cutoff)
proc multiSourceDijkstraPathLength*(
  DG: DiGraph,
  sources: seq[Node],
  cutoff: float = NaN,
  weight: TableRef[Edge, float],
): Table[Node, float] =
  if len(sources) == 0:
    raise newNNError("sources must not be empty")
  for s in sources:
    if s notin DG.nodesSet():
      raise newNNNodeNotFound(s)
  return dijkstraMultiSource(DG, sources, weight=weight, cutoff=cutoff)

proc multiSourceDijkstraPath*(
  G: Graph,
  sources: seq[Node],
  cutoff: float = NaN,
  weight: TableRef[Edge, float],
): Table[Node, seq[Node]] =
  return multiSourceDijkstra(G, sources, cutoff=cutoff, weight=weight).paths
proc multiSourceDijkstraPath*(
  DG: DiGraph,
  sources: seq[Node],
  cutoff: float = NaN,
  weight: TableRef[Edge, float],
): Table[Node, seq[Node]] =
  return multiSourceDijkstra(DG, sources, cutoff=cutoff, weight=weight).paths

proc singleSourceDijkstra*(
  G: Graph,
  source: Node,
  target: Node = None,
  cutoff: float = NaN,
  weight: TableRef[Edge, float],
): tuple[dist: Table[Node, float], paths: Table[Node, seq[Node]]] =
  return multiSourceDijkstra(G, @[source], cutoff=cutoff, target=target, weight=weight)
proc singleSourceDijkstra*(
  DG: DiGraph,
  source: Node,
  target: Node = None,
  cutoff: float = NaN,
  weight: TableRef[Edge, float],
): tuple[dist: Table[Node, float], paths: Table[Node, seq[Node]]] =
  return multiSourceDijkstra(DG, @[source], cutoff=cutoff, target=target, weight=weight)

proc singleSourceDijkstraPathLength*(
  G: Graph,
  source: Node,
  cutoff: float = NaN,
  weight: TableRef[Edge, float],
): Table[Node, float] =
  return multiSourceDijkstraPathLength(G, @[source], cutoff=cutoff, weight=weight)
proc singleSourceDijkstraPathLength*(
  DG: DiGraph,
  source: Node,
  cutoff: float = NaN,
  weight: TableRef[Edge, float],
): Table[Node, float] =
  return multiSourceDijkstraPathLength(DG, @[source], cutoff=cutoff, weight=weight)

proc singleSourceDijkstraPath*(
  G: Graph,
  source: Node,
  cutoff: float = NaN,
  weight: TableRef[Edge, float],
): Table[Node, seq[Node]] =
  return multiSourceDijkstraPath(G, @[source], cutoff=cutoff, weight=weight)
proc singleSourceDijkstraPath*(
  DG: DiGraph,
  source: Node,
  cutoff: float = NaN,
  weight: TableRef[Edge, float],
): Table[Node, seq[Node]] =
  return multiSourceDijkstraPath(DG, @[source], cutoff=cutoff, weight=weight)

proc dijkstraPathLength*(
  G: Graph,
  source: Node,
  target: Node,
  weight: TableRef[Edge, float]
): float =
  if source notin G.nodesSet():
    raise newNNNodeNotFound(source)
  if source == target:
    return 0.0
  let length = dijkstra(G, source, weight=weight, target=target)
  try:
    return length[target]
  except KeyError:
    raise newNNNoPath(fmt"node {target} not reachable from node {source}")
proc dijkstraPathLength*(
  DG: DiGraph,
  source: Node,
  target: Node,
  weight: TableRef[Edge, float]
): float =
  if source notin DG.nodesSet():
    raise newNNNodeNotFound(source)
  if source == target:
    return 0.0
  let length = dijkstra(DG, source, weight=weight, target=target)
  try:
    return length[target]
  except KeyError:
    raise newNNNoPath(fmt"node {target} not reachable from node {source}")

proc dijkstraPath*(
  G: Graph,
  source: Node,
  target: Node,
  weight: TableRef[Edge, float]
): seq[Node] =
  try:
    return singleSourceDijkstra(G, source, target=target, weight=weight).paths[target]
  except KeyError:
    raise newNNNoPath(fmt"node {target} not reachable from node {source}")
proc dijkstraPath*(
  DG: DiGraph,
  source: Node,
  target: Node,
  weight: TableRef[Edge, float]
): seq[Node] =
  try:
    return singleSourceDijkstra(DG, source, target=target, weight=weight).paths[target]
  except KeyError:
    raise newNNNoPath(fmt"node {target} not reachable from node {source}")

iterator allPairsDijkstra*(
  G: Graph,
  cutoff: float = NaN,
  weight: TableRef[Edge, float]
): tuple[node: Node, result: tuple[dist: Table[Node, float], paths: Table[Node, seq[Node]]]] =
  for n in G.nodes():
    var (dist, paths) = singleSourceDijkstra(G, n, cutoff=cutoff, weight=weight)
    yield (n, (dist, paths))
iterator allPairsDijkstra*(
  DG: DiGraph,
  cutoff: float = NaN,
  weight: TableRef[Edge, float]
): tuple[node: Node, result: tuple[dist: Table[Node, float], paths: Table[Node, seq[Node]]]] =
  for n in DG.nodes():
    var (dist, paths) = singleSourceDijkstra(DG, n, cutoff=cutoff, weight=weight)
    yield (n, (dist, paths))

iterator allPairsDijkstraPathLength*(
  G: Graph,
  cutoff: float = NaN,
  weight: TableRef[Edge, float]
): tuple[node: Node, dist: Table[Node, float]] =
  for n in G.nodes():
    let dist = singleSourceDijkstraPathLength(G, n, cutoff=cutoff, weight=weight)
    yield (n, dist)
iterator allPairsDijkstraPathLength*(
  DG: DiGraph,
  cutoff: float = NaN,
  weight: TableRef[Edge, float]
): tuple[node: Node, dist: Table[Node, float]] =
  for n in DG.nodes():
    let dist = singleSourceDijkstraPathLength(DG, n, cutoff=cutoff, weight=weight)
    yield (n, dist)

iterator allPairsDijkstraPath*(
  G: Graph,
  cutoff: float = NaN,
  weight: TableRef[Edge, float]
): tuple[node: Node, paths: Table[Node, seq[Node]]] =
  for n in G.nodes():
    let paths = singleSourceDijkstraPath(G, n, cutoff=cutoff, weight=weight)
    yield (n, paths)
iterator allPairsDijkstraPath*(
  DG: DiGraph,
  cutoff: float = NaN,
  weight: TableRef[Edge, float]
): tuple[node: Node, paths: Table[Node, seq[Node]]] =
  for n in DG.nodes():
    let paths = singleSourceDijkstraPath(DG, n, cutoff=cutoff, weight=weight)
    yield (n, paths)

proc dijkstraPredecessorAndDistance*(
  G: Graph,
  source: Node,
  cutoff: float = NaN,
  weight: TableRef[Edge, float]
): tuple[pred: Table[Node, seq[Node]], distance: Table[Node, float]] =
  if source notin G.nodesSet():
    raise newNNNodeNotFound(source)
  let pred = newTable[Node, seq[Node]]()
  pred[source] = @[]
  let dist = dijkstra(G, source, weight=weight, pred=pred, cutoff=cutoff)
  return (pred[], dist)
proc dijkstraPredecessorAndDistance*(
  DG: DiGraph,
  source: Node,
  cutoff: float = NaN,
  weight: TableRef[Edge, float]
): tuple[pred: Table[Node, seq[Node]], distance: Table[Node, float]] =
  if source notin DG.nodesSet():
    raise newNNNodeNotFound(source)
  let pred = newTable[Node, seq[Node]]()
  pred[source] = @[]
  let dist = dijkstra(DG, source, weight=weight, pred=pred, cutoff=cutoff)
  return (pred[], dist)

iterator buildPathsFromPredecessors(
  sources: HashSet[Node],
  target: Node,
  pred: TableRef[Node, seq[Node]]
): seq[Node] =
  if target notin pred:
    raise newNNNoPath(fmt"target {target} not reachable from give source nodes")
  var seen = @[target].toHashSet()
  var stack: Deque[tuple[node: Node, i: int]] = initDeque[tuple[node: Node, i: int]]()
  stack.addLast((target, 0))
  var top = 0
  while 0 <= top:
    var (node, i) = stack.toSeq()[top]
    if node in sources:
      var ret: seq[Node] = @[]
      var revSubStack: seq[tuple[node: Node, i: int]] = @[]
      for j in 0..top:
        revSubStack.add(stack[j])
      revSubStack.reverse()
      for (p, n) in revSubStack:
        ret.add(p)
      yield ret
    if len(pred[node]) > i:
      var tmp = stack.toSeq()
      tmp[top].i = i + 1
      stack = tmp.toDeque()
      var nextNode = pred[node][i]
      if nextNode in seen:
        continue
      else:
        seen.incl(nextNode)
      top += 1
      if top == len(stack):
        stack.addLast((nextNode, 0))
      else:
        tmp = stack.toSeq()
        tmp[top] = (nextNode, 0)
        stack = tmp.toDeque()
    else:
      seen.excl(node)
      top -= 1

proc innerBellmanFord(
  G: Graph,
  sources: seq[Node],
  weight: TableRef[Edge, float],
  pred: TableRef[Node, seq[Node]] = nil,
  dist: TableRef[Node, float] = nil,
  heuristic: bool = true
): Node =
  if len(G.edges()) != len(weight):
    raise newNNError("all edge weights are needed")
  for s in sources:
    if s notin G.nodesSet():
      raise newNNNodeNotFound(s)
  var predUsing = newTable[Node, seq[Node]]()
  if pred == nil:
    for v in sources:
      predUsing[v] = @[]
  else:
    predUsing = pred
  var distUsing = newTable[Node, float]()
  if dist == nil:
    for v in sources:
      distUsing[v] = 0.0
  else:
    distUsing = dist
  var nonExistentEdge = (None, None)
  var predEdge = initTable[Node, Node]()
  for v in sources:
    predEdge[v] = None
  var recentUpdate = initTable[Node, Edge]()
  for v in sources:
    recentUpdate[v] = nonExistentEdge
  let N = len(G)
  var count = initTable[Node, int]()
  var q = sources.toDeque()
  var inQ = sources.toHashSet()
  while len(q) != 0:
    var u = q.popFirst()
    inQ.excl(u)
    var t: seq[bool] = @[]
    for predU in predUsing[u]:
      t.add(predU notin inQ)
    if all(t, proc(b: bool): bool = return b):
      var distU = distUsing[u]
      for v in G.neighbors(u):
        var distV: float
        if weight != nil:
          distV = distU + weight.getOrDefault((u, v), weight.getOrDefault((v, u), Inf))
        else:
          distV = distU + 1.0
        if distV < distUsing.getOrDefault(v, Inf):
          if heuristic:
            if v == recentUpdate[u].u or v == recentUpdate[u].v:
              predUsing[v].add(u)
              return v
            if v in predEdge and predEdge[v] == u:
              recentUpdate[v] = recentUpdate[u]
            else:
              recentUpdate[v] = (u, v)
          if v notin inQ:
            q.addLast(v)
            inQ.incl(v)
            var countV = count.getOrDefault(v, 0) + 1
            if countV == N:
              return v
            count[v] = countV
          distUsing[v] = distV
          predUsing[v] = @[u]
          predEdge[v] = u
        elif distUsing[v] != 0.0 and distV == distUsing[v]:
          predUsing[v].add(u)
  return None
proc innerBellmanFord(
  DG: DiGraph,
  sources: seq[Node],
  weight: TableRef[Edge, float],
  pred: TableRef[Node, seq[Node]] = nil,
  dist: TableRef[Node, float] = nil,
  heuristic: bool = true
): Node =
  if len(DG.edges()) != len(weight):
    raise newNNError("all edge weights are needed")
  for s in sources:
    if s notin DG.nodesSet():
      raise newNNNodeNotFound(s)
  var predUsing = newTable[Node, seq[Node]]()
  if pred == nil:
    for v in sources:
      predUsing[v] = @[]
  else:
    predUsing = pred
  var distUsing = newTable[Node, float]()
  if dist == nil:
    for v in sources:
      distUsing[v] = 0.0
  else:
    distUsing = dist
  var nonExistentEdge = (None, None)
  var predEdge = initTable[Node, Node]()
  for v in sources:
    predEdge[v] = None
  var recentUpdate = initTable[Node, Edge]()
  for v in sources:
    recentUpdate[v] = nonExistentEdge
  let N = len(DG)
  var count = initTable[Node, int]()
  var q = sources.toDeque()
  var inQ = sources.toHashSet()
  while len(q) != 0:
    var u = q.popFirst()
    inQ.excl(u)
    var t: seq[bool] = @[]
    for predU in predUsing[u]:
      t.add(predU notin inQ)
    if all(t, proc(b: bool): bool = return b):
      var distU = distUsing[u]
      for v in DG.successors(u):
        var distV: float
        if weight != nil:
          distV = distU + weight.getOrDefault((u, v), Inf)
        else:
          distV = distU + 1.0
        if distV < distUsing.getOrDefault(v, Inf):
          if heuristic:
            if v == recentUpdate[u].u or v == recentUpdate[u].v:
              predUsing[v].add(u)
              return v
            if v in predEdge and predEdge[v] == u:
              recentUpdate[v] = recentUpdate[u]
            else:
              recentUpdate[v] = (u, v)
          if v notin inQ:
            q.addLast(v)
            inQ.incl(v)
            var countV = count.getOrDefault(v, 0) + 1
            if countV == N:
              return v
            count[v] = countV
          distUsing[v] = distV
          predUsing[v] = @[u]
          predEdge[v] = u
        elif distUsing[v] != 0.0 and distV == distUsing[v]:
          predUsing[v].add(u)
  return None

proc bellmanFord(
  G: Graph,
  source: seq[Node],
  weight: TableRef[Edge, float],
  pred: TableRef[Node, seq[Node]] = nil,
  paths: TableRef[Node, seq[Node]] = nil,
  dist: TableRef[Node, float] = nil,
  target: Node = None,
  heuristic: bool = true
): TableRef[Node, float] =
  var predUsing = newTable[Node, seq[Node]]()
  if pred == nil:
    for v in source:
      predUsing[v] = @[]
  else:
    predUsing = pred
  var distUsing = newTable[Node, float]()
  if dist == nil:
    for v in source:
      distUsing[v] = 0.0
  else:
    distUsing = dist
  let negativeCycleFound = innerBellmanFord(G, source, weight, predUsing, distUsing, heuristic)
  if negativeCycleFound != None:
    raise newNNUnbounded("negative cycle detected")
  if paths != nil:
    var dsts = newTable[Node, seq[Node]]()
    if target != None:
      dsts[target] = @[target]
    else:
      dsts = predUsing
    for dst in dsts.keys():
      for x in  buildPathsFromPredecessors(source.toHashSet(), dst, predUsing):
        paths[dst] = x
        break
  return distUsing
proc bellmanFord(
  DG: DiGraph,
  source: seq[Node],
  weight: TableRef[Edge, float],
  pred: TableRef[Node, seq[Node]] = nil,
  paths: TableRef[Node, seq[Node]] = nil,
  dist: TableRef[Node, float] = nil,
  target: Node = None,
  heuristic: bool = true
): TableRef[Node, float] =
  var predUsing= newTable[Node, seq[Node]]()
  if pred == nil:
    for v in source:
      predUsing[v] = @[]
  else:
    predUsing = pred
  var distUsing = newTable[Node, float]()
  if dist == nil:
    for v in source:
      distUsing[v] = 0.0
  else:
    distUsing = dist
  let negativeCycleFound = innerBellmanFord(DG, source, weight, predUsing, distUsing, heuristic)
  if negativeCycleFound != None:
    raise newNNUnbounded("negative cycle detected")
  if paths != nil:
    var dsts = newTable[Node, seq[Node]]()
    if target != None:
      dsts[target] = @[target]
    else:
      dsts = predUsing
    for dst in dsts.keys():
      for x in  buildPathsFromPredecessors(source.toHashSet(), dst, predUsing):
        paths[dst] = x
        break
  return distUsing

proc singleSourceBellmanFord*(
  G: Graph,
  source: Node,
  target: Node = None,
  weight: TableRef[Edge, float],
): tuple[distance: Table[Node, float], paths: Table[Node, seq[Node]]] =
  if source == target:
    if source notin G.nodesSet():
      raise newNNNodeNotFound(source)
    var dist = initTable[Node, float]()
    dist[target] = 0
    var paths = initTable[Node, seq[Node]]()
    paths[target] = @[source]
    return (dist, paths)
  var paths = newTable[Node, seq[Node]]()
  var dist = bellmanFord(G, @[source], weight=weight, paths=paths, target=target)
  if target == None:
    return (dist[], paths[])
  try:
    var retDist = initTable[Node, float]()
    retDist[target] = dist[target]
    var retPaths = initTable[Node, seq[Node]]()
    retPaths[target] = paths[target]
    return (retDist, retPaths)
  except KeyError:
    raise newNNNoPath(fmt"target {target} not reachable from source {source}")
proc singleSourceBellmanFord*(
  DG: DiGraph,
  source: Node,
  target: Node = None,
  weight: TableRef[Edge, float],
): tuple[distance: Table[Node, float], paths: Table[Node, seq[Node]]] =
  if source == target:
    if source notin DG.nodesSet():
      raise newNNNodeNotFound(source)
    var dist = initTable[Node, float]()
    dist[target] = 0
    var paths = initTable[Node, seq[Node]]()
    paths[target] = @[source]
    return (dist, paths)
  var paths = newTable[Node, seq[Node]]()
  var dist = bellmanFord(DG, @[source], weight=weight, paths=paths, target=target)
  if target == None:
    return (dist[], paths[])
  try:
    var retDist = initTable[Node, float]()
    retDist[target] = dist[target]
    var retPaths = initTable[Node, seq[Node]]()
    retPaths[target] = paths[target]
    return (retDist, retPaths)
  except KeyError:
    raise newNNNoPath(fmt"target {target} not reachable from source {source}")

proc singleSourceBellmanFordPathLength*(
  G: Graph,
  source: Node,
  weight: TableRef[Edge, float],
): Table[Node, float] =
  return bellmanFord(G, @[source], weight=weight)[]
proc singleSourceBellmanFordPathLength*(
  DG: DiGraph,
  source: Node,
  weight: TableRef[Edge, float],
): Table[Node, float] =
  return bellmanFord(DG, @[source], weight=weight)[]

proc singleSourceBellmanFordPath*(
  G: Graph,
  source: Node,
  weight: TableRef[Edge, float],
): Table[Node, seq[Node]] =
  return singleSourceBellmanFord(G, source, weight=weight).paths
proc singleSourceBellmanFordPath*(
  DG: DiGraph,
  source: Node,
  weight: TableRef[Edge, float],
): Table[Node, seq[Node]] =
  return singleSourceBellmanFord(DG, source, weight=weight).paths

proc bellmanFordPathLength*(
  G: Graph,
  source: Node,
  target: Node,
  weight: TableRef[Edge, float],
): float =
  if source == target:
    if source notin G.nodesSet():
      raise newNNNodeNotFound(source)
    return 0.0
  let length = bellmanFord(G, @[source], weight=weight, target=target)
  try:
    return length[target]
  except KeyError:
    raise newNNNoPath(fmt"target {target} not reachable from source {source}")
proc bellmanFordPathLength*(
  DG: DiGraph,
  source: Node,
  target: Node,
  weight: TableRef[Edge, float],
): float =
  if source == target:
    if source notin DG.nodesSet():
      raise newNNNodeNotFound(source)
    return 0.0
  let length = bellmanFord(DG, @[source], weight=weight, target=target)
  try:
    return length[target]
  except KeyError:
    raise newNNNoPath(fmt"target {target} not reachable from source {source}")

proc bellmanFordPath*(
  G: Graph,
  source: Node,
  target: Node,
  weight: TableRef[Edge, float],
): seq[Node] =
  try:
    return singleSourceBellmanFord(G, source, target=target, weight=weight).paths[target]
  except NNNoPath:
    raise newNNNoPath(fmt"target {target} not reachable from source {source}")
proc bellmanFordPath*(
  DG: DiGraph,
  source: Node,
  target: Node,
  weight: TableRef[Edge, float],
): seq[Node] =
  try:
    return singleSourceBellmanFord(DG, source, target=target, weight=weight).paths[target]
  except NNNoPath:
    raise newNNNoPath(fmt"target {target} not reachable from source {source}")

iterator allPairsBellmanFordPathLength*(
  G: Graph,
  weight: TableRef[Edge, float],
): tuple[node: Node, dist: Table[Node, float]] =
  for n in G.nodes():
    yield (n, singleSourceBellmanFordPathLength(G, n, weight=weight))
iterator allPairsBellmanFordPathLength*(
  DG: DiGraph,
  weight: TableRef[Edge, float],
): tuple[node: Node, dist: Table[Node, float]] =
  for n in DG.nodes():
    yield (n, singleSourceBellmanFordPathLength(DG, n, weight=weight))

iterator allPairsBellmanFordPath*(
  G: Graph,
  weight: TableRef[Edge, float],
): tuple[node: Node, paths: Table[Node, seq[Node]]] =
  for n in G.nodes():
    yield (n, singleSourceBellmanFordPath(G, n, weight=weight))
iterator allPairsBellmanFordPath*(
  DG: DiGraph,
  weight: TableRef[Edge, float],
): tuple[node: Node, paths: Table[Node, seq[Node]]] =
  for n in DG.nodes():
    yield (n, singleSourceBellmanFordPath(DG, n, weight=weight))

proc bellmanFordPredecessorAndDistance*(
  G: Graph,
  source: Node,
  target: Node = None,
  weight: TableRef[Edge, float],
  heuristic: bool = false
): tuple[pred: Table[Node, seq[Node]], dist: Table[Node, float]] =
  if source notin G.nodesSet():
    raise newNNNodeNotFound(source)
  for (u, v) in G.selfloopEdges():
    if weight.getOrDefault((u, v), weight.getOrDefault((v, u), Inf)) < 0:
      raise newNNUnbounded(fmt"negative cycle detected")

  var dist = newTable[Node, float]()
  dist[source] = 0.0

  var pred = newTable[Node, seq[Node]]()
  pred[source] = @[]

  if len(G) == 1:
    return (pred[], dist[])

  dist = bellmanFord(G, @[source], weight=weight, pred=pred, dist=dist, target=target, heuristic=heuristic)
  return (pred[], dist[])
proc bellmanFordPredecessorAndDistance*(
  DG: DiGraph,
  source: Node,
  target: Node = None,
  weight: TableRef[Edge, float],
  heuristic: bool = false
): tuple[pred: Table[Node, seq[Node]], dist: Table[Node, float]] =
  if source notin DG.nodesSet():
    raise newNNNodeNotFound(source)
  for (u, v) in DG.selfloopEdges():
    if weight.getOrDefault((u, v), Inf) < 0:
      raise newNNUnbounded(fmt"negative cycle detected")

  var dist = newTable[Node, float]()
  dist[source] = 0.0

  var pred = newTable[Node, seq[Node]]()
  pred[source] = @[]

  if len(DG) == 1:
    return (pred[], dist[])

  dist = bellmanFord(DG, @[source], weight=weight, pred=pred, dist=dist, target=target, heuristic=heuristic)
  return (pred[], dist[])

proc goldbergRadzik*(
  G: Graph,
  source: Node,
  weight: TableRef[Edge, float],
): tuple[pred: Table[Node, Node], dist: Table[Node, float]] =
  if len(G.edges()) != len(weight):
    raise newNNError("all edge weights are needed")
  if source notin G.nodesSet():
    raise newNNNodeNotFound(source)
  for (u, v) in G.selfloopEdges():
    if weight.getOrDefault((u, v), weight.getOrDefault((v, u), Inf)) < 0:
      raise newNNUnbounded(fmt"negative cycle detected")

  var dist = newTable[Node, float]()
  dist[source] = 0.0
  var pred = newTable[Node, Node]()
  pred[source] = None

  if len(G) == 1:
    return (pred[], dist[])

  for u in G.nodes():
    dist[u] = Inf
  dist[source] = 0.0

  let topoSort = proc(relabeled: HashSet[Node]): seq[Node] =
    var toScan: seq[Node] = @[]
    var negCount = initTable[Node, int]()
    for u in relabeled:
      if u in negCount:
        continue
      var dU = dist[u]
      var tmp: seq[bool] = @[]
      for v in G.neighbors(u):
        tmp.add((dU + weight.getOrDefault((u, v), weight.getOrDefault((v, u), Inf))) >= dist[v])
      if all(tmp, proc(b: bool): bool = return b):
        continue
      var stack = initDeque[tuple[u: Node, nbrs: seq[Node], idx: int]]()
      stack.addFirst((u, G.neighbors(u), 0))
      var inStack = @[u].toHashSet()
      negCount[u] = 0
      while len(stack) != 0:
        var (u, nbrs, idx) = stack.peekLast()
        var v: Node
        if idx < len(nbrs):
          var tmpEle = stack.popLast()
          v = tmpEle.nbrs[idx]
          stack.addLast((u, nbrs, idx + 1))
        else:
          toScan.add(u)
          discard stack.popLast()
          inStack.excl(u)
          continue
        var t = dist[u] + weight.getOrDefault((u, v), weight.getOrDefault((v, u), Inf))
        var dV = dist[v]
        if t <= dV:
          var isNeg = t < dV
          dist[v] = t
          pred[v] = u
          if v notin negCount:
            negCount[v] = negCount[u] + isNeg.int
            stack.addLast((v, G.neighbors(v), 0))
            inStack.incl(v)
          elif v in inStack and negCount[u] + isNeg.int > negCount[v]:
            raise newNNUnbounded("negative cycle detected")
    toScan.reverse()
    return toScan
  let relax = proc(toScan: seq[Node]): HashSet[Node] =
    var relabeled = initHashSet[Node]()
    for u in toScan:
      var dU = dist[u]
      for v in G.neighbors(u):
        var wE = weight.getOrDefault((u, v), weight.getOrDefault((v, u), Inf))
        if dU + wE < dist[v]:
          dist[v] = dU + wE
          pred[v] = u
          relabeled.incl(v)
    return relabeled

  var relabeled = @[source].toHashSet()
  while len(relabeled) != 0:
    var toScan = topoSort(relabeled)
    relabeled = relax(toScan)
  var retD = newTable[Node, float]()
  for u in pred.keys():
    retD[u] = dist[u]
  return (pred[], retD[])
proc goldbergRadzik*(
  DG: DiGraph,
  source: Node,
  weight: TableRef[Edge, float] = nil,
): tuple[pred: Table[Node, Node], dist: Table[Node, float]] =
  if len(DG.edges()) != len(weight):
    raise newNNError("all edge weights are needed")
  if source notin DG.nodesSet():
    raise newNNNodeNotFound(source)
  for (u, v) in DG.selfloopEdges():
    if weight.getOrDefault((u, v), Inf) < 0:
      raise newNNUnbounded(fmt"negative cycle detected")

  var dist = newTable[Node, float]()
  dist[source] = 0.0
  var pred = newTable[Node, Node]()
  pred[source] = None

  if len(DG) == 1:
    return (pred[], dist[])

  for u in DG.nodes():
    dist[u] = Inf
  dist[source] = 0.0

  let topoSort = proc(relabeled: HashSet[Node]): seq[Node] =
    var toScan: seq[Node] = @[]
    var negCount = initTable[Node, int]()
    for u in relabeled:
      if u in negCount:
        continue
      var dU = dist[u]
      var tmp: seq[bool] = @[]
      for v in DG.successors(u):
        tmp.add((dU + weight.getOrDefault((u, v), Inf)) >= dist[v])
      if all(tmp, proc(b: bool): bool = return b):
        continue
      var stack = initDeque[tuple[u: Node, nbrs: seq[Node], idx: int]]()
      stack.addFirst((u, DG.successors(u), 0))
      var inStack = @[u].toHashSet()
      negCount[u] = 0
      while len(stack) != 0:
        var (u, nbrs, idx) = stack.peekLast()
        var v: Node
        if idx < len(nbrs):
          var tmpEle = stack.popLast()
          v = tmpEle.nbrs[idx]
          stack.addLast((u, nbrs, idx + 1))
        else:
          toScan.add(u)
          discard stack.popLast()
          inStack.excl(u)
          continue
        var t = dist[u] + weight.getOrDefault((u, v), Inf)
        var dV = dist[v]
        if t <= dV:
          var isNeg = t < dV
          dist[v] = t
          pred[v] = u
          if v notin negCount:
            negCount[v] = negCount[u] + isNeg.int
            stack.addLast((v, DG.successors(v), 0))
            inStack.incl(v)
          elif v in inStack and negCount[u] + isNeg.int > negCount[v]:
            raise newNNUnbounded("negative cycle detected")
    toScan.reverse()
    return toScan
  let relax = proc(toScan: seq[Node]): HashSet[Node] =
    var relabeled = initHashSet[Node]()
    for u in toScan:
      var dU = dist[u]
      for v in DG.successors(u):
        var wE = weight.getOrDefault((u, v), Inf)
        if dU + wE < dist[v]:
          dist[v] = dU + wE
          pred[v] = u
          relabeled.incl(v)
    return relabeled

  var relabeled = @[source].toHashSet()
  while len(relabeled) != 0:
    var toScan = topoSort(relabeled)
    relabeled = relax(toScan)
  var retD = newTable[Node, float]()
  for u in pred.keys():
    retD[u] = dist[u]
  return (pred[], retD[])

proc negativeEdgeCycle*(
  G: Graph,
  weight: TableRef[Edge, float],
  heuristic: bool = true
): bool =
  var newNode = -1
  var weightUsing = weight
  while newNode in G.nodesSet():
    newNode -= 1
  for n in G.nodes():
    G.addEdge(newNode, n)
    weightUsing[(newNode, n)] = 1.0
  try:
    discard bellmanFordPredecessorAndDistance(G, newNode, weight=weightUsing, heuristic=heuristic)
  except NNUnbounded:
    G.removeNode(newNode)
    return true
  G.removeNode(newNode)
  return true
proc negativeEdgeCycle*(
  DG: DiGraph,
  weight: TableRef[Edge, float],
  heuristic: bool = true
): bool =
  var newNode = -2
  var weightUsing = weight
  while newNode in DG.nodesSet():
    newNode -= 1
  for n in DG.nodes():
    DG.addEdge(newNode, n)
    weightUsing[(newNode, n)] = 1.0
  try:
    discard bellmanFordPredecessorAndDistance(DG, newNode, weight=weightUsing, heuristic=heuristic)
  except NNUnbounded:
    DG.removeNode(newNode)
    return true
  DG.removeNode(newNode)
  return true

proc findNegativeCycle*(
  G: Graph,
  source: Node,
  weight: TableRef[Edge, float]
): seq[Node] =
  var pred = newTable[Node, seq[Node]]()
  pred[source] = @[]
  var v = innerBellmanFord(G, @[source], weight=weight, pred=pred)
  if v == None:
    raise newNNError(fmt"no negative cycles detected")
  var negCycle: seq[Node] = @[]
  var stack = initDeque[tuple[v: Node, pred: seq[Node]]]()
  stack.addLast((v, pred[v]))
  var seen = @[v].toHashSet()
  while len(stack) != 0:
    var (node, preds) = stack.peekLast()
    if v in preds:
      negCycle.add(node)
      negCycle.add(v)
      negCycle.reverse()
      return negCycle
    if len(preds) != 0:
      var nbr = preds.pop()
      if nbr notin seen:
        stack.addLast((nbr, pred[nbr]))
        negCycle.add(node)
        seen.incl(nbr)
    else:
      discard stack.popLast()
      if len(negCycle) != 0:
        discard negCycle.pop()
      else:
        if v in G.neighborsSet(v) and weight.getOrDefault((v, v), Inf) < 0:
          return @[v, v]
        raise newNNError("negative cycle is detected but not found")
  raise newNNUnbounded("negative cycle detected but not identified")
proc findNegativeCycle*(
  DG: DiGraph,
  source: Node,
  weight: TableRef[Edge, float]
): seq[Node] =
  var pred = newTable[Node, seq[Node]]()
  pred[source] = @[]
  var v = innerBellmanFord(DG, @[source], weight=weight, pred=pred)
  if v == None:
    raise newNNError(fmt"no negative cycles detected")
  var negCycle: seq[Node] = @[]
  var stack = initDeque[tuple[v: Node, pred: seq[Node]]]()
  stack.addLast((v, pred[v]))
  var seen = @[v].toHashSet()
  while len(stack) != 0:
    var (node, preds) = stack.peekLast()
    if v in preds:
      negCycle.add(node)
      negCycle.add(v)
      negCycle.reverse()
      return negCycle
    if len(preds) != 0:
      var nbr = preds.pop()
      if nbr notin seen:
        stack.addLast((nbr, pred[nbr]))
        negCycle.add(node)
        seen.incl(nbr)
    else:
      discard stack.popLast()
      if len(negCycle) != 0:
        discard negCycle.pop()
      else:
        if v in DG.neighborsSet(v) and weight.getOrDefault((v, v), Inf) < 0:
          return @[v, v]
        raise newNNError("negative cycle is detected but not found")
  raise newNNUnbounded("negative cycle detected but not identified")

proc bidirectionalDijkstra*(
  G :Graph,
  source: Node,
  target: Node,
  weight: TableRef[Edge, float]
): tuple[length: float, path: seq[Node]] =
  if len(G.edges()) != len(weight):
    raise newNNError("all edge weights are needed")
  if source notin G.nodesSet():
    raise newNNNodeNotFound(source)
  if target notin G.nodesSet():
    raise newNNNodeNotFound(target)

  if source == target:
    return (0.0, @[source])

  var dists: seq[Table[Node, float]] = @[initTable[Node, float](), initTable[Node, float]()]
  var paths: seq[Table[Node, seq[Node]]] = @[{source: @[source]}.toTable(), {target: @[target]}.toTable()]
  var fringe: seq[HeapQueue[tuple[distance: float, cnt: int, node: Node]]] = @[initHeapQueue[tuple[distance: float, cnt: int, node: Node]](), initHeapQueue[tuple[distance: float, cnt: int, node: Node]]()]
  var seen: seq[Table[Node, float]] = @[{source: 0.0}.toTable(), {target: 0.0}.toTable()]
  var c = -1

  push(fringe[0], (0.0, c, source))
  c += 1
  push(fringe[1], (0.0, c, target))
  c += 1
  var neighs = @[G.adj, G.adj]

  var dir = 1
  var finalDist = Inf
  var finalPath: seq[Node] = @[]
  while len(fringe[0]) != 0 and len(fringe[1]) != 0:
    dir = 1 - dir
    var (dist, _, v) = pop(fringe[dir])
    if v in dists[dir]:
      continue
    dists[dir][v] = dist
    if v in dists[1 - dir]:
      return (finalDist, finalPath)
    for w in sorted(neighs[dir][v].toSeq()):
      var cost = weight.getOrDefault((v, w), weight.getOrDefault((w, v), Inf))
      if cost == NaN:
        continue
      var vwLength = dists[dir][v] + cost
      if w in dists[dir]:
        if vwLength < dists[dir][w]:
          raise newNNError("condtradictory paths found: negative weights?")
      elif w notin seen[dir] or vwLength < seen[dir][w]:
        seen[dir][w] = vwLength
        push(fringe[dir], (vwLength, c, w))
        c += 1
        paths[dir][w] = paths[dir][v] & @[w]
        if w in seen[0] and w in seen[1]:
          var totalDist = seen[0][w] + seen[1][w]
          if len(finalPath) == 0 or finalDist > totalDist:
            finalDist = totalDist
            var revPath = reversed(paths[1][w])
            finalPath = paths[0][w]
            for i in 1..<len(revPath):
              finalPath = finalPath & revPath[i]
  raise newNNNoPath(fmt"no path between source {source} and target {target}")
proc bidirectionalDijkstra*(
  DG :DiGraph,
  source: Node,
  target: Node,
  weight: TableRef[Edge, float]
): tuple[length: float, path: seq[Node]] =
  if len(DG.edges()) != len(weight):
    raise newNNError("all edge weights are needed")
  if source notin DG.nodesSet():
    raise newNNNodeNotFound(source)
  if target notin DG.nodesSet():
    raise newNNNodeNotFound(target)

  if source == target:
    return (0.0, @[source])

  var dists: seq[Table[Node, float]] = @[initTable[Node, float](), initTable[Node, float]()]
  var paths: seq[Table[Node, seq[Node]]] = @[{source: @[source]}.toTable(), {target: @[target]}.toTable()]
  var fringe: seq[HeapQueue[tuple[distance: float, cnt: int, node: Node]]] = @[initHeapQueue[tuple[distance: float, cnt: int, node: Node]](), initHeapQueue[tuple[distance: float, cnt: int, node: Node]]()]
  var seen: seq[Table[Node, float]] = @[{source: 0.0}.toTable(), {target: 0.0}.toTable()]
  var c = -1

  push(fringe[0], (0.0, c, source))
  c += 1
  push(fringe[1], (0.0, c, target))
  c += 1
  var neighs = @[DG.succ, DG.pred]

  var dir = 1
  var finalDist = Inf
  var finalPath: seq[Node] = @[]
  while len(fringe[0]) != 0 and len(fringe[1]) != 0:
    dir = 1 - dir
    var (dist, _, v) = pop(fringe[dir])
    if v in dists[dir]:
      continue
    dists[dir][v] = dist
    if v in dists[1 - dir]:
      return (finalDist, finalPath)
    for w in sorted(neighs[dir][v].toSeq()):
      var cost: float
      if dir == 0:
        cost = weight.getOrDefault((v, w), Inf)
      else:
        cost = weight.getOrDefault((w, v), Inf)
      if cost == NaN:
        continue
      var vwLength = dists[dir][v] + cost
      if w in dists[dir]:
        if vwLength < dists[dir][w]:
          raise newNNError("condtradictory paths found: negative weights?")
      elif w notin seen[dir] or vwLength < seen[dir][w]:
        seen[dir][w] = vwLength
        push(fringe[dir], (vwLength, c, w))
        c += 1
        paths[dir][w] = paths[dir][v] & @[w]
        if w in seen[0] and w in seen[1]:
          var totalDist = seen[0][w] + seen[1][w]
          if len(finalPath) == 0 or finalDist > totalDist:
            finalDist = totalDist
            var revPath = reversed(paths[1][w])
            finalPath = paths[0][w]
            for i in 1..<len(revPath):
              finalPath = finalPath & revPath[i]
  raise newNNNoPath(fmt"no path between source {source} and target {target}")

proc johnson*(
  G: Graph,
  weight: TableRef[Edge, float]
): Table[Node, Table[Node, seq[Node]]] =
  if len(G.edges()) != len(weight):
    raise newNNError("all edge weights are needed")
  var dist = newTable[Node, float]()
  for v in G.nodes():
    dist[v] = 0.0
  var pred = newTable[Node, seq[Node]]()
  for v in G.nodes():
    pred[v] = @[]
  var distBellman = bellmanFord(G, G.nodes(), weight=weight, pred=pred, dist=dist)

  var newWeight = newTable[Edge, float]()
  for (u, v) in weight.keys():
    newWeight[(u, v)] = weight.getOrDefault((u, v), weight.getOrDefault((v, u), Inf)) + distBellman[u] - distBellman[v]
  let distPath = proc(v: Node): TableRef[Node, seq[Node]] =
    var paths = newTable[Node, seq[Node]]()
    paths[v] = @[v]
    discard dijkstra(G, v, weight=newWeight, paths=paths)
    return paths

  var ret = initTable[Node, Table[Node, seq[Node]]]()
  for v in G.nodes():
    ret[v] = distPath(v)[]
  return ret
proc johnson*(
  DG: DiGraph,
  weight: TableRef[Edge, float]
): Table[Node, Table[Node, seq[Node]]] =
  if len(DG.edges()) != len(weight):
    raise newNNError("all edge weights are needed")
  var dist = newTable[Node, float]()
  for v in DG.nodes():
    dist[v] = 0.0
  var pred = newTable[Node, seq[Node]]()
  for v in DG.nodes():
    pred[v] = @[]
  var distBellman = bellmanFord(DG, DG.nodes(), weight=weight, pred=pred, dist=dist)

  var newWeight = newTable[Edge, float]()
  for (u, v) in weight.keys():
    newWeight[(u, v)] = weight.getOrDefault((u, v), Inf) + distBellman[u] - distBellman[v]
  let distPath = proc(v: Node): TableRef[Node, seq[Node]] =
    var paths = newTable[Node, seq[Node]]()
    paths[v] = @[v]
    discard dijkstra(DG, v, weight=newWeight, paths=paths)
    return paths

  var ret = initTable[Node, Table[Node, seq[Node]]]()
  for v in DG.nodes():
    ret[v] = distPath(v)[]
  return ret

proc floydWarshallPredecessorAndDistance*(
  G: Graph,
  weight: TableRef[Edge, float]
): tuple[predecessor: Table[Node, Table[Node, Node]], distance: Table[Node, Table[Node, float]]] =
  if len(G.edges()) != len(weight):
    raise newNNError("all edge weights are needed")
  var dist = initTable[Node, Table[Node, float]]()
  for u in G.nodes():
    dist[u] = initTable[Node, float]()
    for v in G.nodes():
      dist[u][v] = Inf
  for u in G.nodes():
    dist[u][u] = 0.0

  var pred = initTable[Node, Table[Node, Node]]()
  for n in G.nodes():
    pred[n] = initTable[Node, Node]()
  for (u, v) in G.edges():
    var eWeight = weight.getOrDefault((u, v), weight.getOrDefault((v, u), Inf))
    dist[u][v] = min(eWeight, dist[u][v])
    pred[u][v] = u
    dist[v][u] = min(eWeight, dist[v][u])
    pred[v][u] = v
  for w in G.nodes():
    for u in G.nodes():
      for v in G.nodes():
        var d = dist[u][w] + dist[w][v]
        if dist[u][v] > d:
          dist[u][v] = d
          pred[u][v] = pred[w][v]
  var retPred = initTable[Node, Table[Node, Node]]()
  var retDist = initTable[Node, Table[Node, float]]()
  for k1 in pred.keys():
    if len(pred[k1].keys().toSeq()) != 0:
      retPred[k1] = pred[k1]
  for k1 in dist.keys():
    if len(dist[k1].keys().toSeq()) != 0:
      retDist[k1] = dist[k1]
  return (retPred, retDist)
proc floydWarshallPredecessorAndDistance*(
  DG: DiGraph,
  weight: TableRef[Edge, float]
): tuple[predecessor: Table[Node, Table[Node, Node]], distance: Table[Node, Table[Node, float]]] =
  if len(DG.edges()) != len(weight):
    raise newNNError("all edge weights are needed")
  var dist = initTable[Node, Table[Node, float]]()
  for u in DG.nodes():
    dist[u] = initTable[Node, float]()
    for v in DG.nodes():
      dist[u][v] = Inf
  for u in DG.nodes():
    dist[u][u] = 0.0

  var pred = initTable[Node, Table[Node, Node]]()
  for n in DG.nodes():
    pred[n] = initTable[Node, Node]()
  for (u, v) in DG.edges():
    var eWeight = weight.getOrDefault((u, v), Inf)
    dist[u][v] = min(eWeight, dist[u][v])
    pred[u][v] = u
  for w in DG.nodes():
    for u in DG.nodes():
      for v in DG.nodes():
        var d = dist[u][w] + dist[w][v]
        if dist[u][v] > d:
          dist[u][v] = d
          pred[u][v] = pred[w][v]
  var retPred = initTable[Node, Table[Node, Node]]()
  var retDist = initTable[Node, Table[Node, float]]()
  for k1 in pred.keys():
    if len(pred[k1].keys().toSeq()) != 0:
      retPred[k1] = pred[k1]
  for k1 in dist.keys():
    if len(dist[k1].keys().toSeq()) != 0:
      retDist[k1] = dist[k1]
  return (retPred, retDist)

proc reconstructPath*(
  source: Node,
  target: Node,
  predecessors: Table[Node, Table[Node, Node]]
): seq[Node] =
  var path: seq[Node] = @[]
  if source == target:
    return path
  var prev = predecessors[source]
  var curr = prev[target]
  path = @[target, curr]
  while curr != source:
    curr = prev[curr]
    path.add(curr)
  return reversed(path)

proc floydWarshall*(
  G: Graph,
  weight: TableRef[Edge, float]
): Table[Node, Table[Node, float]] =
  return floydWarshallPredecessorAndDistance(G, weight=weight).distance
proc floydWarshall*(
  DG: DiGraph,
  weight: TableRef[Edge, float]
): Table[Node, Table[Node, float]] =
  return floydWarshallPredecessorAndDistance(DG, weight=weight).distance

proc astarPath*(
  G: Graph,
  source: Node,
  target: Node,
  weight: TableRef[Edge, float],
  heuristic: proc(n1, n2: Node): float = nil,
): seq[Node] =
  if len(G.edges()) != len(weight):
    raise newNNError("weight of all edges are needed")
  if source notin G.nodesSet():
    raise newNNNodeNotFound(source)
  if target notin G.nodesSet():
    raise newNNNodeNotFound(target)

  var heuristicUsing = heuristic
  if heuristic == nil:
    heuristicUsing = proc(u, v: Node): float = return 0.0

  var c = -1
  var queue = initHeapQueue[tuple[priority: float, c: int, node: Node, cost: float, parent: Node]]()
  c += 1
  push(queue, (0.0, c, source, 0.0, None))

  var enqueued = initTable[Node, tuple[cost: float, heur: float]]()
  var explored = initTable[Node, Node]()

  while len(queue) != 0:
    var (_, _, curNode, dist, parent) = pop(queue)
    if curNode == target:
      var path = @[curNode]
      var node = parent
      while node != None:
        path.add(node)
        node = explored[node]
      path.reverse()
      return path
    if curNode in explored:
      if explored[curNode] == None:
        continue
      var (qCost, _) = enqueued[curNode]
      if qCost < dist:
        continue
    explored[curNode] = parent
    for neighbor in G.neighbors(curNode):
      var w = weight.getOrDefault((curNode, neighbor), weight.getOrDefault((curNode, neighbor), Inf))
      var nCost = dist + w
      var h: float
      var qCost: float
      if neighbor in enqueued:
        (qCost, h) = enqueued[neighbor]
        if qCost <= nCost:
          continue
      else:
        h = heuristicUsing(neighbor, target)
      enqueued[neighbor] = (nCost, h)
      c += 1
      push(queue, (nCost + h, c, neighbor, nCost, curNode))
  raise newNNNoPath(fmt"target {target} not reachable from source {source}")
proc astarPath*(
  DG: DiGraph,
  source: Node,
  target: Node,
  weight: TableRef[Edge, float],
  heuristic: proc(n1, n2: Node): float = nil,
): seq[Node] =
  if len(DG.edges()) != len(weight):
    raise newNNError("weight of all edges are needed")
  if source notin DG.nodesSet():
    raise newNNNodeNotFound(source)
  if target notin DG.nodesSet():
    raise newNNNodeNotFound(target)

  var heuristicUsing = heuristic
  if heuristic == nil:
    heuristicUsing = proc(u, v: Node): float = return 0.0

  var c = -1
  var queue = initHeapQueue[tuple[priority: float, c: int, node: Node, cost: float, parent: Node]]()
  c += 1
  push(queue, (0.0, c, source, 0.0, None))

  var enqueued = initTable[Node, tuple[cost: float, heur: float]]()
  var explored = initTable[Node, Node]()

  while len(queue) != 0:
    var (_, _, curNode, dist, parent) = pop(queue)
    if curNode == target:
      var path = @[curNode]
      var node = parent
      while node != None:
        path.add(node)
        node = explored[node]
      path.reverse()
      return path
    if curNode in explored:
      if explored[curNode] == None:
        continue
      var (qCost, _) = enqueued[curNode]
      if qCost < dist:
        continue
    explored[curNode] = parent
    for neighbor in DG.successors(curNode):
      var w = weight.getOrDefault((curNode, neighbor), Inf)
      var nCost = dist + w
      var h: float
      var qCost: float
      if neighbor in enqueued:
        (qCost, h) = enqueued[neighbor]
        if qCost <= nCost:
          continue
      else:
        h = heuristicUsing(neighbor, target)
      enqueued[neighbor] = (nCost, h)
      c += 1
      push(queue, (nCost + h, c, neighbor, nCost, curNode))
  raise newNNNoPath(fmt"target {target} not reachable from source {source}")

proc astarPathLength*(
  G: Graph,
  source: Node,
  target: Node,
  weight: TableRef[Edge, float],
  heuristic: proc(n1, n2: Node): float = nil,
): float =
  if len(G.edges()) != len(weight):
    raise newNNError("weight of all edges are needed")
  if source notin G.nodesSet():
    raise newNNNodeNotFound(source)
  if target notin G.nodesSet():
    raise newNNNodeNotFound(target)
  var path = astarPath(G, source, target, weight, heuristic)
  var ret = 0.0
  for i in 0..<(len(path) - 1):
    ret += weight.getOrDefault((path[i], path[i + 1]), weight.getOrDefault((path[i + 1], path[i]), Inf))
  return ret
proc astarPathLength*(
  DG: DiGraph,
  source: Node,
  target: Node,
  weight: TableRef[Edge, float],
  heuristic: proc(n1, n2: Node): float = nil,
): float =
  if len(DG.edges()) != len(weight):
    raise newNNError("weight of all edges are needed")
  if source notin DG.nodesSet():
    raise newNNNodeNotFound(source)
  if target notin DG.nodesSet():
    raise newNNNodeNotFound(target)
  var path = astarPath(DG, source, target, weight, heuristic)
  var ret = 0.0
  for i in 0..<(len(path) - 1):
    ret += weight.getOrDefault((path[i], path[i + 1]), weight.getOrDefault((path[i + 1], path[i]), Inf))
  return ret

proc shortestPath*(
  G: Graph,
  source: Node = None,
  target: Node = None,
  weight: TableRef[Edge, float] = nil,
  methodName: string = "dijkstra"
): Table[Node, Table[Node, seq[Node]]] =
  var methodNameUsing = methodName
  if methodNameUsing != "dijkstra" and methodNameUsing != "bellman-ford":
    raise newNNError(fmt"method not supported: {methodNameUsing}")
  if weight == nil:
    methodNameUsing = "unweighted"

  var paths: Table[Node, Table[Node, seq[Node]]]
  if source == None:
    if target == None: # all pairs
      if methodNameUsing == "unweighted":
        for (source, path) in allPairsShortestPath(G):
          paths[source] = path
      elif methodNameUsing == "dijkstra":
        for (source, path) in allPairsDijkstraPath(G, weight=weight):
          paths[source] = path
      elif methodNameUsing == "bellman-ford":
        for (source, path) in allPairsBellmanFordPath(G, weight=weight):
          paths[source] = path

    else: # single target
      if methodNameUsing == "unweighted":
        paths[target] = singleSourceShortestPath(G, source=target)
      elif methodNameUsing == "dijkstra":
        paths[target] = singleSourceDijkstraPath(G, target, weight=weight)
      elif methodNameUsing == "bellman-ford":
        paths[target] = singleSourceBellmanFordPath(G, target, weight=weight)
      for v in paths[target].keys():
        reverse(paths[target][v])

  else:
    if target == None: # single source
      if methodNameUsing == "unweighted":
        paths[source] = singleSourceShortestPath(G, source)
      elif methodNameUsing == "dijkstra":
        paths[source] = singleSourceDijkstraPath(G, source, weight=weight)
      elif methodNameUsing == "bellman-ford":
        paths[source] = singleSourceBellmanFordPath(G, source, weight=weight)

    else: # single source and single target
      if methodNameUsing == "unweighted":
        paths[source] = {target: bidirectionalShortestPath(G, source, target)}.toTable()
      elif methodNameUsing == "dijkstra":
        paths[source] = {target: bidirectionalDijkstra(G, source, target, weight=weight).path}.toTable()
      elif methodNameUsing == "bellman-ford":
        paths[source] = {target: bellmanFordPath(G, source, target, weight=weight)}.toTable()
  return paths
proc shortestPath*(
  DG: DiGraph,
  source: Node = None,
  target: Node = None,
  weight: TableRef[Edge, float] = nil,
  methodName: string = "dijkstra"
): Table[Node, Table[Node, seq[Node]]] =
  var methodNameUsing = methodName
  if methodNameUsing != "dijkstra" and methodNameUsing != "bellman-ford":
    raise newNNError(fmt"method not supported: {methodNameUsing}")
  if weight == nil:
    methodNameUsing = "unweighted"

  var paths: Table[Node, Table[Node, seq[Node]]]
  if source == None:
    if target == None: # all pairs
      if methodNameUsing == "unweighted":
        for (source, path) in allPairsShortestPath(DG):
          paths[source] = path
      elif methodNameUsing == "dijkstra":
        for (source, path) in allPairsDijkstraPath(DG, weight=weight):
          paths[source] = path
      elif methodNameUsing == "bellman-ford":
        for (source, path) in allPairsBellmanFordPath(DG, weight=weight):
          paths[source] = path

    else: # single target
      var RDG = DG.reverse()
      if methodNameUsing == "unweighted":
        paths[target] = singleSourceShortestPath(RDG, source=target)
      elif methodNameUsing == "dijkstra":
        paths[target] = singleSourceDijkstraPath(RDG, target, weight=weight)
      elif methodNameUsing == "bellman-ford":
        paths[target] = singleSourceBellmanFordPath(RDG, target, weight=weight)
      for v in paths[target].keys():
        reverse(paths[target][v])

  else:
    if target == None: # single source
      if methodNameUsing == "unweighted":
        paths[source] = singleSourceShortestPath(DG, source)
      elif methodNameUsing == "dijkstra":
        paths[source] = singleSourceDijkstraPath(DG, source, weight=weight)
      elif methodNameUsing == "bellman-ford":
        paths[source] = singleSourceBellmanFordPath(DG, source, weight=weight)

    else: # single source and single target
      if methodNameUsing == "unweighted":
        paths[source] = {target: bidirectionalShortestPath(DG, source, target)}.toTable()
      elif methodNameUsing == "dijkstra":
        paths[source] = {target: bidirectionalDijkstra(DG, source, target, weight=weight).path}.toTable()
      elif methodNameUsing == "bellman-ford":
        paths[source] = {target: bellmanFordPath(DG, source, target, weight=weight)}.toTable()
  return paths

proc shortestPathLength*(
  G: Graph,
  source: Node = None,
  target: Node = None,
  weight: TableRef[Edge, float] = nil,
  methodName: string = "dijkstra"
): Table[Node, Table[Node, float]] =
  var methodNameUsing = methodName
  if methodNameUsing != "dijkstra" and methodNameUsing != "bellman-ford":
    raise newNNError(fmt"method not supported: {methodNameUsing}")
  if weight == nil:
    methodNameUsing = "unweighted"

  var paths: Table[Node, Table[Node, float]]
  if source == None:
    if target == None: # all pairs
      if methodNameUsing == "unweighted":
        for (source, path) in allPairsShortestPathLength(G):
          paths[source] = initTable[Node, float]()
          for (k, v) in path.pairs():
            paths[source][k] = v.float
      elif methodNameUsing == "dijkstra":
        for (source, path) in allPairsDijkstraPathLength(G, weight=weight):
          paths[source] = path
      elif methodNameUsing == "bellman-ford":
        for (source, path) in allPairsBellmanFordPathLength(G, weight=weight):
          paths[source] = path

    else: # single target
      if methodNameUsing == "unweighted":
        for (k, v) in singleSourceShortestPathLength(G, source=target).pairs():
          paths[k] = {target: v.float}.toTable()
      elif methodNameUsing == "dijkstra":
        for (k, v) in singleSourceDijkstraPathLength(G, target, weight=weight).pairs():
          paths[k] = {target: v.float}.toTable()
      elif methodNameUsing == "bellman-ford":
        for (k, v) in singleSourceBellmanFordPathLength(G, target, weight=weight).pairs():
          paths[k] = {target: v.float}.toTable()

  else:
    if target == None: # single source
      if methodNameUsing == "unweighted":
        paths[source] = initTable[Node, float]()
        for (k, v) in singleSourceShortestPathLength(G, source).pairs():
          paths[source][k] = v.float
      elif methodNameUsing == "dijkstra":
        paths[source] = singleSourceDijkstraPathLength(G, source, weight=weight)
      elif methodNameUsing == "bellman-ford":
        paths[source] = singleSourceBellmanFordPathLength(G, source, weight=weight)

    else: # single source and single target
      if methodNameUsing == "unweighted":
        paths[source] = {target: (len(bidirectionalShortestPath(G, source, target)) - 1).float}.toTable()
      elif methodNameUsing == "dijkstra":
        paths[source] = {target: dijkstraPathLength(G, source, target, weight=weight)}.toTable()
      elif methodNameUsing == "bellman-ford":
        paths[source] = {target: bellmanFordPathLength(G, source, target, weight=weight)}.toTable()
  return paths
proc shortestPathLength*(
  DG: DiGraph,
  source: Node = None,
  target: Node = None,
  weight: TableRef[Edge, float] = nil,
  methodName: string = "dijkstra"
): Table[Node, Table[Node, float]] =
  var methodNameUsing = methodName
  if methodNameUsing != "dijkstra" and methodNameUsing != "bellman-ford":
    raise newNNError(fmt"method not supported: {methodNameUsing}")
  if weight == nil:
    methodNameUsing = "unweighted"

  var paths: Table[Node, Table[Node, float]]
  if source == None:
    if target == None: # all pairs
      if methodNameUsing == "unweighted":
        for (source, path) in allPairsShortestPathLength(DG):
          paths[source] = initTable[Node, float]()
          for (k, v) in path.pairs():
            paths[source][k] = v.float
      elif methodNameUsing == "dijkstra":
        for (source, path) in allPairsDijkstraPathLength(DG, weight=weight):
          paths[source] = path
      elif methodNameUsing == "bellman-ford":
        for (source, path) in allPairsBellmanFordPathLength(DG, weight=weight):
          paths[source] = path

    else: # single target
      var RDG = DG.reverse()
      if methodNameUsing == "unweighted":
        for (k, v) in singleSourceShortestPathLength(RDG, source=target).pairs():
          paths[k] = {target: v.float}.toTable()
      elif methodNameUsing == "dijkstra":
        for (k, v) in singleSourceDijkstraPathLength(RDG, target, weight=weight).pairs():
          paths[k] = {target: v.float}.toTable()
      elif methodNameUsing == "bellman-ford":
        for (k, v) in singleSourceBellmanFordPathLength(RDG, target, weight=weight).pairs():
          paths[k] = {target: v.float}.toTable()

  else:
    if target == None: # single source
      if methodNameUsing == "unweighted":
        paths[source] = initTable[Node, float]()
        for (k, v) in singleSourceShortestPathLength(DG, source).pairs():
          paths[source][k] = v.float
      elif methodNameUsing == "dijkstra":
        paths[source] = singleSourceDijkstraPathLength(DG, source, weight=weight)
      elif methodNameUsing == "bellman-ford":
        paths[source] = singleSourceBellmanFordPathLength(DG, source, weight=weight)

    else: # single source and single target
      if methodNameUsing == "unweighted":
        paths[source] = {target: (len(bidirectionalShortestPath(DG, source, target)) - 1).float}.toTable()
      elif methodNameUsing == "dijkstra":
        paths[source] = {target: dijkstraPathLength(DG, source, target, weight=weight)}.toTable()
      elif methodNameUsing == "bellman-ford":
        paths[source] = {target: bellmanFordPathLength(DG, source, target, weight=weight)}.toTable()
  return paths

proc averageShortestPathLength*(
  G: Graph,
  weight: TableRef[Edge, float] = nil,
  methodName: string = ""
): float =
  let singleSourceMethods = @["unweighted", "dijkstra", "bellman-ford"].toHashSet()
  let allPairsMethods = @["floyd-warshall"].toHashSet()
  let supportedMethods = singleSourceMethods + allPairsMethods
  var methodNameUsing = methodName
  if methodName == "":
    if weight == nil:
      methodNameUsing = "unweighted"
    else:
      methodNameUsing = "dijkstra"
  if methodNameUsing notin supportedMethods:
    raise newNNError(fmt"method not supported: {methodNameUsing}")
  let N = len(G)
  if N == 0:
    raise newNNPointlessConcept("null graph has no paths, thus no average shortest path length")
  if N == 1:
    return 0.0
  let pathLength = proc(v: Node): Table[Node, float] =
    if methodNameUsing == "unweighted":
      var ret = initTable[Node, float]()
      for (n, length) in singleSourceShortestPathLength(G, v).pairs():
        ret[n] = length.float
      return ret
    elif methodNameUsing == "dijkstra":
      return singleSourceDijkstraPathLength(G, v, weight=weight)
    elif methodNameUsing == "bellman-ford":
      return singleSourceBellmanFordPathLength(G, v, weight=weight)
  var s = 0.0
  if methodNameUsing in singleSourceMethods:
    for u in G.nodes():
      for l in pathLength(u).values():
        s += l
  else:
    if methodNameUsing == "floyd-warshall":
      var allPairs = floydWarshall(G, weight=weight)
      for t in allPairs.values():
        for v in t.values():
          s += v
  return s / (N * (N - 1)).float
proc averageShortestPathLength*(
  DG: DiGraph,
  weight: TableRef[Edge, float] = nil,
  methodName: string = ""
): float =
  let singleSourceMethods = @["unweighted", "dijkstra", "bellman-ford"].toHashSet()
  let allPairsMethods = @["floyd-warshall"].toHashSet()
  let supportedMethods = singleSourceMethods + allPairsMethods
  var methodNameUsing = methodName
  if methodName == "":
    if weight == nil:
      methodNameUsing = "unweighted"
    else:
      methodNameUsing = "dijkstra"
  if methodNameUsing notin supportedMethods:
    raise newNNError(fmt"method not supported: {methodNameUsing}")
  let N = len(DG)
  if N == 0:
    raise newNNPointlessConcept("null graph has no paths, thus no average shortest path length")
  if N == 1:
    return 0.0
  let pathLength = proc(v: Node): Table[Node, float] =
    if methodNameUsing == "unweighted":
      var ret = initTable[Node, float]()
      for (n, length) in singleSourceShortestPathLength(DG, v).pairs():
        ret[n] = length.float
      return ret
    elif methodNameUsing == "dijkstra":
      return singleSourceDijkstraPathLength(DG, v, weight=weight)
    elif methodNameUsing == "bellman-ford":
      return singleSourceBellmanFordPathLength(DG, v, weight=weight)
  var s = 0.0
  if methodNameUsing in singleSourceMethods:
    for u in DG.nodes():
      for l in pathLength(u).values():
        s += l
  else:
    if methodNameUsing == "floyd-warshall":
      var allPairs = floydWarshall(DG, weight=weight)
      for t in allPairs.values():
        for v in t.values():
          s += v
  return s / (N * (N - 1)).float

iterator allShortestPaths*(
  G: Graph,
  source: Node,
  target: Node,
  weight: TableRef[Edge, float] = nil,
  methodName: string = "dijkstra"
): seq[Node] =
  var methodNameUsing: string
  if weight == nil:
    methodNameUsing = "unweighted"
  else:
    methodNameUsing = methodName
  var pred: Table[Node, seq[Node]]
  var dist: Table[Node, float]
  if methodNameUsing == "unweighted":
    pred = predecessor(G, source)
  elif methodNameUsing == "dijkstra":
    (pred, dist) = dijkstraPredecessorAndDistance(G, source, weight=weight)
  elif methodNameUsing == "bellman-ford":
    (pred, dist) = bellmanFordPredecessorAndDistance(G, source, weight=weight)
  else:
    raise newNNError(fmt"method not suppoerted: {methodNameUsing}")
  var predRef = newTable[Node, seq[Node]]()
  for (k, v) in pred.pairs():
    predRef[k] = v
  for path in buildPathsFromPredecessors(@[source].toHashSet(), target, predRef):
    yield path
iterator allShortestPaths*(
  DG: DiGraph,
  source: Node,
  target: Node,
  weight: TableRef[Edge, float] = nil,
  methodName: string = "dijkstra"
): seq[Node] =
  var methodNameUsing: string
  if weight == nil:
    methodNameUsing = "unweighted"
  else:
    methodNameUsing = methodName
  var pred: Table[Node, seq[Node]]
  var dist: Table[Node, float]
  if methodNameUsing == "unweighted":
    pred = predecessor(DG, source)
  elif methodNameUsing == "dijkstra":
    (pred, dist) = dijkstraPredecessorAndDistance(DG, source, weight=weight)
  elif methodNameUsing == "bellman-ford":
    (pred, dist) = bellmanFordPredecessorAndDistance(DG, source, weight=weight)
  else:
    raise newNNError(fmt"method not suppoerted: {methodNameUsing}")
  var predRef = newTable[Node, seq[Node]]()
  for (k, v) in pred.pairs():
    predRef[k] = v
  for path in buildPathsFromPredecessors(@[source].toHashSet(), target, predRef):
    yield path

proc hasPath*(G: Graph, source: Node, target: Node): bool =
  try:
    discard shortestPath(G, source, target)
  except NNNoPath:
    return false
  return true

# -------------------------------------------------------------------
# TODO:
# Similarity Measures
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Simple Paths
# -------------------------------------------------------------------

proc isSimplePath*(G: Graph, nodes: seq[Node]): bool =
  if len(nodes) == 0:
    return false
  if len(nodes) == 1:
    return nodes[0] in G
  var checks: seq[bool] = @[]
  for (u, v) in pairwise(nodes):
    checks.add(v in G.neighborsSet(u))
  return len(nodes.toHashSet()) == len(nodes) and all(checks, proc(b: bool): bool = return b)
proc isSimplePath*(DG: DiGraph, nodes: seq[Node]): bool =
  if len(nodes) == 0:
    return false
  if len(nodes) == 1:
    return nodes[0] in DG
  var checks: seq[bool] = @[]
  for (u, v) in pairwise(nodes):
    checks.add(v in DG.successorsSet(u))
  return len(nodes.toHashSet()) == len(nodes) and all(checks, proc(b: bool): bool = return b)

# -------------------------------------------------------------------
# TODO:
# Small World
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# s-metric
# -------------------------------------------------------------------

proc sMetric*(G: Graph, normalized: bool = false): float =
  if normalized:
    raise newNNError("normalization is not implemented yet")
  var s = 0.0
  for (u, v) in G.edges():
    s += (G.degree(u) * G.degree(v)).float
  return s
proc sMetric*(DG: DiGraph, normalized: bool = false): float =
  if normalized:
    raise newNNError("normalization is not implemented yet")
  var s = 0.0
  for (u, v) in DG.edges():
    s += (DG.degree(u) * DG.degree(v)).float
  return s

# -------------------------------------------------------------------
# TODO:
# Sparsifiers
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Structual Holes
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Summarization
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Swap
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Threshold Graphs
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Tournament
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Traversal
# -------------------------------------------------------------------

iterator dfsEdges*(
  G: Graph,
  source: Node = None,
  depthLimit: int = -1,
): Edge =
  var nodes: seq[Node]
  if source == None:
    nodes = G.nodes()
  else:
    nodes = @[source]
  var visited = initHashSet[Node]()

  var depthLimitUsing = depthLimit
  if depthLimit == -1:
    depthLimitUsing = len(G)

  for start in nodes:
    if start in visited:
      continue
    visited.incl(start)
    var stack = initDeque[tuple[parent: Node, depthNow: int, children: seq[Node], idx: int]]()
    stack.addFirst((start, depthLimitUsing, G.neighbors(start), 0))
    while len(stack) != 0:
      var (parent, depthNow, children, idx) = stack.peekLast()
      if idx < len(children):
        discard stack.popLast()
        stack.addLast((parent, depthNow, children, idx+1))
        var child = children[idx]
        if child notin visited:
          yield (parent, child)
          visited.incl(child)
          if 1 < depthNow:
            stack.addLast((child, depthNow - 1, G.neighbors(child), 0))
      else:
        discard stack.popLast()
iterator dfsEdges*(
  DG: DiGraph,
  source: Node = None,
  depthLimit: int = -1,
): Edge =
  var nodes: seq[Node]
  if source == None:
    nodes = DG.nodes()
  else:
    nodes = @[source]
  var visited = initHashSet[Node]()

  var depthLimitUsing = depthLimit
  if depthLimit == -1:
    depthLimitUsing = len(DG)

  for start in nodes:
    if start in visited:
      continue
    visited.incl(start)
    var stack = initDeque[tuple[parent: Node, depthNow: int, children: seq[Node], idx: int]]()
    stack.addFirst((start, depthLimitUsing, DG.neighbors(start), 0))
    while len(stack) != 0:
      var (parent, depthNow, children, idx) = stack.peekLast()
      if idx < len(children):
        discard stack.popLast()
        stack.addLast((parent, depthNow, children, idx+1))
        var child = children[idx]
        if child notin visited:
          yield (parent, child)
          visited.incl(child)
          if 1 < depthNow:
            stack.addLast((child, depthNow - 1, DG.successors(child), 0))
      else:
        discard stack.popLast()

proc dfsTree*(
  G: Graph,
  source: Node = None,
  depthLimit: int = -1
): DiGraph =
  let T = newDiGraph()
  if source == None:
    T.addNodesFrom(G.nodes())
  else:
    T.addNode(source)
  for edge in G.dfsEdges(source, depthLimit):
    T.addEdge(edge)
  return T
proc dfsTree*(
  DG: DiGraph,
  source: Node = None,
  depthLimit: int = -1
): DiGraph =
  let T = newDiGraph()
  if source == None:
    T.addNodesFrom(DG.nodes())
  else:
    T.addNode(source)
  for edge in DG.dfsEdges(source, depthLimit):
    T.addEdge(edge)
  return T

proc dfsPredecessors*(
  G: Graph,
  source: Node = None,
  depthLimit: int = -1
): Table[Node, Node] =
  var ret = initTable[Node, Node]()
  for (s, t) in dfsEdges(G, source, depthLimit):
    ret[t] = s
  return ret
proc dfsPredecessors*(
  DG: DiGraph,
  source: Node = None,
  depthLimit: int = -1
): Table[Node, Node] =
  var ret = initTable[Node, Node]()
  for (s, t) in dfsEdges(DG, source, depthLimit):
    ret[t] = s
  return ret

proc dfsSuccessors*(
  G: Graph,
  source: Node = None,
  depthLimit: int = -1
): Table[Node, seq[Node]] =
  var ret = initTable[Node, seq[Node]]()
  for (s, t) in dfsEdges(G, source, depthLimit):
    if s notin ret:
      ret[s] = @[]
    ret[s].add(t)
  return ret
proc dfsSuccessors*(
  DG: DiGraph,
  source: Node = None,
  depthLimit: int = -1
): Table[Node, seq[Node]] =
  var ret = initTable[Node, seq[Node]]()
  for (s, t) in dfsEdges(DG, source, depthLimit):
    if s notin ret:
      ret[s] = @[]
    ret[s].add(t)
  return ret

iterator dfsLabeledEdges*(
  G: Graph,
  source: Node = None,
  depthLimit: int = -1
): tuple[u, v: Node, direction: string] =
  var nodes: seq[Node]
  if source == None:
    nodes = G.nodes()
  else:
    nodes = @[source]
  var visited = initHashSet[Node]()

  var depthLimitUsing = depthLimit
  if depthLimit == -1:
    depthLimitUsing = len(G)

  for start in nodes:
    if start in visited:
      continue
    yield (start, start, "forward")
    visited.incl(start)
    var stack = initDeque[tuple[parent: Node, depthNow: int, children: seq[Node], idx: int]]()
    stack.addFirst((start, depthLimitUsing, G.neighbors(start), 0))
    while len(stack) != 0:
      var (parent, depthNow, children, idx) = stack.peekLast()
      if idx < len(children):
        discard stack.popLast()
        stack.addLast((parent, depthNow, children, idx+1))
        var child = children[idx]
        if child in visited:
          yield (parent, child, "nontree")
        else:
          yield (parent, child, "forward")
          visited.incl(child)
          if 1 < depthNow:
            stack.addLast((child, depthNow - 1, G.neighbors(child), 0))
      else:
        discard stack.popLast()
        if len(stack) != 0:
          yield (stack.peekLast().parent, parent, "reverse")
    yield (start, start, "reverse")
iterator dfsLabeledEdges*(
  DG: DiGraph,
  source: Node = None,
  depthLimit: int = -1
): tuple[u, v: Node, direction: string] =
  var nodes: seq[Node]
  if source == None:
    nodes = DG.nodes()
  else:
    nodes = @[source]
  var visited = initHashSet[Node]()

  var depthLimitUsing = depthLimit
  if depthLimit == -1:
    depthLimitUsing = len(DG)

  for start in nodes:
    if start in visited:
      continue
    yield (start, start, "forward")
    visited.incl(start)
    var stack = initDeque[tuple[parent: Node, depthNow: int, children: seq[Node], idx: int]]()
    stack.addFirst((start, depthLimitUsing, DG.successors(start), 0))
    while len(stack) != 0:
      var (parent, depthNow, children, idx) = stack.peekLast()
      if idx < len(children):
        discard stack.popLast()
        stack.addLast((parent, depthNow, children, idx+1))
        var child = children[idx]
        if child in visited:
          yield (parent, child, "nontree")
        else:
          yield (parent, child, "forward")
          visited.incl(child)
          if 1 < depthNow:
            stack.addLast((child, depthNow - 1, DG.successors(child), 0))
      else:
        discard stack.popLast()
        if len(stack) != 0:
          yield (stack.peekLast().parent, parent, "reverse")
    yield (start, start, "reverse")

proc dfsPostOrderNodes*(
  G: Graph,
  source: Node = None,
  depthLimit: int = -1
): seq[Node] =
  var ret: seq[Node] = @[]
  for (u, v, d) in dfsLabeledEdges(G, source=source, depthLimit=depthLimit):
    if d == "reverse":
      ret.add(v)
  return ret
proc dfsPostOrderNodes*(
  DG: DiGraph,
  source: Node = None,
  depthLimit: int = -1
): seq[Node] =
  var ret: seq[Node] = @[]
  for (u, v, d) in dfsLabeledEdges(DG, source=source, depthLimit=depthLimit):
    if d == "reverse":
      ret.add(v)
  return ret

proc dfsPreOrderNodes*(
  G: Graph,
  source: Node = None,
  depthLimit: int = -1
): seq[Node] =
  var ret: seq[Node] = @[]
  for (u, v, d) in dfsLabeledEdges(G, source=source, depthLimit=depthLimit):
    if d == "forward":
      ret.add(v)
  return ret
proc dfsPreOrderNodes*(
  DG: DiGraph,
  source: Node = None,
  depthLimit: int = -1
): seq[Node] =
  var ret: seq[Node] = @[]
  for (u, v, d) in dfsLabeledEdges(DG, source=source, depthLimit=depthLimit):
    if d == "forward":
      ret.add(v)
  return ret

iterator genericBfsEdges(
  G: Graph,
  source: Node,
  neighbors: proc(node: Node): iterator: Node = nil,
  depthLimit: int = -1,
  sortNeighbors: proc(nodes: seq[Node]): iterator: Node = nil
): Edge =
  var neighborsUsing: proc(node: Node): iterator: Node = neighbors

  if neighbors == nil:
    neighborsUsing = proc(node: Node): iterator: Node =
      return G.neighborsIterator(node)

  if sortNeighbors != nil:
    neighborsUsing =
      proc(node: Node): iterator: Node =
        return iterator: Node =
          for neighbor in sortNeighbors(neighbors(node).toSeq()):
            yield neighbor

  var visited = initHashSet[Node]()
  visited.incl(source)

  var depthLimitUsing = depthLimit
  if depthLimit == -1:
    depthLimitUsing = len(G)

  var queue = initDeque[tuple[node: Node, depthLimit: int, neighborIterator: iterator: Node]]()
  queue.addLast(
    (source, depthLimitUsing, neighborsUsing(source))
  )

  while len(queue) != 0:
    let (parent, depthNow, children) = queue.popFirst()
    for child in children:
      if child notin visited:
        yield (parent, child)
        visited.incl(child)
        if 1 < depthNow:
          queue.addLast((child, depthNow - 1, neighborsUsing(child)))
iterator genericBfsEdges(
  DG: DiGraph,
  source: Node,
  successors: proc(node: Node): iterator: Node = nil,
  depthLimit: int = -1,
  sortSuccessors: proc(nodes: seq[Node]): iterator: Node = nil
): Edge =
  var successorsUsing: proc(node: Node): iterator: Node = successors

  if successors == nil:
    successorsUsing = proc(node: Node): iterator: Node =
      return DG.successorsIterator(node)

  if sortSuccessors != nil:
    successorsUsing =
      proc(node: Node): iterator: Node =
        return iterator: Node =
          for neighbor in sortSuccessors(successors(node).toSeq()):
            yield neighbor

  var visited = initHashSet[Node]()
  visited.incl(source)

  var depthLimitUsing = depthLimit
  if depthLimit == -1:
    depthLimitUsing = len(DG)

  var queue = initDeque[tuple[node: Node, depthLimit: int, neighborIterator: iterator: Node]]()
  queue.addLast(
    (source, depthLimitUsing, successorsUsing(source))
  )

  while len(queue) != 0:
    let (parent, depthNow, children) = queue.popFirst()
    for child in children:
      if child notin visited:
        yield (parent, child)
        visited.incl(child)
        if 1 < depthNow:
          queue.addLast((child, depthNow - 1, successorsUsing(child)))

iterator bfsEdges*(
  G: Graph,
  source: Node,
  reverse: bool = false,
  depthLimit: int = -1,
  sortNeighbors: proc(nodes: seq[Node]): iterator: Node = nil
): Edge =
  var successors = proc(node: Node): iterator: Node =
    return G.neighborsIterator(node)
  for edge in genericBfsEdges(G, source, successors, depthLimit, sortNeighbors):
    yield edge
iterator bfsEdges*(
  DG: DiGraph,
  source: Node,
  reverse: bool = false,
  depthLimit: int = -1,
  sortSuccessors: proc(nodes: seq[Node]): iterator: Node = nil
): Edge =
  var successors: proc(node: Node): iterator: Node
  if reverse:
    successors = proc(node: Node): iterator: Node =
      return DG.predecessorsIterator(node)
  else:
    successors = proc(node: Node): iterator: Node =
      return DG.successorsIterator(node)
  for edge in genericBfsEdges(DG, source, successors, depthLimit, sortSuccessors):
    yield edge

proc bfsTree*(
  G: Graph,
  source: Node,
  reverse: bool = false,
  depthLimit: int = -1,
  sortNeighbors: proc(nodes: seq[Node]): iterator: Node = nil
): DiGraph =
  let T = newDiGraph()
  T.addNode(source)
  T.addEdgesFrom(
    bfsEdges(G, source, reverse=reverse, depthLimit=depthLimit, sortNeighbors=sortNeighbors).toSeq()
  )
  return T
proc bfsTree*(
  DG: DiGraph,
  source: Node,
  reverse: bool = false,
  depthLimit: int = -1,
  sortSuccessors: proc(nodes: seq[Node]): iterator: Node = nil
): DiGraph =
  let T = newDiGraph()
  T.addNode(source)
  T.addEdgesFrom(
    bfsEdges(DG, source, reverse=reverse, depthLimit=depthLimit, sortSuccessors=sortSuccessors).toSeq()
  )
  return T

iterator bfsPredecessors*(
  G: Graph,
  source: Node,
  depthLimit: int = -1,
  sortNeighbors: proc(nodes: seq[Node]): iterator: Node = nil
): tuple[node: Node, predecessor: Node] =
  for (s, t) in bfsEdges(G, source, depthLimit=depthLimit, sortNeighbors=sortNeighbors):
    yield (t, s)
iterator bfsPredecessors*(
  DG: DiGraph,
  source: Node,
  depthLimit: int = -1,
  sortSuccessors: proc(nodes: seq[Node]): iterator: Node = nil
): tuple[node: Node, predecessor: Node] =
  for (s, t) in bfsEdges(DG, source, depthLimit=depthLimit, sortSuccessors=sortSuccessors):
    yield (t, s)

iterator bfsSuccessors*(
  G: Graph,
  source: Node,
  depthLimit: int = -1,
  sortNeighbors: proc(nodes: seq[Node]): iterator: Node = nil
): tuple[node: Node, successors: seq[Node]] =
  var parent = source
  var children: seq[Node] = @[]
  for (p, c) in bfsEdges(G, source, depthLimit=depthLimit, sortNeighbors=sortNeighbors):
    if p == parent:
      children.add(c)
      continue
    yield (parent, children)
    children = @[c]
    parent = p
  yield (parent, children)
iterator bfsSuccessors*(
  DG: DiGraph,
  source: Node,
  depthLimit: int = -1,
  sortSuccessors: proc(nodes: seq[Node]): iterator: Node = nil
): tuple[node: Node, successors: seq[Node]] =
  var parent = source
  var children: seq[Node] = @[]
  for (p, c) in bfsEdges(DG, source, depthLimit=depthLimit, sortSuccessors=sortSuccessors):
    if p == parent:
      children.add(c)
      continue
    yield (parent, children)
    children = @[c]
    parent = p
  yield (parent, children)

proc descendantsAtDistance*(
  G: Graph,
  source: Node,
  distance: int
): HashSet[Node] =
  if not G.hasNode(source):
    raise newNNError(fmt"node {source} not in graph")

  var currentDistance = 0

  var currentLayer = initHashSet[Node]()
  currentLayer.incl(source)

  var visited = initHashSet[Node]()
  visited.incl(source)

  while currentDistance < distance:
    var nextLayer = initHashSet[Node]()
    for node in currentLayer:
      for child in G.neighbors(node):
        if child notin visited:
          visited.incl(child)
          nextLayer.incl(child)
    currentLayer = nextLayer
    currentDistance += 1

  return currentLayer

proc descendantsAtDistance*(
  DG: DiGraph,
  source: Node,
  distance: int
): HashSet[Node] =
  if not DG.hasNode(source):
    raise newNNError(fmt"node {source} not in graph")

  var currentDistance = 0

  var currentLayer = initHashSet[Node]()
  currentLayer.incl(source)

  var visited = initHashSet[Node]()
  visited.incl(source)

  while currentDistance < distance:
    var nextLayer = initHashSet[Node]()
    for node in currentLayer:
      for child in DG.successors(node):
        if child notin visited:
          visited.incl(child)
          nextLayer.incl(child)
    currentLayer = nextLayer
    currentDistance += 1

  return currentLayer

iterator bfsBeamEdges*(
  G: Graph,
  source: Node,
  value: proc(node: Node): float,
  width: int = -1
): Edge =
  var widthUsing = width
  if width == -1:
    widthUsing = len(G)

  let successors = proc(v: Node): iterator: Node =
    var nodes = G.neighbors(v)
    var nodeWithValue: seq[tuple[value: float, node: Node]] = @[]
    for node in nodes:
      nodeWithValue.add((-value(node), node))
    nodeWithValue.sort()
    for i in 0..<len(nodeWithValue):
      nodes[i] = nodeWithValue[i].node
    var sortedClippedNodes: seq[Node] = @[]
    for i in 0..<min(width, len(nodes)):
      sortedClippedNodes.add(nodes[i])
    return iterator: Node =
      for node in sortedClippedNodes:
        yield node

  for edge in genericBfsEdges(G, source, neighbors=successors):
    yield edge
iterator bfsBeamEdges*(
  DG: DiGraph,
  source: Node,
  value: proc(node: Node): float,
  width: int = -1
): Edge =
  var widthUsing = width
  if width == -1:
    widthUsing = len(DG)

  let successors = proc(v: Node): iterator: Node =
    var nodes = DG.successors(v)
    var nodeWithValue: seq[tuple[value: float, node: Node]] = @[]
    for node in nodes:
      nodeWithValue.add((-value(node), node))
    nodeWithValue.sort()
    for i in 0..<len(nodeWithValue):
      nodes[i] = nodeWithValue[i].node
    var sortedClippedNodes: seq[Node] = @[]
    for i in 0..<min(width, len(nodes)):
      sortedClippedNodes.add(nodes[i])
    return iterator: Node =
      for node in sortedClippedNodes:
        yield node

  for edge in genericBfsEdges(DG, source, successors=successors):
    yield edge

# -------------------------------------------------------------------
# TODO:
# Triads
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Voronoi cells
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Chains
# -------------------------------------------------------------------

proc dfsCycleForest(
  G: Graph, root: Node = None
): tuple[H: DiGraph, nodes: seq[Node], parents: Table[Node, Node], isNonTree: Table[Edge, bool]] =
  let H = newDiGraph()
  var nodes: seq[Node] = @[]
  var parents = initTable[Node, Node]()
  var isNonTree = initTable[Edge, bool]()
  for (u, v, d) in G.dfsLabeledEdges(source=root):
    if d == "forward":
      if u == v:
        H.addNode(v)
        parents[v] = None
        nodes.add(v)
      else:
        H.addNode(v)
        parents[v] = u
        H.addEdge(v, u)
        isNonTree[(v, u)] = false
        nodes.add(v)
    elif d == "nontree" and v notin H.successorsSet(u):
      H.addEdge(v, u)
      isNonTree[(v, u)] = true
  return (H, nodes, parents, isNonTree)
iterator buildChain(
  DG: DiGraph,
  u, v: Node,
  visited: var HashSet[Node],
  parents: var Table[Node, Node]
): tuple[u, v: Node] =
  var uUsing = u
  var vUsing = v
  while vUsing notin visited:
    yield (uUsing, vUsing)
    visited.incl(vUsing)
    uUsing = vUsing
    vUsing = parents[vUsing]
  yield (uUsing, vUsing)

iterator chainDecomposition*(
  G: Graph, root: Node = None
): seq[tuple[u, v: Node]] =
  var (H, nodes, parents, isNonTree) = dfsCycleForest(G, root)
  var visited = initHashSet[Node]()
  for u in nodes:
    visited.incl(u)
    var edges: seq[Edge] = @[]
    for v in H.successors(u):
      if isNonTree[(u, v)]:
        edges.add((u, v))
    for (u, v) in edges:
      yield buildChain(H, u, v, visited, parents).toSeq()

# -------------------------------------------------------------------
# TODO:
# Connectivity
# -------------------------------------------------------------------

proc plainBfs(G: Graph, source: Node): HashSet[Node] =
  var seen = initHashSet[Node]()
  var nextLevel = initHashSet[Node]()
  nextLevel.incl(source)
  while len(nextLevel) != 0:
    var thisLevel = nextLevel
    nextLevel = initHashSet[Node]()
    for v in thisLevel:
      if v notin seen:
        seen.incl(v)
        for adjNode in G.adj[v]:
          nextLevel.incl(adjNode)
  return seen

proc isConnected*(G: Graph): bool =
  if len(G) == 0:
    raise newNNPointlessConcept("connectivity is undifined for null graph")
  var s = 0
  for node in plainBfs(G, G.nodes[0]):
    s += 1
  return s == len(G)

iterator connectedComponents*(G: Graph): HashSet[Node] =
  var seen = initHashSet[Node]()
  for v in G.nodes():
    if v notin seen:
      let c = plainBfs(G, v)
      for node in c:
        seen.incl(node)
      yield c

proc numberOfConnectedComponents*(G: Graph): int =
  var s = 0
  for cc in G.connectedComponents():
    s += 1
  return s

proc nodeConnectedComponents*(G: Graph, n: Node): HashSet[Node] =
  return plainBfs(G, n)

iterator stronglyConnectedComponents*(DG: DiGraph): HashSet[Node] =
  var preorder = initTable[Node, int]()
  var lowlink = initTable[Node, int]()
  var sccFound = initHashSet[Node]()
  var sccQueue = initDeque[Node]()
  var i = 0 # preorder counter
  for source in DG.nodes():
    if source notin sccFound:
      var queue = initDeque[Node]()
      queue.addFirst(source)
      while len(queue) != 0:
        var v = queue.peekLast()
        if v notin preorder:
          i += 1
          preorder[v] = i
        var done = true
        for w in DG.successors(v):
          if w notin preorder:
            queue.addLast(w)
            done = false
            break
        if done:
          lowlink[v] = preorder[v]
          for w in DG.successors(v):
            if w notin sccFound:
              if preorder[w] > preorder[v]:
                lowlink[v] = min(lowlink[v], lowlink[w])
              else:
                lowlink[v] = min(lowlink[v], preorder[w])
          queue.popLast()
          if lowlink[v] == preorder[v]:
            var scc = initHashSet[Node]()
            scc.incl(v)
            while len(sccQueue) != 0 and preorder[sccQueue.peekLast()] > preorder[v]:
              let k = sccQueue.popLast()
              scc.incl(k)
            for c in scc:
              sccFound.incl(c)
            yield scc
          else:
            sccQueue.addLast(v)

proc numberOfStronglyConnectedComponents*(DG: DiGraph): int =
  var s = 0
  for scc in DG.stronglyConnectedComponents():
    s += 1
  return s

proc isStronglyConnected*(DG: DiGraph): bool =
  if len(DG) == 0:
    raise newNNPointlessConcept("connectivity is undefined for null graph")
  return len(DG.stronglyConnectedComponents().toSeq()[0]) == len(DG)

iterator kosarajuStronglyConnectedComponents*(DG: DiGraph, source: Node = None): HashSet[Node] =
  let RDG = DG.reversed()
  var post = RDG.dfsPostOrderNodes(source=source).toSeq().toDeque()
  var seen = initHashSet[Node]()
  while len(post) != 0:
    var r = post.popLast()
    if r in seen:
      continue
    var c = RDG.dfsPreOrderNodes(r)
    var newNodes = initHashSet[Node]()
    for v in c:
      if v notin seen:
        newNodes.incl(v)
    for n in newNodes:
      seen.incl(n)
    yield newNodes

# iterator stronglyConnectedComponents*(DG: DiGraph): HashSet[Node]
# recursion of iterator is not supported in nim?

proc condensation*(DG: DiGraph, scc: seq[HashSet[Node]] = @[]): DiGraph =
  var sccUsing: seq[HashSet[Node]] = scc
  if len(scc) == 0:
    sccUsing = DG.stronglyConnectedComponents().toSeq()
  var mapping = initTable[Node, int]()
  var members = initTable[int, HashSet[Node]]()
  let C = newDiGraph()
  if len(DG) == 0:
    return C
  for i, component in sccUsing:
    members[i] = component
    for n in component:
      mapping[n] = i
  var numberOfComponents = len(sccUsing)
  for i in 0..<numberOfComponents:
    C.addNode(i)
  for (u, v) in DG.edges():
    if mapping[u] != mapping[v]:
      C.addEdge((mapping[u], mapping[v]))
  return C

iterator plainBfs(DG: DiGraph, source: Node): Node =
  var seen = initHashSet[Node]()
  var nextLevel = initHashSet[Node]()
  nextLevel.incl(source)
  while len(nextLevel) != 0:
    var thisLevel = nextLevel
    nextLevel = initHashSet[Node]()
    for v in thisLevel:
      if v notin seen:
        seen.incl(v)
        for succ in DG.succ[v]:
          nextLevel.incl(succ)
        for pred in DG.pred[v]:
          nextLevel.incl(pred)
        yield v

iterator weaklyConnectedComponents*(DG: DiGraph): HashSet[Node] =
  var seen = initHashSet[Node]()
  for v in DG.nodes():
    if v notin seen:
      let c = plainBfs(DG, v).toSeq().toHashSet()
      for ele in c:
        seen.incl(ele)
      yield c

proc numberOfWeaklyConnectedComponents*(DG: DiGraph): int =
  var s = 0
  for wcc in DG.weaklyConnectedComponents():
    s += 1
  return s

proc isWeaklyConnected*(DG: DiGraph): bool =
  if len(DG) == 0:
    raise newNNPointlessConcept("connectivity is undefined for null graph")
  return len(DG.weaklyConnectedComponents().toSeq()[0]) == len(DG)

iterator attractingComponents*(DG: DiGraph): HashSet[Node] =
  var scc = DG.stronglyConnectedComponents().toSeq()
  var cG = DG.condensation(scc)
  for n in cG.nodes():
    if cG.outDegree(n) == 0:
      yield scc[n]

proc numberOfAttractingComponents*(DG: DiGraph): int =
  var s = 0
  for ac in DG.attractingComponents():
    s += 1
  return s

proc isAttractingComponent*(DG: DiGraph): bool =
  let ac = DG.attractingComponents().toSeq()
  if len(ac) == 1:
    return len(ac[0]) == len(DG)
  return false

iterator biconnectedDfsWithoutComponent(G: Graph): Node  =
  var visited = initHashSet[Node]()
  for start in G.nodes():
    if start in visited:
      continue
    var discovery = initTable[Node, int]()
    discovery[start] = 0
    var lowt = initTable[Node, int]()
    var rootChildren = 0
    visited.incl(start)
    var stack = initDeque[tuple[grandparent, parent: Node, children: seq[Node], idx: int]]()
    stack.addFirst((start, start, G.neighbors(start), 0))
    while len(stack) != 0:
      var (grandparent, parent, children, idx) = stack.peekLast()
      if idx < len(children):
        discard stack.popLast()
        stack.addLast((grandparent, parent, children, idx+1))
        var child = children[idx]
        if grandparent == child:
          continue
        if child in visited:
          if discovery[child] <= discovery[parent]:
            lowt[parent] = min(lowt[parent], discovery[child])
        else:
          lowt[child] = len(discovery.keys().toSeq())
          discovery[child] = len(discovery.keys().toSeq())
          visited.incl(child)
          stack.addLast((parent, child, G.neighbors(child), 0))
      else:
        discard stack.popLast()
        if 1 < len(stack):
          if lowt[parent] >= discovery[grandparent]:
            yield grandparent
          lowt[grandparent] = min(lowt[parent], lowt[grandparent])
        elif len(stack) != 0:
          rootChildren  += 1
    if rootChildren > 1:
      yield start
iterator biconnectedDfsWithComponents(G: Graph): seq[Edge]  =
  var visited = initHashSet[Node]()
  for start in G.nodes():
    if start in visited:
      continue
    var discovery = initTable[Node, int]()
    discovery[start] = 0
    var lowt = initTable[Node, int]()
    var rootChildren = 0
    visited.incl(start)
    var edgeStack = initDeque[Edge]()
    var stack = initDeque[tuple[grandparent, parent: Node, children: seq[Node], idx: int]]()
    stack.addFirst((start, start, G.neighbors(start), 0))
    while len(stack) != 0:
      var (grandparent, parent, children, idx) = stack.peekLast()
      if idx < len(children):
        discard stack.popLast()
        stack.addLast((grandparent, parent, children, idx+1))
        var child = children[idx]
        if grandparent == child:
          continue
        if child in visited:
          if discovery[child] <= discovery[parent]:
            lowt[parent] = min(lowt[parent], discovery[child])
            edgeStack.addLast((parent, child))
        else:
          lowt[child] = len(discovery.keys().toSeq())
          discovery[child] = len(discovery.keys().toSeq())
          visited.incl(child)
          stack.addLast((parent, child, G.neighbors(child), 0))
          edgeStack.addLast((parent, child))
      else:
        discard stack.popLast()
        if 1 < len(stack):
          if lowt[parent] >= discovery[grandparent]:
            var ind = 0
            for i, edge in edgeStack.toSeq():
              if edge == (grandparent, parent):
                ind = i
            var ret: seq[Edge] = @[]
            for i in ind..<len(edgeStack.toSeq()):
              ret.add(edgeStack.toSeq()[i])
            yield ret
            var tmp = edgeStack.toSeq()
            edgeStack = initDeque[Edge]()
            for i in 0..<ind:
              edgeStack.addLast(tmp[i])
          lowt[grandparent] = min(lowt[parent], lowt[grandparent])
        elif len(stack) != 0:
          rootChildren  += 1
          var ind = 0
          for i, edge in edgeStack.toSeq():
            if edge == (grandparent, parent):
              ind = i
          var ret: seq[Edge] = @[]
          for i in ind..<len(edgeStack.toSeq()):
              ret.add(edgeStack.toSeq()[i])
          yield ret

iterator articulationPoints*(G: Graph): Node =
  var seen = initHashSet[Node]()
  for articulation in G.biconnectedDfsWithoutComponent():
    if articulation notin seen:
      seen.incl(articulation)
      yield articulation

iterator biconnectedComponents*(G: Graph): HashSet[Node] =
  for comp in G.biconnectedDfsWithComponents():
    var ret = initHashSet[Node]()
    for edge in comp:
      ret.incl(edge.u)
      ret.incl(edge.v)
    yield ret

iterator biconnectedComponentEdges*(G: Graph): seq[Edge] =
  for edges in G.biconnectedDfsWithComponents():
    yield edges

proc isBiconnected*(G: Graph): bool =
  let bcc = G.biconnectedComponents().toSeq()
  if len(bcc) == 1:
    return len(bcc[0]) == len(G)
  return false

# -------------------------------------------------------------------
# TODO:
# Eulerian
# -------------------------------------------------------------------

proc isEulerian*(G: Graph): bool =
  var d: seq[bool] = @[]
  for node in G.nodes():
    d.add(G.degree(node) mod 2 == 0)
  return all(d, proc(b: bool): bool = return b) and G.isConnected()
proc isEulerian*(DG: DiGraph): bool =
  var d: seq[bool] = @[]
  for node in DG.nodes():
    d.add(DG.inDegree(node) == DG.outDegree(node))
  return all(d, proc(b: bool): bool = return b) and DG.isStronglyConnected()

# -------------------------------------------------------------------
# TODO:
# Centrality
# -------------------------------------------------------------------

proc degreeCentrality*(G: Graph): Table[Node, float] =
  var ret = initTable[Node, float]()
  if len(G) <= 1:
    for node in G.nodes():
      ret[node] = 1.0
    return ret
  let s = 1.0 / (len(G) - 1).float
  for node in G.nodes():
    ret[node] = G.degree(node).float * s
  return ret

proc inDegreeCentrality*(DG: DiGraph): Table[Node, float] =
  var ret = initTable[Node, float]()
  if len(DG) <= 1:
    for node in DG.nodes():
      ret[node] = 1.0
    return ret
  let s = 1.0 / (len(DG) - 1).float
  for node in DG.nodes():
    ret[node] = DG.inDegree(node).float * s
  return ret

proc outDegreeCentrality*(DG: DiGraph): Table[Node, float] =
  var ret = initTable[Node, float]()
  if len(DG) <= 1:
    for node in DG.nodes():
      ret[node] = 1.0
    return ret
  let s = 1.0 / (len(DG) - 1).float
  for node in DG.nodes():
    ret[node] = DG.outDegree(node).float * s
  return ret

proc eigenvectorCentrality*(
  G: Graph,
  maxIter: int = 100,
  tol: float = 1.0e-6,
  nstart: TableRef[Node, float] = nil,
  weight: TableRef[Edge, float] = nil,
): Table[Node, float] =
  if len(G) == 0:
    raise newNNPointlessConcept("cannot compute centrality for null graph")
  var nstartUsing: Table[Node, float]
  if nstart == nil:
    for v in G.nodes():
      nstartUsing[v] = 1.0
  else:
    nstartUsing = nstart[]
  var tmp: seq[bool] = @[]
  for v in nstartUsing.values():
    tmp.add(v == 0.0)
  if all(tmp, proc(b: bool): bool = return b):
    raise newNNError("initial vector must not be all zeros")
  var nstartSum = 0.0
  for v in nstartUsing.values():
    nstartSum += v
  var x = initTable[Node, float]()
  for (k, v) in nstartUsing.pairs():
    x[k] = v / nstartSum
  var N = G.numberOfNodes()
  for i in 0..<maxIter:
    var xlast = x
    x = xlast
    for n in x.keys():
      for nbr in G.neighbors(n):
        var w = 1.0
        if weight != nil:
          w = weight.getOrDefault((n, nbr), weight.getOrDefault((nbr, n), 1.0))
        x[nbr] += xlast[n] * w
    var norm = 0.0
    for v in x.values():
      norm += v * v
    norm = sqrt(norm)
    if norm == 0.0:
      norm = 1.0
    for (k, v) in x.pairs():
      x[k] = v / norm
    var s = 0.0
    for n in x.keys():
      s += abs(x[n] - xlast[n])
    if s < N.float * tol:
      return x
  raise newNNPowerIterationFailedConvergence(maxIter)
proc eigenvectorCentrality*(
  DG: DiGraph,
  maxIter: int = 100,
  tol: float = 1.0e-6,
  nstart: TableRef[Node, float] = nil,
  weight: TableRef[Edge, float] = nil,
): Table[Node, float] =
  if len(DG) == 0:
    raise newNNPointlessConcept("cannot compute centrality for null graph")
  var nstartUsing: Table[Node, float]
  if nstart == nil:
    for v in DG.nodes():
      nstartUsing[v] = 1.0
  else:
    nstartUsing = nstart[]
  var tmp: seq[bool] = @[]
  for v in nstartUsing.values():
    tmp.add(v == 0.0)
  if all(tmp, proc(b: bool): bool = return b):
    raise newNNError("initial vector must not be all zeros")
  var nstartSum = 0.0
  for v in nstartUsing.values():
    nstartSum += v
  var x = initTable[Node, float]()
  for (k, v) in nstartUsing.pairs():
    x[k] = v / nstartSum
  var N = DG.numberOfNodes()
  for i in 0..<maxIter:
    var xlast = x
    x = xlast
    for n in x.keys():
      for succ in DG.successors(n):
        var w = 1.0
        if weight != nil:
          w = weight.getOrDefault((n, succ), 1.0)
        x[succ] += xlast[n] * w
    var norm = 0.0
    for v in x.values():
      norm += v * v
    norm = sqrt(norm)
    if norm == 0.0:
      norm = 1.0
    for (k, v) in x.pairs():
      x[k] = v / norm
    var s = 0.0
    for n in x.keys():
      s += abs(x[n] - xlast[n])
    if s < N.float * tol:
      return x
  raise newNNPowerIterationFailedConvergence(maxIter)

proc katzCentrality*(
  G: Graph,
  alpha: float = 0.1,
  beta: float = 1.0,
  maxIter: int = 1000,
  tol: float = 1.0e-6,
  nstart: TableRef[Node, float] = nil,
  normalized: bool = true,
  weight: TableRef[Edge, float] = nil
): Table[Node, float] =
  if len(G) == 0:
    raise newNNPointlessConcept("cannot compute centrality for null graph")
  var N = G.numberOfNodes()
  var x = initTable[Node, float]()
  if nstart == nil:
    for n in G.nodes():
      x[n] = 0.0
  else:
    x = nstart[]

  # TODO: beta table
  var b = initTable[Node, float]()
  for n in G.nodes():
    b[n] = beta

  for i in 0..<maxIter:
    var xlast = x
    for k in x.keys():
      x[k] = 0.0
    var w: float
    for n in x.keys():
      for nbr in G.neighbors(n):
        w = 1.0
        if weight != nil:
          w = weight.getOrDefault((n, nbr), weight.getOrDefault((nbr, n), 1.0))
        x[nbr] += xlast[n] * w
    for n in x.keys():
      x[n] = alpha * x[n] + b[n]
    var err = 0.0
    for n in x.keys():
      err += abs(x[n] - xlast[n])
    if err < N.float * tol:
      if normalized:
        var norm = 0.0
        for v in x.values():
          norm += v * v
        norm = sqrt(norm)
        if norm == 0.0:
          norm = 1.0
        for n in x.keys():
          x[n] /= norm
        return x
  raise newNNPowerIterationFailedConvergence(maxIter)
proc katzCentrality*(
  DG: DiGraph,
  alpha: float = 0.1,
  beta: float = 1.0,
  maxIter: int = 1000,
  tol: float = 1.0e-6,
  nstart: TableRef[Node, float] = nil,
  normalized: bool = true,
  weight: TableRef[Edge, float] = nil
): Table[Node, float] =
  if len(DG) == 0:
    raise newNNPointlessConcept("cannot compute centrality for null graph")
  var N = DG.numberOfNodes()
  var x = initTable[Node, float]()
  if nstart == nil:
    for n in DG.nodes():
      x[n] = 0.0
  else:
    x = nstart[]

  # TODO: beta table
  var b = initTable[Node, float]()
  for n in DG.nodes():
    b[n] = beta

  for i in 0..<maxIter:
    var xlast = x
    for k in x.keys():
      x[k] = 0.0
    var w: float
    for n in x.keys():
      for succ in DG.successors(n):
        w = 1.0
        if weight != nil:
          w = weight.getOrDefault((n, succ), 1.0)
        x[succ] += xlast[n] * w
    for n in x.keys():
      x[n] = alpha * x[n] + b[n]
    var err = 0.0
    for n in x.keys():
      err += abs(x[n] - xlast[n])
    if err < N.float * tol:
      if normalized:
        var norm = 0.0
        for v in x.values():
          norm += v * v
        norm = sqrt(norm)
        if norm == 0.0:
          norm = 1.0
        for n in x.keys():
          x[n] /= norm
        return x
  raise newNNPowerIterationFailedConvergence(maxIter)

proc closenessCentrality*(
  G: Graph,
  u: Node = None,
  distance: TableRef[Edge, float] = nil,
  wfImprove: bool = true
): Table[Node, float] =
  var cc = initTable[Node, float]()
  var nodes: seq[Node]
  if u != None:
    nodes = @[u]
  else:
    nodes = G.nodes()
  if distance == nil: # unweighted
    for n in nodes:
      var sp = singleSourceShortestPathLength(G, n)
      var totsp = sum(sp.values().toSeq()).float
      var lenG = len(G)
      var tmpcc = 0.0
      if totsp > 0.0 and lenG > 1:
        tmpcc = (len(sp.keys().toSeq()).float - 1.0) / totsp
        if wfImprove:
          var s = (len(sp.keys().toSeq()).float - 1.0) / (lenG.float - 1.0)
          tmpcc *= s
      cc[n] = tmpcc
  else: # weighted
    for n in nodes:
      var sp = singleSourceDijkstraPathLength(G, n, weight=distance)
      var totsp = sum(sp.values().toSeq())
      var lenG = len(G)
      var tmpcc = 0.0
      if totsp > 0.0 and lenG > 1:
        tmpcc = (len(sp.keys().toSeq()).float - 1.0) / totsp
        if wfImprove:
          var s = (len(sp.keys().toSeq()).float - 1.0) / (lenG.float - 1.0)
          tmpcc *= s
      cc[n] = tmpcc
  if u != None:
    return {u: cc[u]}.toTable()
  else:
    return cc
proc closenessCentrality*(
  DG: DiGraph,
  u: Node = None,
  distance: TableRef[Edge, float] = nil,
  wfImprove: bool = true
): Table[Node, float] =
  var RDG = DG.reverse()
  var cc = initTable[Node, float]()
  var nodes: seq[Node]
  if u != None:
    nodes = @[u]
  else:
    nodes = RDG.nodes()
  if distance == nil: # unweighted
    for n in nodes:
      var sp = singleSourceShortestPathLength(RDG, n)
      var totsp = sum(sp.values().toSeq()).float
      var lenG = len(RDG)
      var tmpcc = 0.0
      if totsp > 0.0 and lenG > 1:
        tmpcc = (len(sp.keys().toSeq()).float - 1.0) / totsp
        if wfImprove:
          var s = (len(sp.keys().toSeq()).float - 1.0) / (lenG.float - 1.0)
          tmpcc *= s
      cc[n] = tmpcc
  else: # weighted
    for n in nodes:
      var reversedDistance = newTable[Edge, float]()
      for (edge, dist) in distance.pairs():
        reversedDistance[edge.reversed()] = dist
      var sp = singleSourceDijkstraPathLength(RDG, n, weight=reversedDistance)
      var totsp = sum(sp.values().toSeq())
      var lenG = len(RDG)
      var tmpcc = 0.0
      if totsp > 0.0 and lenG > 1:
        tmpcc = (len(sp.keys().toSeq()).float - 1.0) / totsp
        if wfImprove:
          var s = (len(sp.keys().toSeq()).float - 1.0) / (lenG.float - 1.0)
          tmpcc *= s
      cc[n] = tmpcc
  if u != None:
    return {u: cc[u]}.toTable()
  else:
    return cc

proc incrementalClosenessCentrality*(
  G: Graph,
  edge: Edge,
  prevCC: Table[Node, float], # TODO: should be TableRef?
  isInsertion: bool = true,
  wfImproved: bool = true
): Table[Node, float] =
  if len(prevCC) != 0 and prevCC.keys().toSeq().toHashSet() != G.nodesSet():
    raise newNNError("prevCC and G do not have same nodes")

  var (u, v) = edge
  var du: Table[Node, int]
  var dv: Table[Node, int]
  if isInsertion:
    du = singleSourceShortestPathLength(G, u)
    dv = singleSourceShortestPathLength(G, v)
    G.addEdge(u, v)
  else:
    G.removeEdge(u, v)
    du = singleSourceShortestPathLength(G, u)
    dv = singleSourceShortestPathLength(G, v)

  if len(prevCC) == 0:
    return closenessCentrality(G)

  var nodes = G.nodes()
  var cc = initTable[Node, float]()
  for n in nodes:
    if n in du and n in dv and abs(du[n] - dv[n]) <= 1:
      cc[n] = prevCC[n]
    else:
      var sp = singleSourceShortestPathLength(G, n)
      var totsp = sum(sp.values().toSeq()).float
      var lenG = len(G).float
      var tmpcc = 0.0
      if totsp > 0.0 and lenG > 1.0:
        tmpcc = (len(sp) - 1).float / totsp
        if wfImproved:
          var s = (len(sp) - 1).float / (lenG - 1.0)
          tmpcc *= s
      cc[n] = tmpcc
  if isInsertion:
    G.removeEdge(u, v)
  else:
    G.addEdge(u, v)
  return cc

# -------------------------------------------------------------------
# TODO:
# DAG
# -------------------------------------------------------------------

proc descendants*(DG: DiGraph, source: Node): HashSet[Node] =
  if not DG.hasNode(source):
    raise newNNNodeNotFound(source)
  var des = initHashSet[Node]()
  for n in shortestPathLength(DG, source)[source].keys():
    des.incl(n)
  des.excl(source)
  return des

proc ancestors*(DG: DiGraph, source: Node): HashSet[Node] =
  if not DG.hasNode(source):
    raise newNNNodeNotFound(source)
  var anc = initHashSet[Node]()
  for n in shortestPathLength(DG, target=source).keys():
    anc.incl(n)
  anc.excl(source)
  return anc

iterator topologicalGenerations*(DG: DiGraph): seq[Node] =
  var indegreeMap = initTable[Node, int]()
  for (v, ind) in DG.inDegree().pairs():
    if ind > 0:
      indegreeMap[v] = ind
  var zeroIndegree: seq[Node] = @[]
  for (v, ind) in DG.inDegree().pairs():
    if ind == 0:
      zeroIndegree.add(v)
  zeroIndegree.sort()

  while len(zeroIndegree) != 0:
    var thisGeneration = zeroIndegree
    zeroIndegree = @[]
    for node in thisGeneration:
      for child in DG.successors(node):
        indegreeMap[child] -= 1
        if indegreeMap[child] == 0:
          zeroIndegree.add(child)
          indegreeMap.del(child)
    yield thisGeneration
  if len(indegreeMap.keys().toSeq()) != 0:
    raise newNNUnfeasible("graph contains a cycle")

iterator topologicalSort*(DG: DiGraph): Node =
  for generation in DG.topologicalGenerations():
    for node in generation:
      yield node

proc hasCycle*(DG: DiGraph): bool =
  try:
    discard topologicalSort(DG).toSeq()
  except NNUnfeasible:
    return true
  return false

proc isDirectedAcyclicGraph*(DG: DiGraph): bool =
  return not DG.hasCycle()

proc isAperiodic*(DG: DiGraph): bool =
  if len(DG) == 0:
    raise newNNPointlessConcept("cannnot check null graph is aperiodic")
  var s = DG.nodes()[0]
  var levels = {s: 0}.toTable()
  var thisLevel = @[s]
  var g = 0
  var lev = 1
  while len(thisLevel) != 0:
    var nextLevel: seq[Node] = @[]
    for u in thisLevel:
      for v in DG.successors(u):
        if v in levels:
          g = gcd(g, levels[u] - levels[v] + 1)
        else:
          nextLevel.add(v)
          levels[v] = lev
    thisLevel = nextLevel
    lev += 1
  if len(levels) == len(DG):
    return g == 1
  else:
    return g == 1 and isAperiodic(DG.subgraph(DG.nodesSet() - levels.keys().toSeq().toHashSet()))

iterator lexicographicalTopologicalSort*(DG: DiGraph, key: proc(node: Node): int = nil): Node =
  var keyUsing = key
  if key == nil:
    keyUsing = proc(node: Node): int = return node
  var nodeidMap = initTable[Node, int]()
  for i, n in DG.nodesSeq():
    nodeidMap[n] = i
  var indegreeMap = initTable[Node, int]()
  for (v, d) in DG.inDegree().pairs():
    if d > 0:
      indegreeMap[v] = d
  var zeroIndegree = initHeapQueue[tuple[key: int, id: int, node: Node]]()
  for (v, d) in DG.inDegree().pairs():
    if d == 0:
      zeroIndegree.push((keyUsing(v), nodeidMap[v], v))
  while len(zeroIndegree) != 0:
    var (_, _, node) = zeroIndegree.pop()
    for (_, child) in DG.edges(node):
      indegreeMap[child] -= 1
      if indegreeMap[child] == 0:
        zeroIndegree.push((keyUsing(child), nodeidMap[child], child))
        indegreeMap.del(child)
    yield node
  if len(indegreeMap.keys().toSeq()) != 0:
    raise newNNUnfeasible("graph contains a cycle")

iterator allTopologicalSorts*(DG: DiGraph): seq[Node] =
  var count = DG.inDegree()
  var D = initDeque[Node]()
  for v, d in DG.inDegree().pairs():
    if d == 0:
      D.addLast(v)
  var bases: seq[Node] = @[]
  var currentSort: seq[Node] = @[]
  while true:
    if len(currentSort) == len(DG):
      yield currentSort
      while len(currentSort) > 0:
        var q = currentSort.pop()
        for _, j in DG.successorsSeq(q):
          count[j] += 1
        while len(D) > 0 and count[D.peekLast()] > 0:
          D.popLast()
        D.addFirst(q)
        if D.peekLast() == bases[^1]:
          bases.pop()
        else:
          break
    else:
      if len(D) == 0:
        raise newNNUnfeasible("graph contains a cycle")
      var q = D.popLast()
      for _, j in DG.successorsSeq(q):
        count[j] -= 1
        if count[j] == 0:
          D.addLast(j)
      currentSort.add(q)
      if len(bases) < len(currentSort):
        bases.add(q)
    if len(bases) == 0:
      break

# -------------------------------------------------------------------
# TODO:
# Link Prediction
# -------------------------------------------------------------------

proc applyPrediction(
  G: Graph,
  f: proc(edge: Edge): float,
  ebunch: seq[Edge] = @[]
): seq[tuple[edge: Edge, prediction: float]] =
  var ret: seq[tuple[edge: Edge, prediction: float]] = @[]
  if len(ebunch) == 0:
    for edge in G.nonEdges():
      ret.add((edge, f(edge)))
    return ret
  for edge in ebunch:
    ret.add((edge, f(edge)))
  return ret

proc resourceAllocationIndex*(
  G: Graph,
  ebunch: seq[Edge] = @[]
): seq[tuple[edge: Edge, prediction: float]] =
  let f: proc(edge: Edge): float =
    proc(edge: Edge): float =
      var s = 0.0
      for n in G.commonNeighbors(edge.u, edge.v):
        s += 1.0 / G.degree(n).float
      return s
  return applyPrediction(G, f, ebunch)

proc jaccardCoefficient*(
  G: Graph,
  ebunch: seq[Edge] = @[]
): seq[tuple[edge: Edge, prediction: float]] =
  let f: proc(edge: Edge): float =
    proc(edge: Edge): float =
      let u = edge.u
      let v = edge.v
      let unionSize = len(G.neighborsSet(u) + G.neighborsSet(v))
      if unionSize == 0:
        return 0.0
      return len(G.commonNeighbors(u, v)).float / unionSize.float
  return applyPrediction(G, f, ebunch)

proc adamicAdarIndex*(
  G: Graph,
  ebunch: seq[Edge] = @[]
): seq[tuple[edge: Edge, prediction: float]] =
  let f: proc(edge: Edge): float =
    proc(edge: Edge): float =
      let u = edge.u
      let v = edge.v
      var s = 0.0
      for n in G.commonNeighbors(u, v):
        s += 1.0 / log(G.degree(n).float, E)
      return s
  return applyPrediction(G, f, ebunch)

proc commonNeighborCentrality*(
  G: Graph,
  ebunch: seq[Edge] = @[],
  alpha: float = 0.8
): seq[tuple[edge: Edge, prediction: float]] =
  let f: proc(edge: Edge): float =
    proc(edge: Edge): float =
      let sp = shortestPath(G)
      return alpha * len(commonNeighbors(G, edge.u, edge.v)).float + (1.0 - alpha) * (G.numberOfNodes().float / (len(sp[edge.u][edge.v]) - 1).float)
  return applyPrediction(G, f, ebunch)

proc prefentialAttachment*(
  G: Graph,
  ebunch: seq[Edge] = @[]
): seq[tuple[edge: Edge, prediction: float]] =
  let f: proc(edge: Edge): float =
    proc(edge: Edge): float =
      let u = edge.u
      let v = edge.v
      return (G.degree(u) * G.degree(v)).float
  return applyPrediction(G, f, ebunch)

proc cnSoundarajanHopcroft*(
  G: Graph,
  community: Table[Node, int], # {node: community, ...}
  ebunch: seq[Edge] = @[]
): seq[tuple[edge: Edge, prediction: float]] =
  let f: proc(edge: Edge): float =
    proc(edge: Edge): float =
      let u = edge.u
      let v = edge.v
      let cu = community[u]
      let cv = community[v]
      let comnbrs = G.commonNeighbors(u, v)
      var neighbors = 0
      for w in comnbrs:
        if cu == cv:
          if community[w] == cu:
            neighbors += 1
      return (neighbors + len(comnbrs)).float
  return applyPrediction(G, f, ebunch)

proc raIndexSoundarajanHopcroft*(
  G: Graph,
  community: Table[Node, int], # {node: community, ...}
  ebunch: seq[Edge] = @[]
): seq[tuple[edge: Edge, prediction: float]] =
  let f: proc(edge: Edge): float =
    proc(edge: Edge): float =
      let u = edge.u
      let v = edge.v
      let cu = community[u]
      let cv = community[v]
      if cu != cv:
        return 0.0
      var comnbrs = G.commonNeighbors(u, v)
      var s = 0.0
      for w in comnbrs:
        if community[w] == cu:
          s += 1.0 / G.degree(w).float
      return s
  return applyPrediction(G, f, ebunch)

proc withinInterCluster*(
    G: Graph,
    community: Table[Node, int], # {node: community, ...}
    ebunch: seq[Edge] = @[],
    delta: float = 0.001
): seq[tuple[edge: Edge, prediction: float]] =
  if delta <= 0.0:
    raise newNNAlgorithmError("delta must be greater than zero")
  let f: proc(edge: Edge): float =
    proc(edge: Edge): float =
      let u = edge.u
      let v = edge.v
      let cu = community[u]
      let cv = community[v]
      if cu != cv:
        return 0.0
      var comnbrs = G.commonNeighborsSet(u, v)
      var within = initHashSet[Node]()
      for w in comnbrs:
        if community[w] == cu:
          within.incl(w)
      var inter = comnbrs - within
      return len(within).float / (len(inter).float + delta)
  return applyPrediction(G, f, ebunch)

# -------------------------------------------------------------------
# D-Separation
# -------------------------------------------------------------------

proc dSeparated*(
  DG: DiGraph,
  x: HashSet[Node],
  y: HashSet[Node],
  z: HashSet[Node]
): bool =
  if not DG.isDirectedAcyclicGraph():
    raise newNNError("d-separation is not for non DAG")

  var xyz = x + y + z

  for n in xyz:
    if n notin DG.nodesSet():
      raise newNNNodeNotFound(n)

  let DGCopy = DG.copyAsDiGraph()

  var leaves = initDeque[Node]()
  for n in DGCopy.nodes():
    if DGCopy.outDegree(n) == 0:
      leaves.addLast(n)

  while len(leaves) > 0:
    var leaf = leaves.popFirst()
    if leaf notin xyz:
      for p in DGCopy.predecessors(leaf):
        if DGCopy.outDegree(p) == 1:
          leaves.addLast(p)
      DGCopy.removeNode(leaf)

  var edgesToRemove: seq[Edge] = @[]
  for v in z:
    for edge in DGCopy.edges(v):
      edgesToRemove.add(edge)
  DGCopy.removeEdgesFrom(edgesToRemove)

  var disjointSet = newUnionFind(DGCopy.nodesSet())
  for component in DGCopy.weaklyConnectedComponents():
    disjointSet.union(component)
  disjointSet.union(x)
  disjointSet.union(y)

  if len(x) != 0 and len(y) != 0 and disjointSet[x.toSeq()[0]] == disjointSet[y.toSeq()[0]]:
    return false
  return true

# -------------------------------------------------------------------
# TODO:
# Tree
# -------------------------------------------------------------------

proc isTree*(G: Graph): bool =
  if len(G) == 0:
    raise newNNPointlessConcept("graph has no nodes")
  return len(G) - 1 == G.numberOfEdges() and G.isConnected()
proc isTree*(DG: DiGraph): bool =
  if len(DG) == 0:
    raise newNNPointlessConcept("graph has no nodes")
  return len(DG) - 1 == DG.numberOfEdges() and DG.isWeaklyConnected()

proc isForest*(G: Graph): bool =
  if len(G) == 0:
    raise newNNPointlessConcept("graph has no nodes")
  var components: seq[Graph] = @[]
  for c in G.connectedComponents():
    components.add(G.subgraph(c))
  return all(components, proc(c: Graph): bool = return len(c) - 1 == c.numberOfEdges())
proc isForest*(DG: DiGraph): bool =
  if len(DG) == 0:
    raise newNNPointlessConcept("graph has no nodes")
  var components: seq[DiGraph] = @[]
  for c in DG.weaklyConnectedComponents():
    components.add(DG.subgraph(c))
  return all(components, proc(c: DiGraph): bool = return len(c) - 1 == c.numberOfEdges())

proc isBranching*(DG: DiGraph): bool =
  var maxInDegree = 0
  for d in DG.inDegree().values():
    maxInDegree = max(maxInDegree, d)
  return DG.isForest() and maxInDegree <= 1

proc isArborescence*(DG: DiGraph): bool =
  var maxInDegree = 0
  for d in DG.inDegree().values():
    maxInDegree = max(maxInDegree, d)
  return DG.isTree() and maxInDegree <= 1

# -------------------------------------------------------------------
# Wiener Index
# -------------------------------------------------------------------

proc wienerIndex*(
  G: Graph,
  weight: TableRef[Edge, float] = nil,
): float =
  if not G.isConnected():
    return Inf
  var total = 0.0
  for p in G.shortestPathLength(weight=weight).values():
    for v in p.values():
      total += v
  return total / 2.0
proc wienerIndex*(
  DG: DiGraph,
  weight: TableRef[Edge, float] = nil,
): float =
  if not DG.isStronglyConnected():
    return Inf
  var total = 0.0
  for p in DG.shortestPathLength(weight=weight).values():
    for v in p.values():
      total += v
  return total

# -------------------------------------------------------------------
# Vitality
# -------------------------------------------------------------------

proc closenessVitality*(
  G: Graph,
  node: Node = None,
  weight: TableRef[Edge, float] = nil,
): Table[Node, float] =
  let wiener = wienerIndex(G, weight=weight)
  if node != None:
    var newW = newTable[Edge, float]()
    if weight == nil:
      for edge in G.subgraph(G.nodesSet() - @[node].toHashSet()).edges():
        newW[edge] = 1.0
    else:
      for edge in weight.keys():
        if node == edge.u or node == edge.v:
          continue
        discard newW.hasKeyOrPut(edge, weight.getOrDefault(edge, 1.0))
    let after = wienerIndex(G.subgraph(G.nodesSet() - @[node].toHashSet()), weight=newW)
    return {node: wiener - after}.toTable()
  var ret = initTable[Node, float]()
  for v in G.nodes():
    var newW = newTable[Edge, float]()
    if weight == nil:
      for edge in G.subgraph(G.nodesSet() - @[v].toHashSet()).edges():
        newW[edge] = 1.0
    else:
      for edge in weight.keys():
        if v == edge.u or v == edge.v:
          continue
        discard newW.hasKeyOrPut(edge, weight.getOrDefault(edge, 1.0))
    let w = wienerIndex(G, weight=weight)
    let aft = wienerIndex(G.subgraph(G.nodesSet() - @[v].toHashSet()), weight=newW)
    ret[v] = w - aft
  return ret
proc closenessVitality*(
  DG: DiGraph,
  node: Node = None,
  weight: TableRef[Edge, float] = nil,
): Table[Node, float] =
  let wiener = wienerIndex(DG, weight=weight)
  if node != None:
    var newW = newTable[Edge, float]()
    if weight == nil:
      for edge in DG.subgraph(DG.nodesSet() - @[node].toHashSet()).edges():
        newW[edge] = 1.0
    else:
      for edge in weight.keys():
        if node == edge.u or node == edge.v:
          continue
        discard newW.hasKeyOrPut(edge, weight.getOrDefault(edge, 1.0))
    let after = wienerIndex(DG.subgraph(DG.nodesSet() - @[node].toHashSet()), weight=newW)
    return {node: wiener - after}.toTable()
  var ret = initTable[Node, float]()
  for v in DG.nodes():
    var newW = newTable[Edge, float]()
    if weight == nil:
      for edge in DG.subgraph(DG.nodesSet() - @[v].toHashSet()).edges():
        newW[edge] = 1.0
    else:
      for edge in weight.keys():
        if v == edge.u or v == edge.v:
          continue
        discard newW.hasKeyOrPut(edge, weight.getOrDefault(edge, 1.0))
    let w = wienerIndex(DG, weight=weight)
    let aft = wienerIndex(DG.subgraph(DG.nodesSet() - @[v].toHashSet()), weight=newW)
    ret[v] = w - aft
  return ret
