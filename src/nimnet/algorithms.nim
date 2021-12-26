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
# D-Separation
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# DAG
# -------------------------------------------------------------------

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
# TODO:
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

# TODO:
# proc commonNeighborCentrality*(
#   G: Graph,
#   ebunch: seq[Edge] = @[],
#   alpha: float = 0.8
# ): seq[tuple[edge: Edge, prediction: float]]

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
# TODO:
# Lowest Common Ancestor
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Matching
# -------------------------------------------------------------------

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

# -------------------------------------------------------------------
# TODO:
# Rich Club
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Shortest Paths
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Similarity Measures
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Simple Paths
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Small World
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# s-metric
# -------------------------------------------------------------------

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
# TODO:
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
# Tree
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Triads
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Vitality
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Voronoi cells
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Wiener Index
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
