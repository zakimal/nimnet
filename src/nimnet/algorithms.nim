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
# Chains
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
# TODO:
# Connectivity
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
# TODO:
# Dominating Sets
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Efficiency
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Eulerian
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

# proc pageRank*(
#   G: Graph,
#   alpha: float = 0.85,
#   personalization: TableRef[Node, float] = nil,
#   maxIter: int = 100,
#   tol: float = 1.0e-6,
#   nstart: TableRef[Node, float] = nil,
#   dangling: TableRef[Node, float] = nil
# ): Table[Node, float] =

# proc pageRank*(
#   DG: DiGraph,
#   alpha: float = 0.85,
#   personalization: TableRef[Node, float] = nil,
#   maxIter: int = 100,
#   tol: float = 1.0e-6,
#   nstart: TableRef[Node, float] = nil,
#   dangling: TableRef[Node, float] = nil
# ): Table[Node, float] =

# proc pageRank*(
#   G: Graph,
#   alpha: float = 0.85,
#   personalization: TableRef[Node, float] = nil,
#   maxIter: int = 100,
#   tol: float = 1.0e-6,
#   nstart: TableRef[Node, float] = nil,
#   weight: TableRef[Node, float] = nil,
#   dangling: TableRef[Node, float] = nil
# ): Table[Node, float] =

# proc pageRank*(
#   DG: DiGraph,
#   alpha: float = 0.85,
#   personalization: TableRef[Node, float] = nil,
#   maxIter: int = 100,
#   tol: float = 1.0e-6,
#   nstart: TableRef[Node, float] = nil,
#   weight: TableRef[Node, float] = nil,
#   dangling: TableRef[Node, float] = nil
# ): Table[Node, float] =

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