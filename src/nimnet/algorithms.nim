import tables
import strformat
import sequtils
import sets
import math
import heapqueue
import algorithm

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
# TODO:
# Cores
# -------------------------------------------------------------------

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