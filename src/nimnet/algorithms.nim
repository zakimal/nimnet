import tables
import strformat
import sequtils
import sets
import math

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
# TODO:
# Graphical Degree Sequence
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Hierarchy
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Hybrid
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO:
# Isolates
# -------------------------------------------------------------------

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