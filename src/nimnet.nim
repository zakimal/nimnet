import algorithm
import sets
import sequtils
import tables
import strformat

# -------------------------------------------------------------------
#  Basic types
# -------------------------------------------------------------------

type Node* = int
const None* = -1.Node

type Edge* = tuple[u, v: Node]

type Graph* = ref object of RootObj
  ## Undirected graphs with self loops
  adj*: Table[Node, HashSet[Node]]

type DiGraph* = ref object of RootObj
  ## Directed graphs with self loops
  pred*: Table[Node, HashSet[Node]]
  succ*: Table[Node, HashSet[Node]]

type WeightedGraph* = ref object of RootObj
  ## Weighted Undirected graphs with self loops
  adjWithWeight*: Table[Node, HashSet[tuple[node: Node, weight: float]]]

type WeightedDiGraph* = ref object of RootObj
  ## Weighted Directed graphs with self loops
  predWithWeight*: Table[Node, HashSet[tuple[node: Node, weight: float]]]
  succ*: Table[Node, HashSet[Node]]

# -------------------------------------------------------------------
#  Exceptions
# -------------------------------------------------------------------

type
  NNException*                       = ref object of CatchableError
  NNError*                           = ref object of NNException
  NNPointlessConcept*                = ref object of NNException
  NNAlgorithmError*                  = ref object of NNException
  NNUnfeasible*                      = ref object of NNAlgorithmError
  NNNoPath*                          = ref object of NNUnfeasible
  NNNoCycle*                         = ref object of NNUnfeasible
  NNHasACycle*                       = ref object of NNException
  NNUnbounded*                       = ref object of NNAlgorithmError
  NNNotImplemented*                  = ref object of NNException
  NNNodeNotFound*                    = ref object of NNException
  NNAmbiguousSolution*               = ref object of NNException
  NNExceededMaxIterations*           = ref object of NNException
  NNPowerIterationFailedConvergence* = ref object of NNExceededMaxIterations
    numIterations: int
proc newNNException*(msg: string): NNException =
  var e = NNException()
  e.msg = msg
  return e
proc newNNError*(msg: string): NNError =
  var e = NNError()
  e.msg = msg
  return e
proc newNNPointlessConcept*(msg: string): NNPointlessConcept =
  var e = NNPointlessConcept()
  e.msg = msg
  return e
proc newNNAlgorithmError*(msg: string): NNAlgorithmError =
  var e = NNAlgorithmError()
  e.msg = msg
  return e
proc newNNUnfeasible*(msg: string): NNUnfeasible =
  var e = NNUnfeasible()
  e.msg = msg
  return e
proc newNNNoPath*(msg: string): NNNoPath =
  var e = NNNoPath()
  e.msg = msg
  return e
proc newNNNoCycle*(msg: string): NNNoCycle =
  var e = NNNoCycle()
  e.msg = msg
  return e
proc newNNHasACycle*(msg: string): NNHasACycle =
  var e = NNHasACycle()
  e.msg = msg
  return e
proc newNNUnbounded*(msg: string): NNUnbounded =
  var e = NNUnbounded()
  e.msg = msg
  return e
proc newNNNotImplemented*(algorithmName: string, graphType: string): NNNotImplemented =
  var e = NNNotImplemented()
  e.msg = fmt"{algorithmName} is not implemented for {graphType}"
  return e
proc newNNNodeNotFound*(nodeNotFound: Node): NNNodeNotFound =
  var e = NNNodeNotFound()
  e.msg = fmt"node {nodeNotFound} not found"
  return e
proc newNNAmbiguousSolution*(msg: string): NNAmbiguousSolution =
  var e = NNAmbiguousSolution()
  e.msg = msg
  return e
proc newNNExceededMaxIterations*(msg: string): NNExceededMaxIterations =
  var e = NNExceededMaxIterations()
  e.msg = msg
  return e
proc newNNPowerIterationFailedConvergence*(numIterations: int): NNPowerIterationFailedConvergence =
  var e = NNPowerIterationFailedConvergence()
  e.numIterations = numIterations
  e.msg = fmt"power iteration failed to converge within {numIterations} iterations"
  return e

# -------------------------------------------------------------------
# (Undirected) Graph
# -------------------------------------------------------------------

proc newGraph*(): Graph =
  var g = Graph()
  g.adj = initTable[Node, HashSet[Node]]()
  return g
proc newGraph*(nodes: openArray[Node]): Graph =
  var g = Graph()
  g.adj = initTable[Node, HashSet[Node]]()
  for node in nodes:
    if node notin g.adj:
      if node == None:
        raise newNNError("None cannot be a node")
      g.adj[node] = initHashSet[Node]()
  return g
proc newGraph*(nodes: HashSet[Node]): Graph =
  var g = Graph()
  for node in nodes:
    if node notin g.adj:
      if node == None:
        raise newNNError("None cannot be a node")
      g.adj[node] = initHashSet[Node]()
  return g
proc newGraph*(edges: openArray[Edge]): Graph =
  var g = Graph()
  for edge in edges:
    var u = edge.u
    var v = edge.v
    if u notin g.adj:
      if u == None:
        raise newNNError("None cannot be a node")
      g.adj[u] = initHashSet[Node]()
    if v notin g.adj:
      if v == None:
        raise newNNError("None cannot be a node")
      g.adj[v] = initHashSet[Node]()
    g.adj[u].incl(v)
    g.adj[v].incl(u)
  return g
proc newGraph*(edges: HashSet[Edge]): Graph =
  var g = Graph()
  for edge in edges:
    var u = edge.u
    var v = edge.v
    if u notin g.adj:
      if u == None:
        raise newNNError("None cannot be a node")
      g.adj[u] = initHashSet[Node]()
    if v notin g.adj:
      if v == None:
        raise newNNError("None cannot be a node")
      g.adj[v] = initHashSet[Node]()
    g.adj[u].incl(v)
    g.adj[v].incl(u)
  return g

proc addNode*(g: Graph, node: Node) =
  if node notin g.adj:
    if node == None:
      raise newNNError("None cannot be a node")
    g.adj[node] = initHashSet[Node]()
proc addNodesFrom*(g: Graph, nodes: openArray[Node]) =
  for node in nodes:
    g.addNode(node)
proc addNodesFrom*(g: Graph, nodes: HashSet[Node]) =
  for node in nodes:
    g.addNode(node)

proc removeNode*(g: Graph, node: Node) =
  var neighbors: HashSet[Node]
  try:
    neighbors = g.adj[node]
    g.adj.del(node)
  except KeyError:
    raise newNNNodeNotFound(node)
  for neighbor in neighbors:
    g.adj[neighbor].excl(node)
proc removeNodesFrom*(g: Graph, nodes: openArray[Node]) =
  for node in nodes:
    g.removeNode(node)
proc removeNodesFrom*(g: Graph, nodes: HashSet[Node]) =
  for node in nodes:
    g.removeNode(node)

proc addEdge*(g: Graph, u, v: Node) =
  if u notin g.adj:
    if u == None:
      raise newNNError("None cannot be a node")
    g.adj[u] = initHashSet[Node]()
  if v notin g.adj:
    if v == None:
      raise newNNError("None cannot be a node")
    g.adj[v] = initHashSet[Node]()
  g.adj[u].incl(v)
  g.adj[v].incl(u)
proc addEdge*(g: Graph, edge: Edge) =
  g.addEdge(edge.u, edge.v)
proc addEdgesFrom*(g: Graph, edges: openArray[Edge]) =
  for edge in edges:
    g.addEdge(edge)
proc addEdgesFrom*(g: Graph, edges: HashSet[Edge]) =
  for edge in edges:
    g.addEdge(edge)

proc removeEdge*(g: Graph, u, v: Node) =
  try:
    var isMissing: bool
    isMissing = g.adj[u].missingOrExcl(v)
    if isMissing:
      raise newNNError(fmt"edge {u}-{v} is not in graph")
    if u != v:
      g.adj[v].excl(u)
  except KeyError:
    raise newNNError(fmt"edge {u}-{v} is not in grpah")
proc removeEdge*(g: Graph, edge: Edge) =
  g.removeEdge(edge.u, edge.v)
proc removeEdgesFrom*(g: Graph, edges: openArray[Edge]) =
  for edge in edges:
    g.removeEdge(edge)
proc removeEdgesFrom*(g: Graph, edges: HashSet[Edge]) =
  for edge in edges:
    g.removeEdge(edge)

proc clear*(g: Graph) =
  g.adj.clear()
proc clearNodes*(g: Graph) =
  g.adj.clear()
proc clearEdges*(g: Graph) =
  for node in g.adj.keys():
    g.adj[node].clear()

proc nodes*(g: Graph): seq[Node] =
  var ret = newSeq[Node]()
  for node in g.adj.keys():
    ret.add(node)
  ret.sort()
  return ret
proc nodesIterator*(g: Graph): iterator: Node =
  return iterator: Node =
    for node in g.nodes():
      yield node
proc nodesSet*(g: Graph): HashSet[Node] =
  var ret = initHashSet[Node]()
  for node in g.adj.keys():
    ret.incl(node)
  return ret
proc nodesSeq*(g: Graph): seq[Node] =
  return g.nodes()
iterator nodes*(g: Graph): Node =
  var nodes = g.nodesSeq()
  for node in nodes:
    yield node

proc hasNode*(g: Graph, node: Node): bool =
  return node in g.adj
proc contains*(g: Graph, node: Node): bool =
  return node in g.adj

proc edges*(g: Graph): seq[Edge] =
  var ret = newSeq[Edge]()
  for (u, vs) in g.adj.pairs():
    for v in vs:
      if u <= v:
        ret.add((u, v))
  ret.sort()
  return ret
proc edgesIterator*(g: Graph): iterator: Edge =
  return iterator: Edge =
    for (u, v) in g.edges():
      yield (u, v)
proc edgesSet*(g: Graph): HashSet[Edge] =
  var ret = initHashSet[Edge]()
  for (u, vs) in g.adj.pairs():
    for v in vs:
      if u <= v:
        ret.incl((u, v))
  return ret
proc edgesSeq*(g: Graph): seq[Edge] =
  return g.edges()
iterator edges*(g: Graph): Edge =
  var edges = g.edgesSeq()
  for edge in edges:
    yield edge

proc edges*(g: Graph, node: Node): seq[Edge] =
  var ret = newSeq[Edge]()
  for v in g.adj[node]:
    ret.add((node, v))
  ret.sort()
  return ret
proc edgesIterator*(g: Graph, node: Node): iterator: Edge =
  return iterator: Edge =
    for edge in g.edges(node):
      yield edge
proc edgesSet*(g: Graph, node: Node): HashSet[Edge] =
  var ret = initHashSet[Edge]()
  for v in g.adj[node]:
    ret.incl((node, v))
  return ret
proc edgesSeq*(g: Graph, node: Node): seq[Edge] =
  return g.edges(node)
iterator edges*(g: Graph, node: Node): Edge =
  var edges = g.edgesSeq(node)
  for edge in edges:
    yield edge

proc hasEdge*(g: Graph, u, v: Node): bool =
  try:
    return v in g.adj[u]
  except KeyError:
    return false
proc hasEdge*(g: Graph, edge: Edge): bool =
  return g.hasEdge(edge.u, edge.v)
proc contains*(g: Graph, u, v: Node): bool =
  return g.hasEdge(u, v)
proc contains*(g: Graph, edge: Edge): bool =
  return g.hasEdge(edge)

proc adj*(g: Graph): Table[Node, HashSet[Node]] =
  return g.adj
proc adjacency*(g: Graph): seq[tuple[node: Node, adjacentNodes: seq[Node]]] =
  var ret = newSeq[tuple[node: Node, adjacentNodes: seq[Node]]]()
  for node in g.nodes():
    var neighbors = g.adj[node].toSeq()
    neighbors.sort()
    ret.add((node, neighbors))
  return ret
proc adjacencyIterator*(g: Graph): iterator: tuple[node: Node, adjacentNodes: seq[Node]] =
  return iterator: tuple[node: Node, adjacentNodes: seq[Node]] =
    for tpl in g.adjacency():
      yield tpl
proc adjacencySet*(g: Graph): HashSet[tuple[node: Node, adjacentNodes: seq[Node]]] =
  var ret = initHashSet[tuple[node: Node, adjacentNodes: seq[Node]]]()
  for tpl in g.adjacency():
    ret.incl(tpl)
  return ret
proc adjacencySeq*(g: Graph): seq[tuple[node: Node, adjacentNodes: seq[Node]]] =
  return g.adjacency()
iterator adjacency*(g: Graph): tuple[node: Node, adjacentNodes: seq[Node]] =
  var adjacency = g.adjacencySeq()
  for tpl in adjacency:
    yield tpl

proc neighbors*(g: Graph, node: Node): seq[Node] =
  var ret = newSeq[Node]()
  if node notin g.adj:
    raise newNNNodeNotFound(node)
  for neighbor in g.adj[node]:
    ret.add(neighbor)
  ret.sort()
  return ret
proc neighborsIterator*(g: Graph, node: Node): iterator: Node =
  return iterator: Node =
    for neighbor in g.neighbors(node):
      yield neighbor
proc neighborsSet*(g: Graph, node: Node): HashSet[Node] =
  var ret = initHashSet[Node]()
  if node notin g.adj:
    raise newNNNodeNotFound(node)
  for neighbor in g.adj[node]:
    ret.incl(neighbor)
  return ret
proc neighborsSeq*(g: Graph, node: Node): seq[Node] =
  return g.neighbors(node)
iterator neighbors*(g: Graph, node: Node): Node =
  var neighbors = g.neighborsSeq(node)
  for neighbor in neighbors:
    yield neighbor

proc order*(g: Graph): int =
  return g.adj.len()
proc numberOfNodes*(g: Graph): int =
  return g.adj.len()
proc len*(g: Graph): int =
  return g.adj.len()

proc size*(g: Graph): int =
  return len(g.edgesSet())
proc numberOfEdges*(g: Graph): int =
  return len(g.edgesSet())
proc numberOfSelfloop*(g :Graph): int =
  var ret = 0
  for node in g.adj.keys():
    if node in g.adj[node]:
      ret += 1
  return ret

proc degree*(g: Graph): Table[Node, int] =
  var ret = initTable[Node, int]()
  for node in g.adj.keys():
    ret[node] = g.adj[node].len()
  return ret
proc degree*(g: Graph, node: Node): int =
  return g.adj[node].len()
proc degree*(g: Graph, nodes: seq[Node]): Table[Node, int] =
  var ret = initTable[Node, int]()
  for node in nodes:
    ret[node] = g.adj[node].len()
  return ret
proc degree*(g: Graph, nodes: HashSet[Node]): Table[Node, int] =
  var ret = initTable[Node, int]()
  for node in nodes:
    ret[node] = g.adj[node].len()
  return ret
proc degreeHistogram*(g: Graph): seq[int] =
  var counts = initTable[int, int]()
  var maxDegree = 0
  for degree in g.degree().values():
    maxDegree = max(maxDegree, degree)
    if degree notin counts:
      counts[degree] = 1
    else:
      counts[degree] += 1
  var ret = newSeq[int](maxDegree+1)
  for (degree, freq) in counts.pairs():
    ret[degree] = freq
  return ret

proc density*(g: Graph): float =
  var n = g.numberOfNodes()
  var m = g.numberOfEdges()
  if m == 0 or n <= 1:
    return 0.0
  return (2 * m).float / (n*(n - 1)).float

proc nodeSubgraph*(g: Graph, nodes: HashSet[Node]): Graph =
  var ret = newGraph()
  for u in nodes:
    for v in g.adj[u]:
      if v in nodes:
        ret.addEdge((u, v))
  return ret
proc nodeSubgraph*(g: Graph, nodes: seq[Node]): Graph =
  var ret = newGraph()
  let nodesSet = nodes.toHashSet()
  for u in nodes:
    for v in g.adj[u]:
      if v in nodesSet:
        ret.addEdge((u, v))
  return ret
proc edgeSubgraph*(g: Graph, edges: HashSet[Edge]): Graph =
  var ret = newGraph()
  for edge in edges:
    if g.hasEdge(edge):
      ret.addEdge(edge)
  return ret
proc edgeSubgraph*(g: Graph, edges: seq[Edge]): Graph =
  var ret = newGraph()
  for edge in edges:
    if g.hasEdge(edge):
      ret.addEdge(edge)
  return ret
proc subgraph*(g: Graph, nodes: HashSet[Node]): Graph =
  return g.nodeSubgraph(nodes)
proc subgraph*(g: Graph, nodes: seq[Node]): Graph =
  return g.nodeSubgraph(nodes)

proc info*(g :Graph): string =
  var ret = "type: undirected graph"
  ret = ret & "\n"
  ret = ret & fmt"#nodes: {g.numberOfNodes()}"
  ret = ret & "\n"
  ret = ret & fmt"#edges: {g.numberOfEdges()}"
  return ret
proc info*(g: Graph, node: Node): string =
  if node notin g.adj:
    raise newNNNodeNotFound(node)
  var ret = fmt"node {node} has following properties:"
  ret = ret & "\n"
  ret = ret & fmt"degree: {g.degree(node)}"
  ret = ret & "\n"
  ret = ret & fmt"neighbors: {g.neighbors(node)}"
  return ret

proc addStar*(g: Graph, nodes: seq[Node]) =
  if len(nodes) < 1:
    return
  let centerNode = nodes[0]
  g.addNode(centerNode)
  for i in 1..<len(nodes):
    g.addEdge(centerNode, nodes[i])
proc isStar*(g: Graph, nodes: seq[Node]): bool =
  if len(nodes) < 1:
    return true
  let centerNode = nodes[0]
  if centerNode notin g.adj:
    return false
  for i in 1..<len(nodes):
    if nodes[i] notin g.adj[centerNode]:
      return false
  return true
proc addPath*(g: Graph, nodes: seq[Node]) =
  if len(nodes) < 1:
    return
  for i in 0..<len(nodes)-1:
    g.addEdge(nodes[i], nodes[i+1])
proc isPath*(g: Graph, nodes: seq[Node]): bool =
  if len(nodes) < 1:
    return true
  for i in 0..<len(nodes)-1:
    if nodes[i] notin g.adj:
      return false
    if nodes[i+1] notin g.adj[nodes[i]]:
      return false
  return true
proc addCycle*(g: Graph, nodes: seq[Node]) =
  if len(nodes) < 1:
    return
  for i in 0..<len(nodes)-1:
    g.addEdge(nodes[i], nodes[i+1])
  g.addEdge(nodes[0], nodes[^1])
proc isCycle*(g: Graph, nodes: seq[Node]): bool =
  if len(nodes) < 1:
    return true
  for i in 0..<len(nodes)-1:
    if nodes[i] notin g.adj:
      return false
    if nodes[i+1] notin g.adj[nodes[i]]:
      return false
  return nodes[^1] in g.adj[nodes[0]]

proc nonNeighbors*(g: Graph, node: Node): seq[Node] =
  var neighbors = g.neighborsSet(node)
  neighbors.incl(node)
  var nonNeighbors = g.nodesSet() - neighbors
  var ret = nonNeighbors.toSeq()
  ret.sort()
  return ret
proc nonNeighborsIterator*(g: Graph, node: Node): iterator: Node =
  var neighbors = g.neighborsSet(node)
  neighbors.incl(node)
  var nonNeighbors = g.nodesSet() - neighbors
  var ret = nonNeighbors.toSeq()
  ret.sort()
  return iterator: Node =
    for nonNeighbor in ret:
      yield nonNeighbor
proc nonNeighborsSet*(g: Graph, node: Node): HashSet[Node] =
  var neighbors = g.neighborsSet(node)
  neighbors.incl(node)
  var nonNeighbors = g.nodesSet() - neighbors
  return nonNeighbors
proc nonNeighborsSeq*(g: Graph, node: Node): seq[Node] =
  return g.nonNeighbors(node)
iterator nonNeighbors*(g: Graph, node: Node): Node =
  var nonNeighbors = g.nonNeighbors(node)
  for nonNeighbor in nonNeighbors:
    yield nonNeighbor

proc commonNeighbors*(g: Graph, u, v: Node): seq[Node] =
  var commonNeighbors = initHashSet[Node]()
  let neighborsU = g.neighbors(u)
  let neighborsV = g.neighbors(v)
  for nbr in neighborsU:
    if (nbr in neighborsV) and (nbr != u) and (nbr != v):
      commonNeighbors.incl(nbr)
  var ret = commonNeighbors.toSeq()
  ret.sort()
  return ret
proc commonNeighborsIterator*(g: Graph, u, v: Node): iterator: Node =
  var commonNeighbors = initHashSet[Node]()
  let neighborsU = g.neighbors(u)
  let neighborsV = g.neighbors(v)
  for nbr in neighborsU:
    if (nbr in neighborsV) and (nbr != u) and (nbr != v):
      commonNeighbors.incl(nbr)
  var ret = commonNeighbors.toSeq()
  ret.sort()
  return iterator: Node =
    for commonNeighbor in ret:
      yield commonNeighbor
proc commonNeighborsSet*(g: Graph, u, v: Node): HashSet[Node] =
  var commonNeighbors = initHashSet[Node]()
  let neighborsU = g.neighbors(u)
  let neighborsV = g.neighbors(v)
  for nbr in neighborsU:
    if (nbr in neighborsV) and (nbr != u) and (nbr != v):
      commonNeighbors.incl(nbr)
  return commonNeighbors
proc commonNeighborsSeq*(g: Graph, u, v: Node): seq[Node] =
  return g.commonNeighbors(u, v)
iterator commonNeighbors*(g: Graph, u, v: Node): Node =
  var commonNeighbors = g.commonNeighbors(u, v)
  for commonNeighbor in commonNeighbors:
    yield commonNeighbor

proc nonEdges*(g: Graph): seq[Edge] =
  var nodes = g.nodes()
  var nonEdges = newSeq[Edge]()
  for node in nodes:
    for nonNbr in g.nonNeighbors(node):
      if node <= nonNbr:
        nonEdges.add((node, nonNbr))
  nonEdges.sort()
  return nonEdges
proc nonEdgesIterator*(g: Graph): iterator: Edge =
  var nodes = g.nodes()
  var nonEdges = newSeq[Edge]()
  for node in nodes:
    for nonNbr in g.nonNeighbors(node):
      if node <= nonNbr:
        nonEdges.add((node, nonNbr))
  nonEdges.sort()
  return iterator: Edge =
    for nonEdge in nonEdges:
      yield nonEdge
proc nonEdgesSet*(g: Graph): HashSet[Edge] =
  var nodes = g.nodes()
  var nonEdges = initHashSet[Edge]()
  for node in nodes:
    for nonNbr in g.nonNeighbors(node):
      if node <= nonNbr:
        nonEdges.incl((node, nonNbr))
  return nonEdges
proc nonEdgesSeq*(g: Graph): seq[Edge] =
  return g.nonEdges()
iterator nonEdges*(g: Graph): Edge =
  var nodes = g.nodes()
  var nonEdges = newSeq[Edge]()
  for node in nodes:
    for nonNbr in g.nonNeighbors(node):
      if node <= nonNbr:
        nonEdges.add((node, nonNbr))
  nonEdges.sort()
  for nonEdge in nonEdges:
    yield nonEdge

proc nodesWithSelfloopEdge*(g: Graph): seq[Node] =
  var ret = newSeq[Node]()
  for node in g.adj.keys():
    if node in g.adj[node]:
      ret.add(node)
  ret.sort()
  return ret
proc nodesWithSelfloopEdgeIterator*(g: Graph): iterator: Node =
  var ret = newSeq[Node]()
  for node in g.adj.keys():
    if node in g.adj[node]:
      ret.add(node)
  ret.sort()
  return iterator: Node =
    for node in ret:
      yield node
proc nodesWithSelfloopEdgeSet*(g: Graph): HashSet[Node] =
  var ret = initHashSet[Node]()
  for node in g.adj.keys():
    if node in g.adj[node]:
      ret.incl(node)
  return ret
proc nodesWithSelfloopEdgeSeq*(g: Graph): seq[Node] =
  return g.nodesWithSelfloopEdge()
iterator nodesWithSelfloopEdge*(g: Graph): Node =
  var ret = newSeq[Node]()
  for node in g.adj.keys():
    if node in g.adj[node]:
      ret.add(node)
  ret.sort()
  for node in ret:
    yield node

proc selfloopEdges*(g: Graph): seq[Edge] =
  var ret = newSeq[Edge]()
  for node in g.nodesWithSelfloopEdge():
    ret.add((node, node))
  return ret
proc selfloopEdgesIterator*(g: Graph): iterator: Edge =
  var ret = newSeq[Edge]()
  for node in g.nodesWithSelfloopEdge():
    ret.add((node, node))
  return iterator: Edge =
    for edge in ret:
      yield edge
proc selfloopEdgesSet*(g: Graph): HashSet[Edge] =
  var ret = initHashSet[Edge]()
  for node in g.nodesWithSelfloopEdge():
    ret.incl((node, node))
  return ret
proc selfloopEdgesSeq*(g: Graph): seq[Edge] =
  return g.selfloopEdges()
iterator selfloopEdges*(g: Graph): Edge =
  var ret = newSeq[Edge]()
  for node in g.nodesWithSelfloopEdge():
    ret.add((node, node))
  for edge in ret:
    yield edge

proc isDirected*(g: Graph): bool =
  return false

proc `[]`*(g: Graph, node: Node): seq[Node] =
  return g.neighbors(node)

proc `in`*(g: Graph, node: Node): bool =
  return g.contains(node)
proc `notin`*(g: Graph, node: Node): bool =
  return not g.contains(node)
proc `+`*(g: Graph, node: Node): Graph =
  g.addNode(node)
  return g
proc `+`*(g: Graph, nodes: openArray[Node]): Graph =
  g.addNodesFrom(nodes)
  return g
proc `+`*(g: Graph, nodes: HashSet[Node]): Graph =
  g.addNodesFrom(nodes)
  return g
proc `-`*(g: Graph, node: Node): Graph =
  g.removeNode(node)
  return g
proc `-`*(g: Graph, nodes: openArray[Node]): Graph =
  g.removeNodesFrom(nodes)
  return g
proc `-`*(g: Graph, nodes: HashSet[Node]): Graph =
  g.removeNodesFrom(nodes)
  return g
proc `+=`*(g: Graph, node: Node) =
    g.addNode(node)
proc `+=`*(g: Graph, nodes: HashSet[Node]) =
    g.addNodesFrom(nodes)
proc `+=`*(g: Graph, nodes: openArray[Node]) =
    g.addNodesFrom(nodes)
proc `-=`*(g: Graph, node: Node) =
    g.removeNode(node)
proc `-=`*(g: Graph, nodes: HashSet[Node]) =
    g.removeNodesFrom(nodes)
proc `-=`*(g: Graph, nodes: openArray[Node]) =
    g.removeNodesFrom(nodes)

proc `in`*(g: Graph, edge: Edge): bool =
    return g.hasEdge(edge)
proc `+`*(g: Graph, edge: Edge): Graph =
    g.addEdge(edge)
    return g
proc `+`*(g: Graph, edges: HashSet[Edge]): Graph =
    g.addEdgesFrom(edges)
    return g
proc `+`*(g: Graph, edges: openArray[Edge]): Graph =
    g.addEdgesFrom(edges)
    return g
proc `-`*(g: Graph, edge: Edge): Graph =
    g.removeEdge(edge)
    return g
proc `-`*(g: Graph, edges: HashSet[Edge]): Graph =
    g.removeEdgesFrom(edges)
    return g
proc `-`*(g: Graph, edges: openArray[Edge]): Graph =
    g.removeEdgesFrom(edges)
    return g
proc `+=`*(g: Graph, edge: Edge) =
    g.addEdge(edge)
proc `+=`*(g: Graph, edges: HashSet[Edge]) =
    g.addEdgesFrom(edges)
proc `+=`*(g: Graph, edges: openArray[Edge]) =
    g.addEdgesFrom(edges)
proc `-=`*(g: Graph, edge: Edge) =
    g.removeEdge(edge)
proc `-=`*(g: Graph, edges: HashSet[Edge]) =
    g.removeEdgesFrom(edges)
proc `-=`*(g: Graph, edges: openArray[Edge]) =
    g.removeEdgesFrom(edges)

# -------------------------------------------------------------------
# Directed Graph
# -------------------------------------------------------------------

proc newDiGraph*(): DiGraph =
  var dg = DiGraph()
  dg.pred = initTable[Node, HashSet[Node]]()
  dg.succ = initTable[Node, HashSet[Node]]()
  return dg
proc newDiGraph*(nodes: openArray[Node]): DiGraph =
  var dg = DiGraph()
  dg.pred = initTable[Node, HashSet[Node]]()
  dg.succ = initTable[Node, HashSet[Node]]()
  for node in nodes:
    if node notin dg.succ:
      if node == None:
        raise newNNError("None cannot be a node")
      dg.pred[node] = initHashSet[Node]()
      dg.succ[node] = initHashSet[Node]()
  return dg
proc newDiGraph*(nodes: HashSet[Node]): DiGraph =
  var dg = DiGraph()
  dg.pred = initTable[Node, HashSet[Node]]()
  dg.succ = initTable[Node, HashSet[Node]]()
  for node in nodes:
    if node notin dg.succ:
      if node == None:
        raise newNNError("None cannot be a node")
      dg.pred[node] = initHashSet[Node]()
      dg.succ[node] = initHashSet[Node]()
  return dg
proc newDiGraph*(edges: openArray[Edge]): DiGraph =
  var dg = DiGraph()
  for edge in edges:
    var u = edge.u
    var v = edge.v
    if u notin dg.succ:
      if u == None:
        raise newNNError("None cannot be a node")
      dg.succ[u] = initHashSet[Node]()
      dg.pred[u] = initHashSet[Node]()
    if v notin dg.succ:
      if v == None:
        raise newNNError("None cannot be a node")
      dg.succ[v] = initHashSet[Node]()
      dg.pred[v] = initHashSet[Node]()
    dg.succ[u].incl(v)
    dg.pred[v].incl(u)
  return dg
proc newDiGraph*(edges: HashSet[Edge]): DiGraph =
  var dg = DiGraph()
  for edge in edges:
    var u = edge.u
    var v = edge.v
    if u notin dg.succ:
      if u == None:
        raise newNNError("None cannot be a node")
      dg.succ[u] = initHashSet[Node]()
      dg.pred[u] = initHashSet[Node]()
    if v notin dg.succ:
      if v == None:
        raise newNNError("None cannot be a node")
      dg.succ[v] = initHashSet[Node]()
      dg.pred[v] = initHashSet[Node]()
    dg.succ[u].incl(v)
    dg.pred[v].incl(u)
  return dg

proc addNode*(dg: DiGraph, node: Node) =
  if node notin dg.succ:
    if node == None:
      raise newNNError("None cannot be a node")
    dg.succ[node] = initHashSet[Node]()
    dg.pred[node] = initHashSet[Node]()
proc addNodesFrom*(dg: DiGraph, nodes: openArray[Node]) =
  for node in nodes:
    dg.addNode(node)
proc addNodesFrom*(dg: DiGraph, nodes: HashSet[Node]) =
  for node in nodes:
    dg.addNode(node)

proc removeNode*(dg: DiGraph, node: Node) =
  var predecessors: HashSet[Node]
  try:
    predecessors = dg.pred[node]
    dg.succ.del(node)
  except KeyError:
    raise newNNNodeNotFound(node)
  for predecessor in predecessors:
    dg.pred[predecessor].excl(node)
proc removeNodesFrom*(dg: DiGraph, nodes: openArray[Node]) =
  for node in nodes:
    dg.removeNode(node)
proc removeNodesFrom*(dg: DiGraph, nodes: HashSet[Node]) =
  for node in nodes:
    dg.removeNode(node)

proc addEdge*(dg: DiGraph, u, v: Node) =
  if u notin dg.succ:
    if u == None:
      raise newNNError("None cannot be a node")
    dg.succ[u] = initHashSet[Node]()
    dg.pred[u] = initHashSet[Node]()
  if v notin dg.succ:
    if v == None:
      raise newNNError("None cannot be a node")
    dg.succ[v] = initHashSet[Node]()
    dg.pred[v] = initHashSet[Node]()
  dg.succ[u].incl(v)
  dg.pred[v].incl(u)
proc addEdge*(dg: DiGraph, edge: Edge) =
  dg.addEdge(edge.u, edge.v)
proc addEdgesFrom*(dg: DiGraph, edges: openArray[Edge]) =
  for edge in edges:
    dg.addEdge(edge)
proc addEdgesFrom*(dg: DiGraph, edges: HashSet[Edge]) =
  for edge in edges:
    dg.addEdge(edge)

proc removeEdge*(dg: DiGraph, u, v: Node) =
  try:
    var isMissing: bool
    isMissing = dg.succ[u].missingOrExcl(v)
    if isMissing:
      raise newNNError(fmt"edge {u}-{v} is not in graph")
    dg.pred[v].excl(u)
  except KeyError:
    raise newNNError(fmt"edge {u}-{v} is not in grpah")
proc removeEdge*(dg: DiGraph, edge: Edge) =
  dg.removeEdge(edge.u, edge.v)
proc removeEdgesFrom*(dg: DiGraph, edges: openArray[Edge]) =
  for edge in edges:
    dg.removeEdge(edge)
proc removeEdgesFrom*(dg: DiGraph, edges: HashSet[Edge]) =
    for edge in edges:
      dg.removeEdge(edge)

proc clear*(dg: DiGraph) =
  dg.succ.clear()
  dg.pred.clear()
proc clearNodes*(dg: DiGraph) =
  dg.succ.clear()
  dg.pred.clear()
proc clearEdges*(dg: DiGraph) =
  for node in dg.succ.keys():
    dg.succ[node].clear()
    dg.pred[node].clear()

proc nodes*(dg: DiGraph): seq[Node] =
  var ret = newSeq[Node]()
  for node in dg.succ.keys():
    ret.add(node)
  ret.sort()
  return ret
proc nodesIterator*(dg: DiGraph): iterator: Node =
  return iterator: Node =
    for node in dg.nodes():
      yield node
proc nodesSet*(dg: DiGraph): HashSet[Node] =
  var ret = initHashSet[Node]()
  for node in dg.succ.keys():
    ret.incl(node)
  return ret
proc nodesSeq*(dg: DiGraph): seq[Node] =
  return dg.nodes()
iterator nodes*(dg: DiGraph): Node =
  var nodes = dg.nodesSeq()
  for node in nodes:
    yield node

proc hasNode*(dg: DiGraph, node: Node): bool =
  return node in dg.succ
proc contains*(dg: DiGraph, node: Node): bool =
  return node in dg.succ

proc edges*(dg: DiGraph): seq[Edge] =
  var ret = newSeq[Edge]()
  for (u, vs) in dg.succ.pairs():
    for v in vs:
      ret.add((u, v))
  ret.sort()
  return ret
proc edgesIterator*(dg: DiGraph): iterator: Edge =
  return iterator: Edge =
    for (u, v) in dg.edges():
      yield (u, v)
proc edgesSet*(dg: DiGraph): HashSet[Edge] =
  var ret = initHashSet[Edge]()
  for (u, vs) in dg.succ.pairs():
    for v in vs:
      ret.incl((u, v))
  return ret
proc edgesSeq*(dg: DiGraph): seq[Edge] =
  return dg.edges()
iterator edges*(dg: DiGraph): Edge =
  var edges = dg.edgesSeq()
  for edge in edges:
    yield edge

proc edges*(dg: DiGraph, node: Node): seq[Edge] =
  var ret = newSeq[Edge]()
  for v in dg.succ[node]:
    ret.add((node, v))
  ret.sort()
  return ret
proc edgesIterator*(dg: DiGraph, node: Node): iterator: Edge =
  return iterator: Edge =
    for edge in dg.edges(node):
      yield edge
proc edgesSet*(dg: DiGraph, node: Node): HashSet[Edge] =
  var ret = initHashSet[Edge]()
  for v in dg.succ[node]:
    ret.incl((node, v))
  return ret
proc edgesSeq*(dg: DiGraph, node: Node): seq[Edge] =
  return dg.edges(node)
iterator edges*(dg: DiGraph, node: Node): Edge =
  var edges = dg.edgesSeq(node)
  for edge in edges:
    yield edge

proc hasEdge*(dg: DiGraph, u, v: Node): bool =
  try:
    return v in dg.succ[u]
  except KeyError:
    return false
proc hasEdge*(dg: DiGraph, edge: Edge): bool =
  return dg.hasEdge(edge.u, edge.v)
proc contains*(dg: DiGraph, u, v: Node): bool =
  return dg.hasEdge(u, v)
proc contains*(dg: DiGraph, edge: Edge): bool =
  return dg.hasEdge(edge)

proc pred*(dg: DiGraph): Table[Node, HashSet[Node]] =
  return dg.pred
proc predecence*(dg: DiGraph): seq[tuple[node: Node, predecentNodes: seq[Node]]] =
  var ret = newSeq[tuple[node: Node, predecentNodes: seq[Node]]]()
  for node in dg.nodes():
    var predecessors = dg.pred[node].toSeq()
    predecessors.sort()
    ret.add((node, predecessors))
  return ret
proc predecenceIterator*(dg: DiGraph): iterator: tuple[node: Node, predecentNodes: seq[Node]] =
  return iterator: tuple[node: Node, predecentNodes: seq[Node]] =
    for tpl in dg.predecence():
      yield tpl
proc predecenceSet*(dg: DiGraph): HashSet[tuple[node: Node, predecentNodes: seq[Node]]] =
  var ret = initHashSet[tuple[node: Node, predecentNodes: seq[Node]]]()
  for tpl in dg.predecence():
    ret.incl(tpl)
  return ret
proc predecenceSeq*(dg: DiGraph): seq[tuple[node: Node, predecentNodes: seq[Node]]] =
  return dg.predecence()
iterator predecence*(dg: DiGraph): tuple[node: Node, predecentNodes: seq[Node]] =
  var predecence = dg.predecenceSeq()
  for tpl in predecence:
    yield tpl

proc predecessors*(dg: DiGraph, node: Node): seq[Node] =
  var ret = newSeq[Node]()
  if node notin dg.pred:
    raise newNNNodeNotFound(node)
  for predecessor in dg.pred[node]:
    ret.add(predecessor)
  ret.sort()
  return ret
proc predecessorsIterator*(dg: DiGraph, node: Node): iterator: Node =
  return iterator: Node =
    for predecessor in dg.predecessors(node):
      yield predecessor
proc predecessorsSet*(dg: DiGraph, node: Node): HashSet[Node] =
  var ret = initHashSet[Node]()
  if node notin dg.pred:
    raise newNNNodeNotFound(node)
  for predecessor in dg.pred[node]:
    ret.incl(predecessor)
  return ret
proc predecessorsSeq*(dg: DiGraph, node: Node): seq[Node] =
  return dg.predecessors(node)
iterator predecessors*(dg: DiGraph, node: Node): Node =
  var predecessors = dg.predecessorsSeq(node)
  for predecessor in predecessors:
    yield predecessor

proc succ*(dg: DiGraph): Table[Node, HashSet[Node]] =
  return dg.succ
proc succession*(dg: DiGraph): seq[tuple[node: Node, succeedingNodes: seq[Node]]] =
  var ret = newSeq[tuple[node: Node, succeedingNodes: seq[Node]]]()
  for node in dg.nodes():
    var successors = dg.succ[node].toSeq()
    successors.sort()
    ret.add((node, successors))
  return ret
proc successionIterator*(dg: DiGraph): iterator: tuple[node: Node, succeedingNodes: seq[Node]] =
  return iterator: tuple[node: Node, succeedingNodes: seq[Node]] =
    for tpl in dg.succession():
      yield tpl
proc successionSet*(dg: DiGraph): HashSet[tuple[node: Node, succeedingNodes: seq[Node]]] =
  var ret = initHashSet[tuple[node: Node, succeedingNodes: seq[Node]]]()
  for tpl in dg.succession():
    ret.incl(tpl)
  return ret
proc successionSeq*(dg: DiGraph): seq[tuple[node: Node, succeedingNodes: seq[Node]]] =
  return dg.succession()
iterator succession*(dg: DiGraph): tuple[node: Node, succeedingNodes: seq[Node]] =
  var succession = dg.successionSeq()
  for tpl in succession:
    yield tpl

proc successors*(dg: DiGraph, node: Node): seq[Node] =
  var ret = newSeq[Node]()
  if node notin dg.succ:
    raise newNNNodeNotFound(node)
  for successor in dg.succ[node]:
    ret.add(successor)
  ret.sort()
  return ret
proc successorsIterator*(dg: DiGraph, node: Node): iterator: Node =
  return iterator: Node =
    for successor in dg.successors(node):
      yield successor
proc successorsSet*(dg: DiGraph, node: Node): HashSet[Node] =
  var ret = initHashSet[Node]()
  if node notin dg.succ:
    raise newNNNodeNotFound(node)
  for successor in dg.succ[node]:
    ret.incl(successor)
  return ret
proc successorsSeq*(dg: DiGraph, node: Node): seq[Node] =
  return dg.successors(node)
iterator successors*(dg: DiGraph, node: Node): Node =
  var successors = dg.successorsSeq(node)
  for successor in successors:
    yield successor

proc neighbors*(dg: DiGraph, node: Node): seq[Node] =
  let predecessors = dg.predecessors(node)
  let successors = dg.successors(node)
  var neighbors = concat(predecessors, successors)
  neighbors.sort()
  return neighbors
proc neighborsIterator*(dg: DiGraph, node: Node): iterator: Node =
  let predecessors = dg.predecessors(node)
  let successors = dg.successors(node)
  var neighbors = concat(predecessors, successors)
  neighbors.sort()
  return iterator: Node =
    for neighbor in neighbors:
      yield neighbor
proc neighborsSet*(dg: DiGraph, node: Node): HashSet[Node] =
  let predecessors = dg.predecessorsSet(node)
  let successors = dg.successorsSet(node)
  let neighbors = union(predecessors, successors)
  return neighbors
proc neighborsSeq*(dg: DiGraph, node: Node): seq[Node] =
  return dg.neighbors(node)
iterator neighbors*(dg: DiGraph, node: Node): Node =
  var neighbors = dg.neighborsSeq(node)
  for neighbor in neighbors:
    yield neighbor

proc order*(dg: DiGraph): int =
  return dg.succ.len()
proc numberOfNodes*(dg: DiGraph): int =
  return dg.succ.len()
proc len*(dg: DiGraph): int =
  return dg.succ.len()

proc size*(dg: DiGraph): int =
  return len(dg.edgesSet())
proc numberOfEdges*(dg: DiGraph): int =
  return len(dg.edgesSet())
proc numberOfSelfloop*(dg: DiGraph): int =
  var ret = 0
  for node in dg.succ.keys():
    if node in dg.succ[node]:
      ret += 1
  return ret

proc inDegree*(dg: DiGraph): Table[Node, int] =
  var ret = initTable[Node, int]()
  for node in dg.pred.keys():
    ret[node] = dg.pred[node].len()
  return ret
proc inDegree*(dg: DiGraph, node: Node): int =
  return dg.pred[node].len()
proc inDegree*(dg: DiGraph, nodes: seq[Node]): Table[Node, int] =
  var ret = initTable[Node, int]()
  for node in nodes:
    ret[node] = dg.pred[node].len()
  return ret
proc inDegree*(dg: DiGraph, nodes: HashSet[Node]): Table[Node, int] =
  var ret = initTable[Node, int]()
  for node in nodes:
    ret[node] = dg.pred[node].len()
  return ret
proc inDegreeHistogram*(dg: DiGraph): seq[int] =
  var counts = initTable[int, int]()
  var maxInDegree = 0
  for inDegree in dg.inDegree().values():
    maxInDegree = max(maxInDegree, inDegree)
    if inDegree notin counts:
      counts[inDegree] = 1
    else:
      counts[inDegree] += 1
  var ret = newSeq[int](maxInDegree+1)
  for (inDegree, freq) in counts.pairs():
    ret[inDegree] = freq
  return ret

proc outDegree*(dg: DiGraph): Table[Node, int] =
  var ret = initTable[Node, int]()
  for node in dg.succ.keys():
    ret[node] = dg.succ[node].len()
  return ret
proc outDegree*(dg: DiGraph, node: Node): int =
  return dg.succ[node].len()
proc outDegree*(dg: DiGraph, nodes: seq[Node]): Table[Node, int] =
  var ret = initTable[Node, int]()
  for node in nodes:
    ret[node] = dg.succ[node].len()
  return ret
proc outDegree*(dg: DiGraph, nodes: HashSet[Node]): Table[Node, int] =
  var ret = initTable[Node, int]()
  for node in nodes:
    ret[node] = dg.succ[node].len()
  return ret
proc outDegreeHistogram*(dg: DiGraph): seq[int] =
  var counts = initTable[int, int]()
  var maxOutDegree = 0
  for outDegree in dg.outDegree().values():
    maxOutDegree = max(maxOutDegree, outDegree)
    if outDegree notin counts:
      counts[outDegree] = 1
    else:
      counts[outDegree] += 1
  var ret = newSeq[int](maxOutDegree+1)
  for (outDegree, freq) in counts.pairs():
    ret[outDegree] = freq
  return ret

proc density*(dg: DiGraph): float =
  var n = dg.numberOfNodes()
  var m = dg.numberOfEdges()
  if m == 0 or n <= 1:
    return 0.0
  return m.float / (n*(n - 1)).float

proc nodeSubgraph*(dg: DiGraph, nodes: HashSet[Node]): DiGraph =
  var ret = newDiGraph()
  for u in nodes:
    for v in dg.succ[u]:
      if v in nodes:
        ret.addEdge((u, v))
  return ret
proc nodeSubgraph*(dg: DiGraph, nodes: seq[Node]): DiGraph =
  var ret = newDiGraph()
  let nodesSet = nodes.toHashSet()
  for u in nodes:
    for v in dg.succ[u]:
      if v in nodesSet:
        ret.addEdge((u, v))
  return ret
proc edgeSubgraph*(dg: DiGraph, edges: HashSet[Edge]): DiGraph =
  var ret = newDiGraph()
  for edge in edges:
    if dg.hasEdge(edge):
      ret.addEdge(edge)
  return ret
proc edgeSubgraph*(dg: DiGraph, edges: seq[Edge]): DiGraph =
  var ret = newDiGraph()
  for edge in edges:
    if dg.hasEdge(edge):
      ret.addEdge(edge)
  return ret
proc subgraph*(dg: DiGraph, nodes: HashSet[Node]): DiGraph =
  return dg.nodeSubgraph(nodes)
proc subgraph*(dg: DiGraph, nodes: seq[Node]): DiGraph =
  return dg.nodeSubgraph(nodes)

proc info*(dg :DiGraph): string =
  var ret = "type: directed graph"
  ret = ret & "\n"
  ret = ret & fmt"#nodes: {dg.numberOfNodes()}"
  ret = ret & "\n"
  ret = ret & fmt"#edges: {dg.numberOfEdges()}"
  return ret
proc info*(dg: DiGraph, node: Node): string =
  if node notin dg.succ:
    raise newNNNodeNotFound(node)
  var ret = fmt"node {node} has following properties:"
  ret = ret & "\n"
  ret = ret & fmt"in-degree: {dg.inDegree(node)}"
  ret = ret & "\n"
  ret = ret & fmt"predecessors: {dg.predecessors(node)}"
  ret = ret & "\n"
  ret = ret & fmt"out-degree: {dg.outDegree(node)}"
  ret = ret & "\n"
  ret = ret & fmt"successors: {dg.successors(node)}"
  return ret

proc addStar*(dg: DiGraph, nodes: seq[Node]) =
  if len(nodes) < 1:
    return
  let centerNode = nodes[0]
  dg.addNode(centerNode)
  for i in 1..<len(nodes):
    dg.addEdge(centerNode, nodes[i])
proc isStar*(dg: DiGraph, nodes: seq[Node]): bool =
  if len(nodes) < 1:
    return true
  let centerNode = nodes[0]
  if centerNode notin dg.succ:
    return false
  for i in 1..<len(nodes):
    if nodes[i] notin dg.succ[centerNode]:
      return false
  return true
proc addPath*(dg: DiGraph, nodes: seq[Node]) =
  if len(nodes) < 1:
    return
  for i in 0..<len(nodes)-1:
    dg.addEdge(nodes[i], nodes[i+1])
proc isPath*(dg: DiGraph, nodes: seq[Node]): bool =
  if len(nodes) < 1:
    return true
  for i in 0..<len(nodes)-1:
    if nodes[i] notin dg.succ:
      return false
    if nodes[i+1] notin dg.succ[nodes[i]]:
      return false
  return true
proc addCycle*(dg: DiGraph, nodes: seq[Node]) =
  if len(nodes) < 1:
    return
  for i in 0..<len(nodes)-1:
    dg.addEdge(nodes[i], nodes[i+1])
  dg.addEdge(nodes[0], nodes[^1])
proc isCycle*(dg: DiGraph, nodes: seq[Node]): bool =
  if len(nodes) < 1:
    return true
  for i in 0..<len(nodes)-1:
    if nodes[i] notin dg.succ:
      return false
    if nodes[i+1] notin dg.succ[nodes[i]]:
      return false
  return nodes[^1] in dg.succ[nodes[0]]

proc nonNeighbors*(dg: DiGraph, node: Node): seq[Node] =
  var neighbors = dg.neighborsSet(node)
  neighbors.incl(node)
  var nonNeighbors = dg.nodesSet() - neighbors
  var ret = nonNeighbors.toSeq()
  ret.sort()
  return ret
proc nonNeighborsIterator*(dg: DiGraph, node: Node): iterator: Node =
  var neighbors = dg.neighborsSet(node)
  neighbors.incl(node)
  var nonNeighbors = dg.nodesSet() - neighbors
  var ret = nonNeighbors.toSeq()
  ret.sort()
  return iterator: Node =
    for nonNeighbor in ret:
      yield nonNeighbor
proc nonNeighborsSet*(dg: DiGraph, node: Node): HashSet[Node] =
  var neighbors = dg.neighborsSet(node)
  neighbors.incl(node)
  var nonNeighbors = dg.nodesSet() - neighbors
  return nonNeighbors
proc nonNeighborsSeq*(dg: DiGraph, node: Node): seq[Node] =
  return dg.nonNeighbors(node)
iterator nonNeighbors*(dg: DiGraph, node: Node): Node =
  var nonNeighbors = dg.nonNeighbors(node)
  for nonNeighbor in nonNeighbors:
    yield nonNeighbor

proc nonPredecessors*(dg: DiGraph, node: Node): seq[Node] =
  var predecessors = dg.predecessorsSet(node)
  predecessors.incl(node)
  var nonPredecessors = dg.nodesSet() - predecessors
  var ret = nonPredecessors.toSeq()
  ret.sort()
  return ret
proc nonPredecessorsIterator*(dg: DiGraph, node: Node): iterator: Node =
  var predecessors = dg.predecessorsSet(node)
  predecessors.incl(node)
  var nonPredecessors = dg.nodesSet() - predecessors
  var ret = nonPredecessors.toSeq()
  ret.sort()
  return iterator: Node =
    for nonNeighbor in ret:
      yield nonNeighbor
proc nonPredecessorsSet*(dg: DiGraph, node: Node): HashSet[Node] =
  var predecessors = dg.predecessorsSet(node)
  predecessors.incl(node)
  var nonPredecessors = dg.nodesSet() - predecessors
  return nonPredecessors
proc nonPredecessorsSeq*(dg: DiGraph, node: Node): seq[Node] =
  return dg.nonPredecessors(node)
iterator nonPredecessors*(dg: DiGraph, node: Node): Node =
  var nonPredecessors = dg.nonPredecessors(node)
  for nonNeighbor in nonPredecessors:
    yield nonNeighbor

proc nonSuccessors*(dg: DiGraph, node: Node): seq[Node] =
  var successors = dg.successorsSet(node)
  successors.incl(node)
  var nonSuccessors = dg.nodesSet() - successors
  var ret = nonSuccessors.toSeq()
  ret.sort()
  return ret
proc nonSuccessorsIterator*(dg: DiGraph, node: Node): iterator: Node =
  var successors = dg.successorsSet(node)
  successors.incl(node)
  var nonSuccessors = dg.nodesSet() - successors
  var ret = nonSuccessors.toSeq()
  ret.sort()
  return iterator: Node =
    for nonNeighbor in ret:
      yield nonNeighbor
proc nonSuccessorsSet*(dg: DiGraph, node: Node): HashSet[Node] =
  var successors = dg.successorsSet(node)
  successors.incl(node)
  var nonSuccessors = dg.nodesSet() - successors
  return nonSuccessors
proc nonSuccessorsSeq*(dg: DiGraph, node: Node): seq[Node] =
  return dg.nonSuccessors(node)
iterator nonSuccessors*(dg: DiGraph, node: Node): Node =
  var nonSuccessors = dg.nonSuccessors(node)
  for nonNeighbor in nonSuccessors:
    yield nonNeighbor

proc commonNeighbors*(dg: DiGraph, u, v: Node): seq[Node] =
  var commonNeighbors = initHashSet[Node]()
  let neighborsU = dg.neighbors(u)
  let neighborsV = dg.neighbors(v)
  for nbr in neighborsU:
    if (nbr in neighborsV) and (nbr != u) and (nbr != v):
      commonNeighbors.incl(nbr)
  var ret = commonNeighbors.toSeq()
  ret.sort()
  return ret
proc commonNeighborsIterator*(dg: DiGraph, u, v: Node): iterator: Node =
  var commonNeighbors = initHashSet[Node]()
  let neighborsU = dg.neighbors(u)
  let neighborsV = dg.neighbors(v)
  for nbr in neighborsU:
    if (nbr in neighborsV) and (nbr != u) and (nbr != v):
      commonNeighbors.incl(nbr)
  var ret = commonNeighbors.toSeq()
  ret.sort()
  return iterator: Node =
    for commonNeighbor in ret:
      yield commonNeighbor
proc commonNeighborsSet*(dg: DiGraph, u, v: Node): HashSet[Node] =
  var commonNeighbors = initHashSet[Node]()
  let neighborsU = dg.neighbors(u)
  let neighborsV = dg.neighbors(v)
  for nbr in neighborsU:
    if (nbr in neighborsV) and (nbr != u) and (nbr != v):
      commonNeighbors.incl(nbr)
  return commonNeighbors
proc commonNeighborsSeq*(dg: DiGraph, u, v: Node): seq[Node] =
  return dg.commonNeighbors(u, v)
iterator commonNeighbors*(dg: DiGraph, u, v: Node): Node =
  var commonNeighbors = dg.commonNeighbors(u, v)
  for commonNeighbor in commonNeighbors:
    yield commonNeighbor

proc commonPredecessors*(dg: DiGraph, u, v: Node): seq[Node] =
  var commonPredecessors = initHashSet[Node]()
  let predecessorsU = dg.predecessors(u)
  let predecessorsV = dg.predecessors(v)
  for nbr in predecessorsU:
    if (nbr in predecessorsV) and (nbr != u) and (nbr != v):
      commonPredecessors.incl(nbr)
  var ret = commonPredecessors.toSeq()
  ret.sort()
  return ret
proc commonPredecessorsIterator*(dg: DiGraph, u, v: Node): iterator: Node =
  var commonPredecessors = initHashSet[Node]()
  let predecessorsU = dg.predecessors(u)
  let predecessorsV = dg.predecessors(v)
  for nbr in predecessorsU:
    if (nbr in predecessorsV) and (nbr != u) and (nbr != v):
      commonPredecessors.incl(nbr)
  var ret = commonPredecessors.toSeq()
  ret.sort()
  return iterator: Node =
    for commonNeighbor in ret:
      yield commonNeighbor
proc commonPredecessorsSet*(dg: DiGraph, u, v: Node): HashSet[Node] =
  var commonPredecessors = initHashSet[Node]()
  let predecessorsU = dg.predecessors(u)
  let predecessorsV = dg.predecessors(v)
  for nbr in predecessorsU:
    if (nbr in predecessorsV) and (nbr != u) and (nbr != v):
      commonPredecessors.incl(nbr)
  return commonPredecessors
proc commonPredecessorsSeq*(dg: DiGraph, u, v: Node): seq[Node] =
  return dg.commonPredecessors(u, v)
iterator commonPredecessors*(dg: DiGraph, u, v: Node): Node =
  var commonPredecessors = dg.commonPredecessors(u, v)
  for commonNeighbor in commonPredecessors:
    yield commonNeighbor

proc commonSuccessors*(dg: DiGraph, u, v: Node): seq[Node] =
  var commonSuccessors = initHashSet[Node]()
  let successorsU = dg.successors(u)
  let successorsV = dg.successors(v)
  for nbr in successorsU:
    if (nbr in successorsV) and (nbr != u) and (nbr != v):
      commonSuccessors.incl(nbr)
  var ret = commonSuccessors.toSeq()
  ret.sort()
  return ret
proc commonSuccessorsIterator*(dg: DiGraph, u, v: Node): iterator: Node =
  var commonSuccessors = initHashSet[Node]()
  let successorsU = dg.successors(u)
  let successorsV = dg.successors(v)
  for nbr in successorsU:
    if (nbr in successorsV) and (nbr != u) and (nbr != v):
      commonSuccessors.incl(nbr)
  var ret = commonSuccessors.toSeq()
  ret.sort()
  return iterator: Node =
    for commonNeighbor in ret:
      yield commonNeighbor
proc commonSuccessorsSet*(dg: DiGraph, u, v: Node): HashSet[Node] =
  var commonSuccessors = initHashSet[Node]()
  let successorsU = dg.successors(u)
  let successorsV = dg.successors(v)
  for nbr in successorsU:
    if (nbr in successorsV) and (nbr != u) and (nbr != v):
      commonSuccessors.incl(nbr)
  return commonSuccessors
proc commonSuccessorsSeq*(dg: DiGraph, u, v: Node): seq[Node] =
  return dg.commonSuccessors(u, v)
iterator commonSuccessors*(dg: DiGraph, u, v: Node): Node =
  var commonSuccessors = dg.commonSuccessors(u, v)
  for commonNeighbor in commonSuccessors:
    yield commonNeighbor

proc nonEdges*(dg: DiGraph): seq[Edge] =
  var nodes = dg.nodes()
  var nonEdges = newSeq[Edge]()
  for node in nodes:
    for nonSucc in dg.nonSuccessors(node):
      nonEdges.add((node, nonSucc))
  nonEdges.sort()
  return nonEdges
proc nonEdgesIterator*(dg: DiGraph): iterator: Edge =
  var nodes = dg.nodes()
  var nonEdges = newSeq[Edge]()
  for node in nodes:
    for nonSucc in dg.nonSuccessors(node):
      nonEdges.add((node, nonSucc))
  nonEdges.sort()
  return iterator: Edge =
    for nonEdge in nonEdges:
      yield nonEdge
proc nonEdgesSet*(dg: DiGraph): HashSet[Edge] =
  var nodes = dg.nodes()
  var nonEdges = initHashSet[Edge]()
  for node in nodes:
    for nonSucc in dg.nonSuccessors(node):
      nonEdges.incl((node, nonSucc))
  return nonEdges
proc nonEdgesSeq*(dg: DiGraph): seq[Edge] =
  return dg.nonEdges()
iterator nonEdges*(dg: DiGraph): Edge =
  var nodes = dg.nodes()
  var nonEdges = newSeq[Edge]()
  for node in nodes:
    for nonSucc in dg.nonSuccessors(node):
      nonEdges.add((node, nonSucc))
  nonEdges.sort()
  for nonEdge in nonEdges:
    yield nonEdge

proc nodesWithSelfloopEdge*(dg: DiGraph): seq[Node] =
  var ret = newSeq[Node]()
  for node in dg.succ.keys():
    if node in dg.succ[node]:
      ret.add(node)
  ret.sort()
  return ret
proc nodesWithSelfloopEdgeIterator*(dg: DiGraph): iterator: Node =
  var ret = newSeq[Node]()
  for node in dg.succ.keys():
    if node in dg.succ[node]:
      ret.add(node)
  ret.sort()
  return iterator: Node =
    for node in ret:
      yield node
proc nodesWithSelfloopEdgeSet*(dg: DiGraph): HashSet[Node] =
  var ret = initHashSet[Node]()
  for node in dg.succ.keys():
    if node in dg.succ[node]:
      ret.incl(node)
  return ret
proc nodesWithSelfloopEdgeSeq*(dg: DiGraph): seq[Node] =
  return dg.nodesWithSelfloopEdge()
iterator nodesWithSelfloopEdge*(dg: DiGraph): Node =
  var ret = newSeq[Node]()
  for node in dg.succ.keys():
    if node in dg.succ[node]:
      ret.add(node)
  ret.sort()
  for node in ret:
    yield node

proc selfloopEdges*(dg: DiGraph): seq[Edge] =
  var ret = newSeq[Edge]()
  for node in dg.nodesWithSelfloopEdge():
    ret.add((node, node))
  return ret
proc selfloopEdgesIterator*(dg: DiGraph): iterator: Edge =
  var ret = newSeq[Edge]()
  for node in dg.nodesWithSelfloopEdge():
    ret.add((node, node))
  return iterator: Edge =
    for edge in ret:
      yield edge
proc selfloopEdgesSet*(dg: DiGraph): HashSet[Edge] =
  var ret = initHashSet[Edge]()
  for node in dg.nodesWithSelfloopEdge():
    ret.incl((node, node))
  return ret
proc selfloopEdgesSeq*(dg: DiGraph): seq[Edge] =
  return dg.selfloopEdges()
iterator selfloopEdges*(dg: DiGraph): Edge =
  var ret = newSeq[Edge]()
  for node in dg.nodesWithSelfloopEdge():
    ret.add((node, node))
  for edge in ret:
    yield edge

proc isDirected*(dg: DiGraph): bool =
  return true

proc `[]`*(dg: DiGraph, node: Node): seq[Node] =
  return dg.neighbors(node)

proc `in`*(dg: DiGraph, node: Node): bool =
  return dg.contains(node)
proc `notin`*(dg: DiGraph, node: Node): bool =
  return not dg.contains(node)
proc `+`*(dg: DiGraph, node: Node): DiGraph =
  dg.addNode(node)
  return dg
proc `+`*(dg: DiGraph, nodes: openArray[Node]): DiGraph =
  dg.addNodesFrom(nodes)
  return dg
proc `+`*(dg: DiGraph, nodes: HashSet[Node]): DiGraph =
  dg.addNodesFrom(nodes)
  return dg
proc `-`*(dg: DiGraph, node: Node): DiGraph =
  dg.removeNode(node)
  return dg
proc `-`*(dg: DiGraph, nodes: openArray[Node]): DiGraph =
  dg.removeNodesFrom(nodes)
  return dg
proc `-`*(dg: DiGraph, nodes: HashSet[Node]): DiGraph =
  dg.removeNodesFrom(nodes)
  return dg
proc `+=`*(dg: DiGraph, node: Node) =
    dg.addNode(node)
proc `+=`*(dg: DiGraph, nodes: HashSet[Node]) =
    dg.addNodesFrom(nodes)
proc `+=`*(dg: DiGraph, nodes: openArray[Node]) =
    dg.addNodesFrom(nodes)
proc `-=`*(dg: DiGraph, node: Node) =
    dg.removeNode(node)
proc `-=`*(dg: DiGraph, nodes: HashSet[Node]) =
    dg.removeNodesFrom(nodes)
proc `-=`*(dg: DiGraph, nodes: openArray[Node]) =
    dg.removeNodesFrom(nodes)

proc `in`*(dg: DiGraph, edge: Edge): bool =
    return dg.hasEdge(edge)
proc `+`*(dg: DiGraph, edge: Edge): DiGraph =
    dg.addEdge(edge)
    return dg
proc `+`*(dg: DiGraph, edges: HashSet[Edge]): DiGraph =
    dg.addEdgesFrom(edges)
    return dg
proc `+`*(dg: DiGraph, edges: openArray[Edge]): DiGraph =
    dg.addEdgesFrom(edges)
    return dg
proc `-`*(dg: DiGraph, edge: Edge): DiGraph =
    dg.removeEdge(edge)
    return dg
proc `-`*(dg: DiGraph, edges: HashSet[Edge]): DiGraph =
    dg.removeEdgesFrom(edges)
    return dg
proc `-`*(dg: DiGraph, edges: openArray[Edge]): DiGraph =
    dg.removeEdgesFrom(edges)
    return dg
proc `+=`*(dg: DiGraph, edge: Edge) =
    dg.addEdge(edge)
proc `+=`*(dg: DiGraph, edges: HashSet[Edge]) =
    dg.addEdgesFrom(edges)
proc `+=`*(dg: DiGraph, edges: openArray[Edge]) =
    dg.addEdgesFrom(edges)
proc `-=`*(dg: DiGraph, edge: Edge) =
    dg.removeEdge(edge)
proc `-=`*(dg: DiGraph, edges: HashSet[Edge]) =
    dg.removeEdgesFrom(edges)
proc `-=`*(dg: DiGraph, edges: openArray[Edge]) =
    dg.removeEdgesFrom(edges)

# -------------------------------------------------------------------
# TODO: Weighted (Undirected) Graph
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# TODO: Weighted Directed Graph
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Converter etc
# -------------------------------------------------------------------

proc copyAsGraph*(g: Graph): Graph =
  let ret = newGraph()
  ret.addEdgesFrom(g.edges())
  return ret
proc copyAsDiGraph*(g: Graph): DiGraph =
  let ret = newDiGraph()
  for edge in g.edges():
    ret.addEdge(edge.u, edge.v)
    ret.addEdge(edge.v, edge.u)
  return ret
proc copyAsDiGraph*(dg: DiGraph): DiGraph =
  let ret = newDiGraph()
  ret.addEdgesFrom(dg.edges())
  return ret
proc copyAsGraph*(dg: DiGraph): Graph =
  let ret = newGraph()
  for edge in dg.edges():
    ret.addEdge(edge)
  return ret

proc toDirected*(g: Graph): DiGraph =
  let ret = newDiGraph()
  for edge in g.edges():
    ret.addEdge(edge.u, edge.v)
    ret.addEdge(edge.v, edge.u)
  return ret
proc toUndirected*(dg: DiGraph): Graph =
  let ret = newGraph()
  for edge in dg.edges():
    ret.addEdge(edge.u, edge.v)
  return ret

proc createEmptyCopyAsGraph*(g: Graph): Graph =
  let ret = newGraph(g.nodes())
  return ret
proc createEmptyCopyAsDiGraph*(g: Graph): DiGraph =
  let ret = newDiGraph(g.nodes())
  return ret
proc createEmptyCopyAsGraph*(dg: DiGraph): Graph =
  let ret = newGraph(dg.nodes())
  return ret
proc createEmptyCopyAsDiGraph*(dg: DiGraph): DiGraph =
  let ret = newDiGraph(dg.nodes())
  return ret

proc isEmpty*(g: Graph): bool =
  return len(g.edges()) == 0
proc isEmpty*(dg: DiGraph): bool =
  return len(dg.edges()) == 0

proc reversed*(edge: Edge): Edge =
  return (edge.v, edge.u)
proc reversed*(dg: DiGraph): DiGraph =
  let ret = newDiGraph()
  for edge in dg.edges():
    ret.addEdge(reversed(edge))
  return ret
