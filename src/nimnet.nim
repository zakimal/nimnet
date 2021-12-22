import algorithm
import sets
import sequtils
import tables
import strformat

###
#  Basic types
###

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

###
#  Exceptions
###

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

###
#  Basic Operations
###

# (Undirected) Graph
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
  g.hasEdge(edge.u, edge.v)
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
  return m.float / (n*(n - 1)).float

proc nodeSubgraph*(g: Graph, nodes: HashSet[Node]): Graph =
  var ret = newGraph()
  for (u, v) in g.edges():
    if u in nodes and v in nodes:
      ret.addEdge((u, v))
  return ret
proc nodeSubgraph*(g: Graph, nodes: seq[Node]): Graph =
  var ret = newGraph()
  let nodesSet = nodes.toHashSet()
  for (u, v) in g.edges():
    if u in nodesSet and v in nodesSet:
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
  let edgesSet = edges.toHashSet()
  for edge in edgesSet:
    if g.hasEdge(edge):
      ret.addEdge(edge)
  return ret
proc subgraph*(g: Graph, nodes: HashSet[Node]): Graph =
  return g.nodeSubgraph(nodes)
proc subgraph*(g: Graph, nodes: seq[Node]): Graph =
  return g.nodeSubgraph(nodes)

proc info*(g: Graph, node: Node): string =
  if node notin g.adj:
    raise newNNNodeNotFound(node)
  var ret = fmt"node {node} has following properties:"
  ret = ret & "\n"
  ret = ret & fmt"Degree: {g.degree(node)}"
  ret = ret & "\n"
  ret = ret & fmt"Neighbors: {g.neighbors(node)}"
  ret = ret & "\n"
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
  for nbr in g.neighbors(u):
    if (nbr in g.neighborsSet(v)) and (nbr != u) and (nbr != v):
      commonNeighbors.incl(nbr)
  var ret = commonNeighbors.toSeq()
  ret.sort()
  return ret
proc commonNeighborsIterator*(g: Graph, u, v: Node): iterator: Node =
  var commonNeighbors = initHashSet[Node]()
  for nbr in g.neighbors(u):
    if (nbr in g.neighborsSet(v)) and (nbr != u) and (nbr != v):
      commonNeighbors.incl(nbr)
  var ret = commonNeighbors.toSeq()
  ret.sort()
  return iterator: Node =
    for commonNeighbor in ret:
      yield commonNeighbor
proc commonNeighborsSet*(g: Graph, u, v: Node): HashSet[Node] =
  var commonNeighbors = initHashSet[Node]()
  for nbr in g.neighbors(u):
    if (nbr in g.neighborsSet(v)) and (nbr != u) and (nbr != v):
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



# Directed Graph

# proc copy*(g: Graph): Graph =
#   var ret = newGraph()
#   ret.addEdgesFrom(g.edges())
#   return ret
# proc copyAsDiGraph*(g: Graph): DiGraph
# proc copy*(dg: DiGraph): DiGraph
# proc copyAsGraph*(dg: DiGraph): Graph
# proc toDirected*(g: Graph): DiGraph
# proc toUndirected*(dg: DiGraph): Graph
