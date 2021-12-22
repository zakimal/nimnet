import parsecsv
import strformat
import strutils
import sets

import ../nimnet.nim

###
# adjlist format
#
# e.g.) delimiter = ' '
# ```tsv
# 0 1 2
# 1 2
# ```
#
###

proc readAdjlistAsGraph*(path: string, delimiter: char = ' '): Graph =
  if path == "":
    raise newException(Exception, fmt"file not found at path='{path}'")
  var parser: CsvParser
  parser.open(path, separator=delimiter)
  defer: parser.close()
  var G = newGraph()
  var edges: seq[Edge] = @[]
  while parser.readRow():
    var adjlistStr: seq[string] = parser.row
    let adjlistLen = len(adjlistStr)
    let node = parseInt(adjlistStr[0])
    for i in 1..<adjlistLen:
      let adj = parseInt(adjlistStr[i])
      edges.add((node, adj))
  G.addEdgesFrom(edges)
  return G
proc writeAdjlist*(path: string, G: Graph, delimiter: char = ' ') =
  if path == "":
    raise newException(Exception, fmt"file not found at path='{path}'")
  let fp = open(path, fmWrite)
  defer: fp.close()
  for node in G.nodes():
    var line: string = $node
    for nbr in G.neighbors(node):
      line = line & delimiter & $nbr
    fp.writeLine(line)
iterator generatedAdjlist*(G: Graph, delimiter: char = ' '): string =
  for node in G.nodes():
    var line: string = $node
    for nbr in G.neighbors(node):
      line = line & delimiter & $nbr
    yield line
proc parseAdjlistAsGraph*(lines: openArray[string], delimiter: char = ' '): Graph =
  let G = newGraph()
  var edges: seq[Edge] = @[]
  for line in lines:
    let adjlistStr = line.split(delimiter)
    let adjlistLen = len(adjlistStr)
    let node = parseInt(adjlistStr[0])
    for i in 1..<adjlistLen:
      let adj = parseInt(adjlistStr[i])
      edges.add((node, adj))
  G.addEdgesFrom(edges)
  return G
proc parseAdjlistAsGraph*(lines: HashSet[string], delimiter: char = ' '): Graph =
  let G = newGraph()
  var edges: seq[Edge] = @[]
  for line in lines:
    let adjlistStr = line.split(delimiter)
    let adjlistLen = len(adjlistStr)
    let node = parseInt(adjlistStr[0])
    for i in 1..<adjlistLen:
      let adj = parseInt(adjlistStr[i])
      edges.add((node, adj))
  G.addEdgesFrom(edges)
  return G

proc readAdjlistAsDiGraph*(path: string, delimiter: char = ' '): DiGraph =
  if path == "":
    raise newException(Exception, fmt"file not found at path='{path}'")
  var parser: CsvParser
  parser.open(path, separator=delimiter)
  defer: parser.close()
  var DG = newDiGraph()
  var edges: seq[Edge] = @[]
  while parser.readRow():
    var adjlistStr: seq[string] = parser.row
    let adjlistLen = len(adjlistStr)
    let node = parseInt(adjlistStr[0])
    for i in 1..<adjlistLen:
      let adj = parseInt(adjlistStr[i])
      edges.add((node, adj))
  DG.addEdgesFrom(edges)
  return DG
proc writeAdjlist*(path: string, DG: DiGraph, delimiter: char = ' ') =
  if path == "":
    raise newException(Exception, fmt"file not found at path='{path}'")
  let fp = open(path, fmWrite)
  defer: fp.close()
  for node in DG.nodes():
    var line: string = $node
    for succ in DG.successors(node):
      line = line & delimiter & $succ
    fp.writeLine(line)
iterator generatedAdjlist*(DG: DiGraph, delimiter: char = ' '): string =
  for node in DG.nodes():
    var line: string = $node
    for nbr in DG.neighbors(node):
      line = line & delimiter & $nbr
    yield line
proc parseAdjlistAsDiGraph*(lines: openArray[string], delimiter: char = ' '): DiGraph =
  let DG = newDiGraph()
  var edges: seq[Edge] = @[]
  for line in lines:
    let adjlistStr = line.split(delimiter)
    let adjlistLen = len(adjlistStr)
    let node = parseInt(adjlistStr[0])
    for i in 1..<adjlistLen:
      let adj = parseInt(adjlistStr[i])
      edges.add((node, adj))
  DG.addEdgesFrom(edges)
  return DG
proc parseAdjlistAsDiGraph*(lines: HashSet[string], delimiter: char = ' '): DiGraph =
  let DG = newDiGraph()
  var edges: seq[Edge] = @[]
  for line in lines:
    let adjlistStr = line.split(delimiter)
    let adjlistLen = len(adjlistStr)
    let node = parseInt(adjlistStr[0])
    for i in 1..<adjlistLen:
      let adj = parseInt(adjlistStr[i])
      edges.add((node, adj))
  DG.addEdgesFrom(edges)
  return DG

###
# edgelist format
#
# e.g.) delimiter = ' '
# ```
# 0 1
# 1 2
# ```
#
###

proc readEdgelistAsGraph*(path: string, delimiter: char = ' '): Graph =
  ## Read edgelist file from `path` as undirected graph
  if path == "":
    raise newException(Exception, fmt"file not found at path='{path}'")
  var parser: CsvParser
  parser.open(path, separator=delimiter)
  defer: parser.close()
  var G = newGraph()
  var edges: seq[Edge] = @[]
  while parser.readRow():
    var edgeStr: seq[string] = parser.row
    var edge: tuple[u, v: int]
    edge.u = parseInt(edgeStr[0])
    edge.v = parseInt(edgeStr[1])
    edges.add(edge)
  G.addEdgesFrom(edges)
  return G
proc writeEdgelist*(path: string, G: Graph, delimiter: char = ' ') =
  if path == "":
    raise newException(Exception, fmt"file not found at path='{path}'")
  let fp = open(path, fmWrite)
  defer: fp.close()
  for edge in G.edges():
    fp.writeLine(fmt"{edge.u}{delimiter}{edge.v}")
iterator generatedEdgelist*(G: Graph, delimiter: char = ' '): string =
  for edge in G.edges():
    yield fmt"{edge.u}{delimiter}{edge.v}"
proc parseEdgelistAsGraph*(lines: openArray[string], delimiter: char = ' '): Graph =
  let G = newGraph()
  for line in lines:
    let splitted = line.split(delimiter)
    let u = parseInt(splitted[0])
    let v = parseInt(splitted[2])
    G.addEdge(u, v)
  return G
proc parseEdgelistAsGraph*(lines: HashSet[string], delimiter: char = ' '): Graph =
  let G = newGraph()
  for line in lines:
    let splitted = line.split(delimiter)
    let u = parseInt(splitted[0])
    let v = parseInt(splitted[2])
    G.addEdge(u, v)
  return G

proc readEdgelistAsDiGraph*(path: string, delimiter: char = ' '): DiGraph =
  ## Read edgelist file from `path` as directed graph
  if path == "":
    raise newException(Exception, fmt"file not found at path='{path}'")
  var parser: CsvParser
  parser.open(path, separator=delimiter)
  defer: parser.close()
  var DG = newDiGraph()
  var edges: seq[Edge] = @[]
  while parser.readRow():
    var edgeStr: seq[string] = parser.row
    var edge: tuple[u, v: int]
    edge.u = parseInt(edgeStr[0])
    edge.v = parseInt(edgeStr[1])
    edges.add(edge)
  DG.addEdgesFrom(edges)
  return DG
proc writeEdgelist*(path: string, DG: DiGraph, delimiter: char = ' ') =
  if path == "":
    raise newException(Exception, fmt"file not found at path='{path}'")
  let fp = open(path, fmWrite)
  defer: fp.close()
  for edge in DG.edges():
    fp.writeLine(fmt"{edge.u}{delimiter}{edge.v}")
iterator generatedEdgelist*(DG: DiGraph, delimiter: char = ' '): string =
  for edge in DG.edges():
    yield fmt"{edge.u}{delimiter}{edge.v}"
proc parseEdgelistAsDiGraph*(lines: openArray[string], delimiter: char = ' '): DiGraph =
  let DG = newDiGraph()
  for line in lines:
    let splitted = line.split(delimiter)
    let u = parseInt(splitted[0])
    let v = parseInt(splitted[2])
    DG.addEdge(u, v)
  return DG
proc parseEdgelistAsDiGraph*(lines: HashSet[string], delimiter: char = ' '): DiGraph =
  let DG = newDiGraph()
  for line in lines:
    let splitted = line.split(delimiter)
    let u = parseInt(splitted[0])
    let v = parseInt(splitted[2])
    DG.addEdge(u, v)
  return DG

# TODO
# - GML
# - GraphML
# - JSON
# - LEDA
# - SparseGraph6
# - Pejak
# - GIS Shapefile