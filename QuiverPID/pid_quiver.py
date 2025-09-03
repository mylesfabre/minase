import json
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set
import math
import numpy as np

@dataclass
# Define PID Component with:
# id
# type
# attributes
class PidNode:
    id: str
    type: str
    attrs: Dict[str, Any] = field(default_factory=dict)

# Define PID Component with id
# type
# source
# destination
# attributes
@dataclass
class PidEdge:
    id: str
    type: str
    src: str
    dst: str
    attrs: Dict[str, Any] = field(default_factory=dict)

@dataclass
# Compose the above PID components 
# and be able to convert json representation of P&ID to self type
class PidGraph:
    nodes: Dict[str, PidNode]
    edges: Dict[str, PidEdge]

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "PidGraph":
        nodes = {n["id"]: PidNode(n["id"], n["type"], n.get("attrs", {})) for n in data.get("nodes", [])}
        edges = {e["id"]: PidEdge(e["id"], e["type"], e["src"], e["dst"], e.get("attrs", {})) for e in data.get("edges", [])}
        return PidGraph(nodes, edges)

# Define Quiver Component with
# id
# node_type
# dimension
# attributes
class QuiverVertex:
    def __init__(self, id: str, node_type: str, dim: int, attrs=None):
        self.id = id
        self.node_type = node_type
        self.dim = dim
        self.attrs = dict(attrs or {})

class QuiverArrow:
    def __init__(self, id: str, arrow_type: str, src: str, dst: str, attrs=None, A=None):
        self.id = id
        self.arrow_type = arrow_type
        self.src = src
        self.dst = dst
        self.attrs = dict(attrs or {})
        self.A = A

class QuiverRep:
    def __init__(self, vertices: Dict[str, QuiverVertex], arrows: Dict[str, QuiverArrow]):
        self.vertices = vertices
        self.arrows = arrows

    def dimension_vector(self) -> Dict[str, int]:
        return {vid: v.dim for vid, v in self.vertices.items()}

    def relation_types(self) -> set:
        return set(a.arrow_type for a in self.arrows.values())

class PidToQuiver:
    DEFAULT_DIMS = {
        "tank": 1, "vessel": 1, "reactor": 2, "exchanger": 2,
        "pump": 0, "compressor": 0, "valve": 0, "junction": 0,
        "manifold": 0, "sensor": 0, "level_tx": 0, "pressure_tx": 0,
        "flow_tx": 0, "controller": 0, "source": 0, "sink": 0, "node": 0
    }
    TRANSPORT_TYPES = {"pipe", "valve", "pump", "compressor", "line"}
    SIGNAL_TYPES = {"meas", "ctrl", "signal"}

    def __init__(self, linearize: bool = True):
        self.linearize = linearize

    def __call__(self, g: PidGraph) -> QuiverRep:
        vertices = {}
        arrows = {}

        for nid, n in g.nodes.items():
            dim = self.DEFAULT_DIMS.get(n.type, 0)
            if dim > 0 or n.attrs.get("stateful", False):
                dim = n.attrs.get("dim", dim if dim > 0 else 1)
                vertices[nid] = QuiverVertex(nid, n.type, dim, n.attrs)

        for eid, e in g.edges.items():
            if e.type in self.TRANSPORT_TYPES or e.type in self.SIGNAL_TYPES:
                arrows[eid] = QuiverArrow(eid, e.type, e.src, e.dst, e.attrs)

        if self.linearize:
            for a in arrows.values():
                src_dim = vertices.get(a.src).dim if a.src in vertices else 1
                dst_dim = vertices.get(a.dst).dim if a.dst in vertices else 1
                shape = (max(1, dst_dim), max(1, src_dim))
                if a.arrow_type in {"pipe", "line"}:
                    gain = self._pipe_linear_gain(a.attrs)
                elif a.arrow_type == "valve":
                    gain = self._valve_linear_gain(a.attrs)
                elif a.arrow_type in {"pump", "compressor"}:
                    gain = self._pump_linear_gain(a.attrs)
                else:
                    gain = 1.0
                a.A = np.full(shape, float(gain))
        return QuiverRep(vertices, arrows)

    @staticmethod
    def _pipe_linear_gain(attrs: Dict[str, Any]) -> float:
        D = attrs.get("D", 0.1)
        L = attrs.get("L", 10.0)
        mu = attrs.get("mu", 1e-3)
        k = (math.pi * (D ** 4)) / (128.0 * mu * max(L, 1e-6))
        return float(max(1e-6, min(k, 1e3)))

    @staticmethod
    def _valve_linear_gain(attrs: Dict[str, Any]) -> float:
        Cv = attrs.get("Cv", 10.0)
        openness = attrs.get("openness", 0.5)
        return float(max(1e-4, Cv * max(0.0, min(1.0, openness))))

    @staticmethod
    def _pump_linear_gain(attrs: Dict[str, Any]) -> float:
        head = attrs.get("rated_head", 20.0)
        speed = attrs.get("speed", 1.0)
        return float(max(1e-3, head * speed))

def gnn_tensors(quiv: QuiverRep) -> Dict[str, Any]:
    node_ids = list(quiv.vertices.keys())
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    type_vocab = {t: i for i, t in enumerate(sorted({v.node_type for v in quiv.vertices.values()}))}
    X = np.zeros((len(node_ids), 2), dtype=float)
    for nid, v in quiv.vertices.items():
        i = node_index[nid]
        X[i, 0] = v.dim
        X[i, 1] = type_vocab[v.node_type]

    rel_groups = {}
    for a in quiv.arrows.values():
        rel_groups.setdefault(a.arrow_type, []).append(a)

    edge_index = {}
    edge_attr = {}
    A_linear = {}
    for r, arrs in rel_groups.items():
        rows, cols, attrs, mats = [], [], [], []
        for a in arrs:
            cols.append(node_index.get(a.src, 0))
            rows.append(node_index.get(a.dst, 0))
            attrs.append(a.attrs)
            mats.append(a.A if a.A is not None else np.array([[1.0]]))
        edge_index[r] = np.vstack([np.array(rows, dtype=int), np.array(cols, dtype=int)])
        edge_attr[r] = attrs
        A_linear[r] = mats

    return {
        "node_ids": node_ids,
        "node_index": node_index,
        "node_features": X,
        "type_vocab": type_vocab,
        "edge_index_by_relation": edge_index,
        "edge_attr_by_relation": edge_attr,
        "edge_linear_by_relation": A_linear,
        "dim_vector": {vid: v.dim for vid, v in quiv.vertices.items()}
    }

# Compute SCF via maxflow/reachability
def build_capacity_graph(g: PidGraph, capacity_field: str = "capacity") -> Dict[str, Dict[str, float]]:
    cap_graph = {}
    functor = PidToQuiver(linearize=True)
    quiv = functor(g)

    def proxy_capacity(e: PidEdge) -> float:
        if e.type == "pipe":
            return PidToQuiver._pipe_linear_gain(e.attrs)
        if e.type == "valve":
            return PidToQuiver._valve_linear_gain(e.attrs)
        if e.type in {"pump", "compressor"}:
            return 1e3 * PidToQuiver._pump_linear_gain(e.attrs)
        return 1.0

    for e in g.edges.values():
        if e.type in PidToQuiver.TRANSPORT_TYPES:
            cap = float(e.attrs.get(capacity_field, proxy_capacity(e)))
            cap_graph.setdefault(e.src, {})[e.dst] = cap
    return cap_graph

def dinic_maxflow(cap_graph: Dict[str, Dict[str, float]], s: str, t: str) -> float:
    """
    Compute maximum flow from s to t in a directed graph with capacities
    using Dinic's algorithm.

    cap_graph: adjacency dict {u: {v: capacity}}
    s, t: source and sink node IDs
    """
    # Build residual capacity graph
    res = {u: dict(v) for u, v in cap_graph.items()}
    for u in list(cap_graph.keys()):
        for v in cap_graph[u].keys():
            res.setdefault(v, {})
            res[v].setdefault(u, 0.0)

    def bfs_level() -> Dict[str, int]:
        """Build level graph (distance from source)."""
        level = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for v, cap in res[u].items():
                if cap > 1e-12 and v not in level:
                    level[v] = level[u] + 1
                    q.append(v)
        return level

    def dfs_flow(u: str, pushed: float, level: Dict[str, int], next_iter: Dict[str, int]) -> float:
        """DFS blocking flow with edge iterator."""
        if u == t or pushed == 0:
            return pushed
        neighbors = list(res[u].items())
        while next_iter[u] < len(neighbors):
            v, cap = neighbors[next_iter[u]]
            next_iter[u] += 1
            if cap > 1e-12 and level.get(v, -1) == level[u] + 1:
                tr = dfs_flow(v, min(pushed, cap), level, next_iter)
                if tr > 0:
                    res[u][v] -= tr
                    res[v][u] = res.get(v, {}).get(u, 0.0) + tr
                    return tr
        return 0.0

    flow = 0.0
    while True:
        level = bfs_level()
        if t not in level:  # no augmenting path
            break
        next_iter = {u: 0 for u in res}
        while True:
            pushed = dfs_flow(s, float("inf"), level, next_iter)
            if pushed == 0:
                break
            flow += pushed

    return flow

from collections import deque
from typing import Dict
from pid_quiver import PidGraph  # assuming your class is in pid_quiver.py

# Use PID Objects Directly
# Don't be mad about code duplication... let it flow..
def dinic_maxflow_pid(g: PidGraph, s: str, t: str) -> float:
    """
    Run Dinic's MaxFlow algorithm on a PidGraph.
    g: PidGraph instance
    s: source node id
    t: sink node id
    """
    # Convert PidGraph edges into residual capacity dict
    res: Dict[str, Dict[str, float]] = {}
    for e in g.edges.values():
        res.setdefault(e.src, {})[e.dst] = float(e.capacity)
        res.setdefault(e.dst, {})  # ensure reverse exists
        res[e.dst].setdefault(e.src, 0.0)

    def bfs_level() -> Dict[str, int]:
        """Build level graph (distance from source)."""
        level = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for v, cap in res.get(u, {}).items():
                if cap > 1e-12 and v not in level:
                    level[v] = level[u] + 1
                    q.append(v)
        return level

    def dfs_flow(u: str, pushed: float, level: Dict[str, int], next_iter: Dict[str, int]) -> float:
        """DFS blocking flow with edge iterator."""
        if u == t or pushed == 0:
            return pushed
        neighbors = list(res[u].items())
        while next_iter[u] < len(neighbors):
            v, cap = neighbors[next_iter[u]]
            next_iter[u] += 1
            if cap > 1e-12 and level.get(v, -1) == level[u] + 1:
                tr = dfs_flow(v, min(pushed, cap), level, next_iter)
                if tr > 0:
                    res[u][v] -= tr
                    res[v][u] = res.get(v, {}).get(u, 0.0) + tr
                    return tr
        return 0.0

    flow = 0.0
    while True:
        level = bfs_level()
        if t not in level:  # no augmenting path
            break
        next_iter = {u: 0 for u in res}
        while True:
            pushed = dfs_flow(s, float("inf"), level, next_iter)
            if pushed == 0:
                break
            flow += pushed

    return flow

# Get source/sink roles from graph
def detect_sources_sinks(g: PidGraph):
    sources = [nid for nid, n in g.nodes.items() if n.attrs.get("role") == "source"]
    sinks = [nid for nid, n in g.nodes.items() if n.attrs.get("role") == "sink"]
    return sources, sinks

def maxflow_throughput(g: PidGraph) -> Optional[float]:
    sources, sinks = detect_sources_sinks(g)
    if not sources or not sinks:
        return None
    cap = build_capacity_graph(g)
    total = 0.0
    for s in sources:
        for t in sinks:
            if s in cap:
                total += dinic_maxflow(cap, s, t)
    return total

def reachability_score(g: PidGraph) -> float:
    nodes = list(g.nodes.keys())
    adj = {u: set() for u in nodes}
    for e in g.edges.values():
        if e.src in adj and e.dst in adj:   # safer if data might be inconsistent
            adj[e.src].add(e.dst)

    reach = 0
    for u in nodes:
        seen = {u}
        q = deque([u])
        while q:
            x = q.popleft()
            for v in adj[x]:
                if v not in seen:
                    seen.add(v); q.append(v)
        reach += len(seen) - 1

    denom = max(1, len(nodes) * (len(nodes) - 1))
    return reach / denom

def scf_scores(g: PidGraph, J: Optional[str] = "maxflow"):
    baseline = None
    if J == "maxflow":
        baseline = maxflow_throughput(g)
        if baseline is None:
            J = "reachability"
    if baseline is None and J == "reachability":
        baseline = reachability_score(g)

    results = []

    def eval_J(graph: PidGraph) -> float:
        if J == "maxflow":
            val = maxflow_throughput(graph)
            if val is None:
                return reachability_score(graph)
            return val
        else:
            return reachability_score(graph)

    for nid in list(g.nodes.keys()):
        g2 = PidGraph(
            nodes={k: v for k, v in g.nodes.items() if k != nid},
            edges={eid: e for eid, e in g.edges.items() if e.src != nid and e.dst != nid}
        )
        val = eval_J(g2)
        drop = (baseline - val) / (abs(baseline) + 1e-9)
        results.append((f"node:{nid}", float(drop)))

    for eid in list(g.edges.keys()):
        g2 = PidGraph(
            nodes=g.nodes.copy(),
            edges={k: v for k, v in g.edges.items() if k != eid}
        )
        val = eval_J(g2)
        drop = (baseline - val) / (abs(baseline) + 1e-9)
        results.append((f"edge:{eid}", float(drop)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def resilience_analysis(g: PidGraph, source: str, sink: str) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute resilience metrics for a P&ID system by simulating
    the failure of each edge one at a time.

    Returns a dictionary:
      edge_id -> (baseline_flow, flow_after_failure, resilience_ratio)

    resilience_ratio = flow_after_failure / baseline_flow
    """
    results = {}

    # Compute baseline max flow
    baseline_flow = dinic_maxflow_pid(g, source, sink)

    for edge_id in list(g.edges.keys()):
        # Copy graph
        g_copy = PidGraph(
            nodes=g.nodes.copy(),
            edges=g.edges.copy()
        )

        # Simulate failure: remove edge
        g_copy.edges.pop(edge_id, None)

        # Compute new max flow
        flow_after = dinic_maxflow_pid(g_copy, source, sink)

        # Avoid division by zero
        resilience_ratio = flow_after / baseline_flow if baseline_flow > 1e-12 else 0.0

        results[edge_id] = (baseline_flow, flow_after, resilience_ratio)

    return results
