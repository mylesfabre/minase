import pytest
import tempfile
import json
from pid_quiver import load_pid_from_json, pid_to_quiver, simulate_single_component_failure, PidGraph, Node, Edge, dinic_maxflow_pid, resilience_analysis

@pytest.fixture
def sample_pid_data():
    return {
        "nodes": [
            {"id": "T1", "type": "Tank"},
            {"id": "P1", "type": "Pump"},
            {"id": "V1", "type": "Valve"}
        ],
        "edges": [
            {"src": "T1", "dst": "P1", "type": "flow"},
            {"src": "P1", "dst": "V1", "type": "flow"}
        ]
    }

def test_load_pid_from_json(sample_pid_data):
    # Write temporary JSON file
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp:
        json.dump(sample_pid_data, tmp)
        tmp.flush()
        path = tmp.name

    loaded = load_pid_from_json(path)
    assert "nodes" in loaded
    assert "edges" in loaded
    assert loaded["nodes"][0]["id"] == "T1"

def test_pid_to_quiver(sample_pid_data):
    quiver = pid_to_quiver(sample_pid_data)

    assert "vertices" in quiver
    assert "arrows" in quiver
    assert len(quiver["vertices"]) == 3
    assert len(quiver["arrows"]) == 2
    assert quiver["arrows"][0]["src"] == "T1"
    assert quiver["arrows"][0]["dst"] == "P1"

def test_single_component_failure(sample_pid_data):
    quiver = pid_to_quiver(sample_pid_data)
    failed = simulate_single_component_failure(quiver, "P1")

    # Check that the pump and its connections are removed
    vertex_ids = [v["id"] for v in failed["vertices"]]
    assert "P1" not in vertex_ids

    arrow_pairs = [(a["src"], a["dst"]) for a in failed["arrows"]]
    assert ("T1", "P1") not in arrow_pairs
    assert ("P1", "V1") not in arrow_pairs

def test_failure_nonexistent_component(sample_pid_data):
    quiver = pid_to_quiver(sample_pid_data)
    failed = simulate_single_component_failure(quiver, "X999")

    # Should not modify the graph if component doesn't exist
    assert failed == quiver

def test_dinic_maxflow_simple_chain():
    """
    Simple linear chain A -> B -> C -> D
    Bottleneck is edge (B->C) with capacity 5.
    """
    g = PidGraph(
        nodes={
            "A": Node("A", "source"),
            "B": Node("B", "pump"),
            "C": Node("C", "tank"),
            "D": Node("D", "sink"),
        },
        edges={
            "AB": Edge("AB", "A", "B", capacity=10.0),
            "BC": Edge("BC", "B", "C", capacity=5.0),
            "CD": Edge("CD", "C", "D", capacity=8.0),
        },
    )

    maxflow = dinic_maxflow_pid(g, "A", "D")
    assert maxflow == 5.0


def test_dinic_maxflow_parallel_paths():
    """
    Graph: A -> B -> D and A -> C -> D
    Each path has capacity 5, so total flow = 10.
    """
    g = PidGraph(
        nodes={
            "A": Node("A", "source"),
            "B": Node("B", "pump"),
            "C": Node("C", "pump"),
            "D": Node("D", "sink"),
        },
        edges={
            "AB": Edge("AB", "A", "B", capacity=5.0),
            "BD": Edge("BD", "B", "D", capacity=5.0),
            "AC": Edge("AC", "A", "C", capacity=5.0),
            "CD": Edge("CD", "C", "D", capacity=5.0),
        },
    )

    maxflow = dinic_maxflow_pid(g, "A", "D")
    assert maxflow == 10.0


def test_dinic_maxflow_no_path():
    """
    Graph: A and D are disconnected → maxflow = 0
    """
    g = PidGraph(
        nodes={
            "A": Node("A", "source"),
            "B": Node("B", "pump"),
            "D": Node("D", "sink"),
        },
        edges={
            "AB": Edge("AB", "A", "B", capacity=3.0),
        },
    )

    maxflow = dinic_maxflow_pid(g, "A", "D")
    assert maxflow == 0.0

def test_dinic_maxflow_single_component_failure():
    """
    Parallel paths: A -> B -> D and A -> C -> D.
    Each path has capacity 5, so baseline flow = 10.
    If edge AC fails, flow should drop to 5.
    """
    g = PidGraph(
        nodes={
            "A": Node("A", "source"),
            "B": Node("B", "pump"),
            "C": Node("C", "pump"),
            "D": Node("D", "sink"),
        },
        edges={
            "AB": Edge("AB", "A", "B", capacity=5.0),
            "BD": Edge("BD", "B", "D", capacity=5.0),
            "AC": Edge("AC", "A", "C", capacity=5.0),
            "CD": Edge("CD", "C", "D", capacity=5.0),
        },
    )

    # Baseline maxflow
    maxflow_before = dinic_maxflow_pid(g, "A", "D")
    assert maxflow_before == 10.0

    # Simulate single-component failure (edge AC removed)
    g.edges.pop("AC")

    maxflow_after = dinic_maxflow_pid(g, "A", "D")
    assert maxflow_after == 5.0

    # Check that the failure reduced capacity by half
    assert maxflow_after < maxflow_before

def test_resillience():
    g = PidGraph(
    nodes={
        "A": Node("A", "source"),
        "B": Node("B", "pump"),
        "C": Node("C", "pump"),
        "D": Node("D", "sink"),
    },
    edges={
        "AB": Edge("AB", "A", "B", capacity=5.0),
        "BD": Edge("BD", "B", "D", capacity=5.0),
        "AC": Edge("AC", "A", "C", capacity=5.0),
        "CD": Edge("CD", "C", "D", capacity=5.0),
    },
)

    resilience = resilience_analysis(g, "A", "D")
    for edge, (base, after, ratio) in resilience.items():
        print(f"Failure of {edge}: flow {after}/{base} ({ratio:.2f})")