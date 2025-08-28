from pid_quiver import PidGraph, PidToQuiver, gnn_tensors, scf_scores
import json

print("PID GRAPH")
g = PidGraph.from_json(json.load(open('example_flow.json')))
print(g)
print("PID QUIVER")
q = PidToQuiver()(g)
print(q)
print("TENSORS")
tensors = gnn_tensors(q)
print(tensors)
print("SINGLE COMPONENT FAILURE SCORES")
scf = scf_scores(g, J='maxflow')
print(scf)