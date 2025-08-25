TESTING:
- pytest -q test_pid_quiver.py

FILES:
- pid_quiver.py : Reusable module with P&ID to Quiver mapping, GNN tensors, and SingleComponentFailure utilities.
- example_flow.json : Source to Sink demo with a pump and valve.
- example_loop.json : Closed loop with a level transmitter.

TO USE,
1. Use/update example_flow.json and example_loop.json
2. run the following command in terminal:
    python3 demo.py

TIPS:
- change dinic_maxflow() to dinic_maxflow_pid() to use PID Objects directly

INFO:
	A P&ID (piping & instrumentation diagram) is usually interpreted visually. By encoding it as a graph/quiver, we translate pipes, pumps, valves, sensors into nodes and edges with capacity, direction, and state.
	•	This makes the system computable — you can run graph algorithms (flow, reliability, controllability) directly instead of manually inspecting schematics.


🔧  MaxFlow + Dinic’s Algorithm enables Capacity Analysis
	•	The max flow represents the theoretical throughput of a process unit or network (e.g., how much material can reach a reactor).
	•	Using Dinic’s (or Edmonds–Karp earlier) means the computation scales better for large P&ID systems with hundreds of components.
	•	This provides a quantitative benchmark: the maximum safe operating rate before bottlenecks appear.

⸻

⚠️ 3. Resilience Analysis → Single Component Failure
	•	By systematically removing each edge and recomputing flow, the resilience function reveals:
	•	Which component failures degrade performance the most.
	•	Redundancy in parallel paths (robust design).
	•	Vulnerable bottlenecks (single points of failure).
	•	This directly supports HAZOP (Hazard and Operability) studies and FMEA (Failure Mode and Effects Analysis) in industrial engineering.

⸻

📊 4. Link to PCA (Principal Component Analysis)

Here’s where it gets interesting: PCA in industrial engineering is often used for multivariate process monitoring (many correlated sensor signals).
	•	The graph embedding of the P&ID can be treated as a structural prior for PCA:
	•	Instead of treating every sensor independently, you know which nodes interact via pipes/flows.
	•	You can cluster correlated signals by graph connectivity, improving interpretability of PCA loadings.
	•	Resilience scores can serve as features in PCA space:
	•	e.g., include resilience ratio under edge failures as meta-features when analyzing system stability.
	•	End result: PCA shifts from “just statistical dimensionality reduction” → “process-aware anomaly detection” guided by physical connectivity.

⸻

🚀 5. Holistic Value to Industrial Engineering
	1.	Model-based monitoring: Algorithms grounded in the P&ID itself, not just black-box data.
	2.	Predictive reliability: Quantify how process flow degrades under failures.
	3.	Safer operations: Identify single points of failure before commissioning or during revamp.
	4.	Enhanced PCA: Integrate structure + data, making fault detection more explainable.
	5.	Digital twin: You’re basically building a graph-based digital twin that can connect to real-time plant data.

⸻

👉🏾 So, holistically, this code improves industrial engineering PCA by embedding physical plant topology into data analysis, turning PCA into a process-aware diagnostic tool instead of a purely statistical one.