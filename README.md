# SCOPE: Sibling-Contrast Procedural Evolution

SCOPE is an LLM-based transmission-grid reinforcement planner. The main agent proposes alternative reinforcement plans, verifies them with AC power flow and peak-hour N-1 contingency analysis, and reuses verified experience through episodic and procedural memory.

## Repository Structure

```text
.
├── README.md
├── agent.py                         # main SCOPE planner and controller
├── serve_qwen.sh                    # starts the local Qwen/vLLM server
├── vllm_backend_full.py             # LLM inference and tool-call transport
│
├── grid_tools/                      # grid simulator and planner-facing tools
│   ├── __init__.py                  
│   ├── config.py                    # paths, security limits, costs, memory location
│   ├── network.py                   # loads IEEE-118 grid and operating scenarios
│   ├── actions.py                   # applies and prices reinforcement actions
│   ├── measurement.py               # AC power flow and N-1 checks
│   ├── state.py                     # current retained grid/planning state
│   ├── episodes.py                  # episode state, cost, finalization
│   ├── api.py                       # tool schemas and dispatch
│   └── tools/
│       ├── scan_day.py              # scans the 24-hour operating scenario
│       ├── analyse.py               # diagnoses current security violations
│       └── test_plan.py             # verifies A/B/C candidates, plan fusion, and commits
│
├── episode_store.py                 # store verified episode trajectories
├── evolution_memory.py              # procedural-memory runtime and strategy recall
├── consolidate_trajectories.py      # turns completed trajectories into strategies
├── trajectory_llm.py                # LLM interface used during consolidation
│
└── data/
    ├── topology_ieee118.json         # augmented IEEE 118-bus network
    └── ieee118_scenarios/
        ├── mapping_report.json       # validates scenario-to-network mapping
        └── records/                  # 24-hour operating scenario JSON files
