# SCOPE: Sibling-Contrast Procedural Evolution

SCOPE is an LLM-based transmission-grid reinforcement planner. The main agent proposes alternative reinforcement plans, verifies them with AC power flow and peak-hour N-1 contingency analysis, and reuses verified experience through episodic and procedural memory.

<!-- ## Repository Structure

```text
.
├── README.md
├── agent.py                         # main SCOPE planner and controller
├── prompts.py                       # candidate, fusion, and memory prompts
├── requirements.txt                 # simulator dependencies
├── serve_qwen.sh                    # starts the local Qwen/vLLM server
├── vllm_backend_full.py             # LLM inference and tool-call transport
│
├── grid_tools/                      # grid simulator and planner-facing tools
│   ├── __init__.py
│   ├── config.py                    # paths, security limits, costs, memory location
│   ├── network.py                   # loads the IEEE 118-bus grid and operating scenarios
│   ├── actions.py                   # applies and prices reinforcement actions
│   ├── measurement.py               # AC power flow and N-1 checks
│   ├── state.py                     # current retained grid and planning state
│   ├── episodes.py                  # episode state, cost, and finalization
│   ├── api.py                       # tool schemas and dispatch
│   └── tools/
│       ├── scan_day.py              # scans the 24-hour operating scenario
│       ├── analyse.py               # diagnoses current security violations
│       └── test_plan.py             # verifies A, B, and C sequentially, tests D, and handles commits
│
├── episode_store.py                 # stores verified episode trajectories
├── evolution_memory.py              # procedural-memory runtime and strategy recall
├── consolidate_trajectories.py      # turns completed trajectories into strategies
├── trajectory_llm.py                # LLM interface used during consolidation
├── memory/                          # episodic and procedural memory banks by evaluation fold
├── tests/
│   └── smoke_test.py                # checks the simulator and memory setup
│
└── data/
    ├── topology_ieee118.json        # augmented IEEE 118-bus network
    └── ieee118_scenarios/
        ├── mapping_report.json      # scenario-to-network mapping report
        └── records/                 # 24-hour operating scenario JSON files
```-->

## Run

From the extracted `scope` directory, install the simulator dependencies and start the local model server in an environment with vLLM installed.

```bash
pip install -r requirements.txt
bash serve_qwen.sh
```

The default server uses the 4-bit Qwen3.5-9B model. Once the server is ready, open a second terminal in the `scope` directory and run one scenario.

```bash
python agent.py --scenario SCENARIO_NAME_HERE
```

## Verify the installation

Run the smoke test from the `scope` directory. It checks the grid simulator and memory setup without starting the model server.

```bash
python tests/smoke_test.py
```
