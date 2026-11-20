import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import grid_tools as G
import evolution_memory as ER

scenario = sys.argv[1] if len(sys.argv) > 1 else "cloudy_084"
net, scenarios = G.load_grid()
print(
    f"grid: {len(net.bus)} buses, {len(net.line)} lines, {len(net.trafo)} transformers | {len(scenarios)} scenarios"
)
state = G.call_tool("analyse", {"profile": scenario})
print(
    f"{scenario}: Security5 {state['security5']['vector']} | "
    f"{len(state.get('n1_overloaded_elements') or [])} N-1 overloads"
)
folds = json.load(open("memory/folds.json"))
fold = next(f for f, v in folds.items() if scenario in v["eval"])
episodes = len(os.listdir(f"memory/{fold}/episodic/episodes"))
print(
    f"memory: {fold} ({episodes} source episodes; {scenario} is not among them: "
    f"{scenario not in folds[fold]['source']})"
)


class Session:
    profile = scenario

    def record(self, *a, **k):
        pass


ER.ACTIVE = Session()
ER.set_strategy_bank(f"memory/{fold}/strategies.json")
for c in "BAC":
    pk = ER.procedural_packet(c, state)
    lines = (pk.get("scope2_strategy_applied_to_current_diagnostics") or {}).get(
        "lines"
    ) or []
    ex = lines[0] if lines else {}
    print(
        f"candidate {c}: {len(pk.get('scope2_strategies', []))} strategies, sizing targets for "
        f"{len(lines)} lines"
        + (
            f" (e.g. {ex.get('line')}: {ex.get('target_mva_for_this_candidate')} MVA)"
            if ex
            else ""
        )
    )
print("smoke test passed")
