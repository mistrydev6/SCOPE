from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path


from vllm_backend_full import chat_vllm
import grid_tools as G
import evolution_memory as ER
from prompts import (
    INITIAL_DIAGNOSIS_INSTRUCTION,
    RECALL_STAGE_INSTRUCTION,
    SELECTION_STAGE_INSTRUCTION,
    AGENT_SELECTION_SYSTEM,
    SEQUENTIAL_SYSTEMS,
    SEQUENTIAL_REPAIR,
    DIAGNOSIS_SYSTEM,
    FUSION_SYSTEM,
    HISTORICAL_PRECEDENT_NOTE,
)

PROJECT_DIR = Path(__file__).resolve().parent
MEMORY_DIR = PROJECT_DIR / "memory"
RUNS_DIR = PROJECT_DIR / "runs"


def _fd_catalog(packet):
    raw = packet.get("action_catalog")
    if isinstance(raw, dict) and raw:
        return {
            str(ref): entry for ref, entry in raw.items() if isinstance(entry, dict)
        }

    rows = packet.get("catalog") or []
    if isinstance(rows, list):
        return {
            str(row["ref"]): {
                "candidate_id": row["source"],
                "equipment_key": row["equipment"],
                "action": row["action"],
            }
            for row in rows
            if isinstance(row, dict)
            and all(k in row for k in ("ref", "source", "equipment", "action"))
        }
    return {}


def _fd_equipment(catalog, ref):
    key = catalog[ref].get("equipment_key")
    if not key:
        raise ValueError(f"Missing equipment key for {ref}")
    return tuple(key)


def _fd_signature(catalog, refs):
    items = []
    for ref in refs:
        entry = catalog[ref]
        items.append(
            json.dumps(
                [list(_fd_equipment(catalog, ref)), entry.get("action")],
                sort_keys=True,
                default=str,
            )
        )
    return tuple(sorted(items))


def _fd_legal_options(packet, limit=48):
    packet = packet or {}
    profile = packet.get("profile")
    round_id = packet.get("round_id")
    catalog = _fd_catalog(packet)

    if not profile or not round_id:
        raise RuntimeError("MANDATORY_D_CONTEXT_ERROR: missing profile or round_id")
    if not catalog:
        raise RuntimeError("MANDATORY_D_CONTEXT_ERROR: missing action catalog")

    by_source = {
        cid: [
            ref
            for ref, entry in catalog.items()
            if entry.get("candidate_id") == cid and entry.get("equipment_key")
        ]
        for cid in ("A", "B", "C")
    }

    complete_signatures = {_fd_signature(catalog, refs) for refs in by_source.values()}

    rows = packet.get("candidate_results") or packet.get("siblings") or []
    results = {
        row.get("candidate_id", row.get("id")): row
        for row in rows
        if isinstance(row, dict)
    }

    secure = {
        cid for cid in ("A", "B", "C") if results.get(cid, {}).get("solved") is True
    }

    def cost(cid):
        row = results.get(cid, {})
        value = row.get("plan_cost_musd", row.get("cost_musd"))
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("inf")

    if secure:
        anchors = sorted(
            ("A", "B", "C"),
            key=lambda cid: (cid not in secure, cost(cid), cid),
        )
        objective = "economic"
    else:

        anchors = list(("A", "B", "C"))
        objective = "security-closure"

    proposals = []
    seen = set()

    def offer(refs, anchor, removed, imported, priority):
        if not refs:
            return

        if len(refs) != len(set(refs)):
            return

        try:
            keys = [_fd_equipment(catalog, ref) for ref in refs]
        except (KeyError, ValueError, TypeError):
            return

        if len(set(keys)) != len(keys):
            return

        sources = {catalog[ref].get("candidate_id") for ref in refs}
        if len(sources) < 2:
            return

        signature = _fd_signature(catalog, refs)

        if signature in complete_signatures or signature in seen:
            return
        seen.add(signature)

        args = {
            "profile": profile,
            "round_id": round_id,
            "action_refs": list(refs),
            "strategy": (
                f"{objective}: exact-action D based on {anchor}; "
                f"import {imported}, omit {removed}. "
                "Security and cost require full AC/N-1 verification."
            ),
        }
        proposals.append((priority, args))

    for rank, anchor in enumerate(anchors):
        original = by_source[anchor]

        for source in ("A", "B", "C"):
            if source == anchor:
                continue

            for external in by_source[source]:
                external_key = _fd_equipment(catalog, external)

                for removed in [None] + original:
                    retained = [
                        ref
                        for ref in original
                        if ref != removed
                        and _fd_equipment(catalog, ref) != external_key
                    ]
                    candidate = retained + [external]

                    is_pruning = removed is not None
                    is_complementary = external_key not in {
                        _fd_equipment(catalog, ref) for ref in original
                    }

                    if objective == "economic":
                        priority = (
                            rank,
                            0 if is_pruning else 1,
                            len(candidate),
                        )
                    else:
                        priority = (
                            rank,
                            0 if is_complementary else 1,
                            0 if not is_pruning else 1,
                        )

                    offer(
                        candidate,
                        anchor,
                        [] if removed is None else [removed],
                        [external],
                        priority,
                    )

    refs = list(catalog)
    for left, right in itertools.combinations(refs, 2):
        if catalog[left].get("candidate_id") == catalog[right].get("candidate_id"):
            continue
        if _fd_equipment(catalog, left) == _fd_equipment(catalog, right):
            continue

        offer(
            [left, right],
            "two-source subset",
            [],
            [left, right],
            (1000, 0, 0),
        )

    proposals.sort(key=lambda pair: pair[0])
    return [args for _, args in proposals[: max(1, int(limit))]]


def _fc_security(item):
    raw = item.get("security5") or {}
    values = raw.get("vector") if isinstance(raw, dict) else raw
    if not isinstance(values, (list, tuple)) or len(values) != 5:
        raise RuntimeError("Missing verified sibling Security5")
    vals = [float(x) for x in values]
    if not all(math.isfinite(x) for x in vals):
        raise RuntimeError("Invalid sibling Security5")
    return vals


def _fc_number(value, digits=4):
    try:
        value = float(value)
        return round(value, digits) if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _fc_diagnostics(rows, limit):
    keys = (
        "element",
        "bus",
        "hour",
        "loading_pct",
        "worst_loading_pct",
        "rating_mva",
        "worst_vm_pu",
        "min_vm_pu",
        "max_vm_pu",
        "outage_count",
        "n_outage_tests",
        "example_outage",
        "outage",
    )
    compact = []
    for row in (rows or [])[:limit]:
        if isinstance(row, dict):
            entry = {k: row[k] for k in keys if k in row and row[k] is not None}
            if entry:
                compact.append(entry)
    return compact


def _fc_target(action):
    for field in (
        "target_capacity_mva",
        "target_sn_mva",
        "target_mvar",
        "capacity_mva",
    ):
        if field in action:
            return action[field]
    return None


def _fusion_build_messages(profile, pending_result, retry_feedback=None):
    pending = pending_result or {}
    catalog = pending.get("action_catalog") or {}
    round_id = pending.get("round_id")

    if not round_id or not catalog:
        raise RuntimeError("Incomplete mandatory D catalog")

    by_id = {
        str(row.get("candidate_id")): row
        for row in pending.get("candidate_results") or []
        if isinstance(row, dict)
    }
    if set(by_id) != {"A", "B", "C"}:
        raise RuntimeError("Exactly three verified siblings required")

    siblings = []
    vectors = []
    secure_costs = []

    for cid in "ABC":
        row = by_id[cid]
        s5 = _fc_security(row)
        cost = _fc_number(row.get("plan_cost_musd"))
        if cost is None:
            raise RuntimeError(f"Missing verified cost for {cid}")

        solved = row.get("solved") is True
        if solved:
            secure_costs.append(cost)
        vectors.append(s5)

        siblings.append(
            {
                "id": cid,
                "solved": solved,
                "cost_M": cost,
                "S5": s5,
                "N1_overloads": _fc_diagnostics(row.get("n1_overloaded_elements"), 3),
                "N1_voltage": _fc_diagnostics(row.get("n1_voltage_problem_buses"), 2),
            }
        )

    measurements = {
        str(row["action_ref"]): row
        for row in pending.get("action_evaluations") or []
        if isinstance(row, dict) and row.get("action_ref")
    }

    groups = defaultdict(list)

    for ref, entry in catalog.items():
        key = entry.get("equipment_key")
        action = entry.get("action")
        source = entry.get("candidate_id")

        if not isinstance(key, (list, tuple)) or not key:
            raise RuntimeError(f"Invalid equipment for {ref}")
        if not isinstance(action, dict):
            raise RuntimeError(f"Invalid action for {ref}")
        if source not in ("A", "B", "C"):
            raise RuntimeError(f"Invalid source for {ref}")

        row = measurements.get(ref) or {}
        option = {
            "ref": ref,
            "source": source,
            "target": _fc_target(action),
        }

        added_cost = _fc_number(row.get("added_cost_musd"))
        if added_cost is not None:
            option["individual_cost_M"] = added_cost

        delta = row.get("security5_change_vs_parent")
        if isinstance(delta, (list, tuple)) and len(delta) == 5:

            option["individual_security5_change"] = [
                _fc_number(delta[j], 6) for j in range(5)
            ]

        if row.get("evaluation_status") == "error":
            option["individual_evaluation_error"] = True

        groups[tuple(key)].append(option)

    genes = [
        {
            "equipment": list(key),
            "options": sorted(options, key=lambda x: x["ref"]),
        }
        for key, options in sorted(groups.items(), key=lambda kv: repr(kv[0]))
    ]

    secure_budget = min(secure_costs) if secure_costs else None

    packet = {
        "profile": profile,
        "round_id": round_id,
        "mode": (
            "CHEAPER_SECURE_D" if secure_budget is not None else "SECURITY_DOMINATING_D"
        ),
        "cheapest_secure_cost_M": secure_budget,
        "componentwise_S5_target": [min(v[j] for v in vectors) for j in range(5)],
        "parents": siblings,
        "retained_grid": {
            "N1_overloads": _fc_diagnostics(pending.get("n1_overloaded_elements"), 10),
            "N1_voltage": _fc_diagnostics(pending.get("n1_voltage_problem_buses"), 6),
            "base_overloads": _fc_diagnostics(
                pending.get("base_case_overloaded_elements"), 5
            ),
            "base_voltage": _fc_diagnostics(
                pending.get("base_case_voltage_problem_buses"), 4
            ),
        },
        "equipment_choices": genes,
        "retained_plan": pending.get("current_plan", []),
        "legal_d_options": _fd_legal_options({**pending, "profile": profile}),
        "previous_error": retry_feedback,
    }

    content = json.dumps(packet, separators=(",", ":"), default=str)
    print(
        f"[v32d-compact] profile={profile} round={round_id} "
        f"genes={len(genes)} refs={len(catalog)} "
        f"packet_chars={len(content)} "
        f"mode={packet['mode']}",
        flush=True,
    )

    return [
        {"role": "system", "content": FUSION_SYSTEM},
        {"role": "user", "content": content},
    ]


def _fg_signature(catalog, refs):

    return frozenset(
        (
            json.dumps(
                catalog[ref]["equipment_key"],
                sort_keys=True,
                default=str,
            ),
            json.dumps(
                catalog[ref]["action"],
                sort_keys=True,
                default=str,
            ),
        )
        for ref in refs
    )


def _fusion_validate_choice(args, pending):
    if not isinstance(args, dict):
        return "D tool arguments must be an object"

    if "anchor_candidate" in args:
        return "Use full action_refs gene choices, not an anchor edit"

    refs = args.get("action_refs")
    catalog = (pending or {}).get("action_catalog") or {}

    if not isinstance(refs, list) or not refs:
        return "action_refs must contain the complete D chromosome"
    if any(not isinstance(ref, str) for ref in refs):
        return "Every selected reference must be a string"
    if len(refs) != len(set(refs)):
        return "Repeated action reference; choose each ref once"

    unknown = [ref for ref in refs if ref not in catalog]
    if unknown:
        return f"Unknown refs: {unknown}"

    seen = {}
    sources = set()

    for ref in refs:
        entry = catalog[ref]
        key = tuple(entry.get("equipment_key") or ())
        if not key:
            return f"Missing physical equipment key for {ref}"

        if key in seen:
            return (
                f"Equipment conflict: {seen[key]} and {ref} "
                f"select the same equipment {list(key)}. "
                "Choose only ONE of those refs."
            )
        seen[key] = ref
        sources.add(str(entry.get("candidate_id") or "").upper())

    if len(sources) < 2 or not sources <= {"A", "B", "C"}:
        return "D requires exact refs from at least two of A/B/C"

    d_signature = _fg_signature(catalog, refs)

    for cid in ("A", "B", "C"):
        parent_refs = [
            ref for ref, entry in catalog.items() if entry.get("candidate_id") == cid
        ]
        if d_signature == _fg_signature(catalog, parent_refs):
            return (
                f"D is a physical copy of Candidate {cid}. "
                "Select a genuinely different combination."
            )

    return None


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
MODEL = "qwen3.5:9b"
MAX_PLANNING_ROUNDS = int(os.environ.get("SCOPE_MAX_PLANNING_ROUNDS", "30"))
MAX_LLM_TURNS = int(os.environ.get("SCOPE_MAX_LLM_TURNS", "240"))
MAX_STAGE_ATTEMPTS = max(1, int(os.environ.get("SCOPE_MAX_STAGE_ATTEMPTS", "6")))
MAX_CROSSOVER_ATTEMPTS = max(
    5, int(os.environ.get("SCOPE_MAX_CROSSOVER_ATTEMPTS", "5"))
)
MAX_SELECTION_ATTEMPTS = max(
    1, int(os.environ.get("SCOPE_MAX_SELECTION_ATTEMPTS", "3"))
)
THINK_BUDGET = 2400
CANDIDATE_NUM_PREDICT = max(
    1000, int(os.environ.get("SCOPE_CANDIDATE_NUM_PREDICT", "4200"))
)
CANDIDATE_REPAIR_NUM_PREDICT = max(
    300, int(os.environ.get("SCOPE_CANDIDATE_REPAIR_NUM_PREDICT", "3200"))
)
CROSSOVER_NUM_PREDICT = max(
    700, int(os.environ.get("SCOPE_CROSSOVER_NUM_PREDICT", "2400"))
)
CROSSOVER_REPAIR_NUM_PREDICT = max(
    256, int(os.environ.get("SCOPE_CROSSOVER_REPAIR_NUM_PREDICT", "480"))
)
SELECTION_NUM_PREDICT = int(os.environ.get("SCOPE_SELECTION_NUM_PREDICT", "800"))
RECALL_NUM_PREDICT = 700
DIAGNOSIS_NUM_PREDICT = 900

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def _historical_plan_to_candidate_actions(final_plan):
    actions = []
    for item in final_plan or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("action")
        if kind == "upgrade_line" and item.get("line"):
            target = item.get("to_capacity_mva", item.get("target_capacity_mva"))
            if target is not None:
                actions.append(
                    {
                        "action": "upgrade_line",
                        "line": item["line"],
                        "target_capacity_mva": float(target),
                    }
                )
        elif kind == "upgrade_transformer" and item.get("transformer"):
            target = item.get("to_sn_mva", item.get("target_sn_mva"))
            if target is not None:
                actions.append(
                    {
                        "action": "upgrade_transformer",
                        "transformer": item["transformer"],
                        "target_sn_mva": float(target),
                    }
                )
        elif kind == "set_bus_reactive_support" and item.get("bus"):
            target = item.get("to_reactive_support_mvar", item.get("target_mvar"))
            if target is not None:
                actions.append(
                    {
                        "action": "set_bus_reactive_support",
                        "bus": item["bus"],
                        "target_mvar": float(target),
                    }
                )
        elif kind == "new_line":
            from_bus = item.get("from_bus")
            to_bus = item.get("to_bus")
            capacity = item.get("capacity_mva", item.get("to_capacity_mva"))
            if from_bus and to_bus and capacity is not None:
                actions.append(
                    {
                        "action": "new_line",
                        "from_bus": from_bus,
                        "to_bus": to_bus,
                        "capacity_mva": float(capacity),
                    }
                )
    return actions


def _compact_recall_for_model(result):
    if not isinstance(result, dict) or not result.get("found"):
        return result

    retrieval = result.get("retrieval") or {}
    outcome = result.get("historical_outcome") or {}
    initial_state = result.get("historical_initial_state") or {}
    security5 = initial_state.get("security5") or {}
    final_plan = result.get("historical_final_plan") or []
    actions = _historical_plan_to_candidate_actions(final_plan)

    return {
        "found": True,
        "retrieval": {
            "rank": retrieval.get("rank"),
            "episode_id": retrieval.get("episode_id"),
            "profile": retrieval.get("profile"),
            "metric": retrieval.get("metric"),
            "l2_distance": retrieval.get("l2_distance"),
        },
        "historical_outcome": {
            "solved": bool(outcome.get("solved")),
            "final_cost_musd": outcome.get("final_cost_musd"),
        },
        "historical_initial_security5": security5.get("vector", []),
        "candidate_a_seed": {
            "source_episode_id": retrieval.get("episode_id"),
            "source_profile": retrieval.get("profile"),
            "actions": actions,
        },
        "note": (
            "The historical actions are retrieval evidence for Candidate A, not a required plan. "
            "Use them as inspiration and verify any adapted action on the current scenario."
        ),
    }


def _latest_tool_payload(messages, tool_name: str) -> dict:
    for message in reversed(messages):
        if message.get("role") != "tool" or message.get("name") != tool_name:
            continue
        content = message.get("content")
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            try:
                value = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return {}


def _compact_current_planning_state(state: dict) -> dict:
    if not isinstance(state, dict):
        return {}
    out = {
        "security5": state.get("security5", {}),
        "peak_hour": state.get("peak_hour"),
        "n_base_case_problem_hours": state.get("n_base_case_problem_hours"),
        "nonconverged_base_case_hours": state.get("nonconverged_base_case_hours", []),
        "n_failed_n1_outage_tests": state.get("n_failed_n1_outage_tests"),
        "n_n1_outage_tests": state.get("n_n1_outage_tests"),
        "plan_cost_musd": state.get("plan_cost_musd", 0.0),
        "base_case_overloaded_elements": state.get("base_case_overloaded_elements", []),
        "base_case_voltage_problem_buses": state.get(
            "base_case_voltage_problem_buses", []
        ),
        "n1_overloaded_elements": state.get("n1_overloaded_elements", []),
        "n1_voltage_problem_buses": state.get("n1_voltage_problem_buses", []),
        "base_case_worst_hours": state.get("base_case_worst_hours", []),
    }
    current_plan = state.get("current_plan")
    if current_plan:
        out["retained_current_plan"] = current_plan
    return {k: v for k, v in out.items() if v not in (None, [], {})}


def _candidate_a_inspiration(
    current_state: dict,
    recall_result: dict,
    max_thermal_examples: int = 5,
    max_voltage_examples: int = 3,
) -> dict:
    if not isinstance(recall_result, dict) or not recall_result.get("found"):
        return {}

    seed = (recall_result.get("candidate_a_seed") or {}).get("actions") or []
    if not seed:
        return {}

    line_actions = {}
    bus_actions = {}
    for action in seed:
        if not isinstance(action, dict):
            continue
        kind = action.get("action")
        if kind == "upgrade_line" and action.get("line"):
            line_actions.setdefault(action["line"], action)
        elif kind == "upgrade_transformer" and action.get("transformer"):
            line_actions.setdefault(action["transformer"], action)
        elif kind == "set_bus_reactive_support" and action.get("bus"):
            bus_actions.setdefault(action["bus"], action)

    thermal_examples = []
    seen_equipment = set()
    for source_key, id_key in (
        ("n1_overloaded_elements", "element"),
        ("base_case_overloaded_elements", "element"),
    ):
        for item in current_state.get(source_key) or []:
            equipment = item.get(id_key)
            if not equipment or equipment in seen_equipment:
                continue
            action = line_actions.get(equipment)
            if action is None:
                continue
            seen_equipment.add(equipment)
            example = {
                "equipment": equipment,
                "historical_action": action,
            }
            if item.get("worst_loading_pct") is not None:
                example["current_worst_loading_pct"] = item.get("worst_loading_pct")
            elif item.get("loading_pct") is not None:
                example["current_loading_pct"] = item.get("loading_pct")
            thermal_examples.append(example)
            if len(thermal_examples) >= max(1, int(max_thermal_examples)):
                break
        if len(thermal_examples) >= max(1, int(max_thermal_examples)):
            break

    voltage_examples = []
    seen_buses = set()
    for source_key in ("n1_voltage_problem_buses", "base_case_voltage_problem_buses"):
        for item in current_state.get(source_key) or []:
            bus = item.get("bus")
            if not bus or bus in seen_buses:
                continue
            action = bus_actions.get(bus)
            if action is None:
                continue
            seen_buses.add(bus)
            example = {
                "bus": bus,
                "historical_action": action,
            }
            if item.get("worst_vm_pu") is not None:
                example["current_worst_vm_pu"] = item.get("worst_vm_pu")
            elif item.get("min_vm_pu") is not None:
                example["current_min_vm_pu"] = item.get("min_vm_pu")
            voltage_examples.append(example)
            if len(voltage_examples) >= max(1, int(max_voltage_examples)):
                break
        if len(voltage_examples) >= max(1, int(max_voltage_examples)):
            break

    retrieval = recall_result.get("retrieval") or {}
    outcome = recall_result.get("historical_outcome") or {}
    return {
        "source_episode_id": retrieval.get("episode_id"),
        "source_profile": retrieval.get("profile"),
        "historical_initial_security5": recall_result.get(
            "historical_initial_security5", []
        ),
        "historical_final_cost_musd": outcome.get("final_cost_musd"),
        "intervention_pattern": (HISTORICAL_PRECEDENT_NOTE),
        "matched_thermal_examples": thermal_examples,
        "matched_voltage_examples": voltage_examples,
        "usage_note": (
            "Candidate A may reuse, retarget, ignore, or supplement these few examples. "
            "Choose A from CURRENT diagnostics; do not reconstruct the historical final plan."
        ),
    }


def _compact_crossover_candidate(item: dict) -> dict:
    if not isinstance(item, dict):
        return {}
    return {
        "candidate_id": item.get("candidate_id"),
        "strategy": item.get("strategy", ""),
        "security5": item.get("security5", {}),
        "plan_cost_musd": item.get("plan_cost_musd"),
        "improving": bool(item.get("improving", False)),
        "solved": bool(item.get("solved", False)),
        "security5_relation_vs_parent": item.get("security5_relation_vs_parent"),
        "fusion_superior_to_all_siblings": item.get("fusion_superior_to_all_siblings"),
        "fusion_comparison": item.get("fusion_comparison", []),
    }


def _selection_stage_messages(
    profile: str,
    pending_result: dict | None,
) -> list[dict]:
    pending = pending_result or {}
    eligible = [
        str(x)
        for x in (pending.get("eligible_candidate_ids") or [])
        if str(x) in {"A", "B", "C", "D"}
    ]

    packet = {
        "profile": profile,
        "eligible_candidate_ids": eligible,
        "parent_security5": pending.get("parent_score") or pending.get("security5"),
        "candidate_results": [
            _compact_crossover_candidate(item)
            for item in (pending.get("candidate_results") or [])
        ],
        "selection_rule": pending.get("selection_rule"),
        "requirements": [
            "Use only an ID in eligible_candidate_ids.",
            "Use verified Security5 outcomes; lower is better componentwise.",
            (
                "When solved candidates exist, select the lowest-cost "
                "verified solved candidate. Otherwise balance verified "
                "Security5 improvement against total investment."
            ),
            "Call commit_candidate immediately after one brief comparison.",
        ],
    }

    return [
        {"role": "system", "content": AGENT_SELECTION_SYSTEM},
        {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
    ]


def _selected_sequential_system(candidate_id):
    return (
        "Each round verifies A, then B, then C in separate calls from the same retained parent. "
        "Submit only the candidate assigned to this call.\n\n"
        + SEQUENTIAL_SYSTEMS[candidate_id]
    )


def _candidate_stage_messages(
    messages: list[dict],
    profile: str,
    parent_result: dict | None,
    candidate_id: str,
    retry_feedback: str | None,
) -> list[dict]:
    analyse_result = _latest_tool_payload(messages, "analyse")
    current = (
        parent_result
        if isinstance(parent_result, dict) and parent_result.get("security5")
        else analyse_result
    )
    packet = {
        "profile": profile,
        "candidate_id": candidate_id,
        "parent_state": _compact_current_planning_state(current),
        "retry_feedback": retry_feedback,
    }
    if candidate_id == "A" and parent_result is None:
        recall = _latest_tool_payload(messages, "recall_experience")
        if recall.get("found"):
            packet["optional_historical_inspiration"] = _candidate_a_inspiration(
                current, recall
            )
    packet.update(ER.procedural_packet(candidate_id, current))
    return [
        {"role": "system", "content": _selected_sequential_system(candidate_id)},
        {"role": "user", "content": json.dumps(packet, separators=(",", ":"))},
    ]


def _recover_explicit_sequential_plan(out: dict, profile: str, candidate_id: str):
    parts = [str(out.get(k) or "") for k in ("thinking", "content")]
    source = "\n".join(parts)
    decoder = json.JSONDecoder()
    candidates = []
    for index, character in enumerate(source):
        if character != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(source[index:])
        except (ValueError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("profile") != profile or obj.get("candidate_id") != candidate_id:
            continue
        acts = obj.get("actions")
        if (
            not isinstance(acts, list)
            or not acts
            or not all(isinstance(a, dict) for a in acts)
        ):
            continue

        required = {
            "upgrade_line": ("line", "target_capacity_mva"),
            "upgrade_transformer": ("transformer", "target_sn_mva"),
            "new_line": ("from_bus", "to_bus", "capacity_mva"),
            "set_bus_reactive_support": ("bus", "target_mvar"),
        }
        if not all(
            a.get("action") in required
            and all(a.get(k) is not None for k in required[a["action"]])
            for a in acts
        ):
            continue
        candidates.append(obj)
    if not candidates:
        return None

    plan = candidates[-1]
    return [
        {
            "type": "function",
            "function": {
                "name": "test_staged_candidate",
                "arguments": {
                    "profile": profile,
                    "candidate_id": candidate_id,
                    "strategy": str(plan.get("strategy") or "independent candidate"),
                    "actions": plan["actions"],
                },
            },
        }
    ]


def _sequential_stage_tool(tools: list[dict], parent: dict | None) -> list[dict]:
    if (
        len(tools) != 1
        or (tools[0].get("function") or {}).get("name") != "test_staged_candidate"
    ):
        return tools
    packet = (parent or {}).get("parent_state") or {}
    evidence = (packet.get("base_case_overloaded_elements") or []) + (
        packet.get("n1_overloaded_elements") or []
    )
    has_transformer = any(
        "transformer" in str(x.get("element_type") or "").lower()
        or "trafo" in str(x.get("element_type") or "").lower()
        for x in evidence
        if isinstance(x, dict)
    )
    result = copy.deepcopy(tools)
    properties = result[0]["function"]["parameters"]["properties"]
    properties["candidate_id"]["enum"] = [parent["candidate_id"]]
    fields = properties["actions"]["items"]["properties"]
    if parent["candidate_id"] != "C":
        fields["action"]["enum"].remove("new_line")
    if not has_transformer:
        fields["action"]["enum"].remove("upgrade_transformer")
        fields.pop("transformer", None)
        fields.pop("target_sn_mva", None)
    return result


def _normalise_staged_action_array(calls):
    if len(calls) != 1:
        return False
    fn = calls[0].get("function")
    if not isinstance(fn, dict):
        return False
    args, error = _decode_tool_arguments(calls[0])
    if error or not isinstance(args, dict):
        return False
    raw = args.get("actions")
    if not isinstance(raw, str):
        return False
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        return False
    if not isinstance(decoded, list):
        return False
    args["actions"] = decoded
    fn["arguments"] = json.dumps(args)
    return True


def _validate_sequential_native_call(
    calls: list,
    profile: str,
    candidate_id: str,
    staged_tool: dict | None = None,
) -> str:
    if len(calls) != 1:
        return "Exactly one native test_staged_candidate tool call is required."
    call = calls[0]
    if (call.get("function") or {}).get("name") != "test_staged_candidate":
        return "Only test_staged_candidate is allowed in this stage."
    args, error = _decode_tool_arguments(call)
    if error:
        return error
    if args.get("profile") != profile or args.get("candidate_id") != candidate_id:
        return "Use the locked profile and candidate_id, without changing either."
    actions = args.get("actions")
    if (
        not isinstance(actions, list)
        or not actions
        or not all(isinstance(a, dict) for a in actions)
    ):
        return "Supply a nonempty actions array of action objects."

    if actions:
        for action in actions:
            if action.get("action") != "upgrade_line":
                continue
            line = action.get("line")
            if (
                isinstance(line, str)
                and line.startswith("Line_Bus")
                and "#" not in line
            ):
                return (
                    f"Ambiguous line identifier {line!r}. "
                    "The full physical circuit identifier must include "
                    "its exact #NN suffix from the current diagnostic "
                    "equipment ID. Parallel circuits are distinct. "
                    "Correct your own tool call; never guess, strip the "
                    "suffix, or reuse an unchanged parent target."
                )

    if staged_tool:
        try:
            allowed_kinds = set(
                staged_tool["function"]["parameters"]["properties"]["actions"]["items"][
                    "properties"
                ]["action"]["enum"]
            )
        except (KeyError, TypeError):
            allowed_kinds = set()
        if allowed_kinds:
            bad = [
                a.get("action") for a in actions if a.get("action") not in allowed_kinds
            ]
            if bad:
                return (
                    "Unsupported action type for this parent: "
                    f"{bad!r}. Do not invent transformers; use valid observed equipment."
                )
    return ""


def chat(
    messages,
    model,
    stream=True,
    available_tools=None,
    stage_instruction=None,
    num_predict=None,
    *,
    enable_thinking=True,
    temperature=0.3,
    top_p=None,
    top_k=None,
) -> dict:
    messages = list(messages)

    request_messages = [dict(message) for message in messages]
    if stage_instruction:
        if request_messages and request_messages[0].get("role") == "system":
            base_system = str(request_messages[0].get("content") or "").rstrip()
            request_messages[0]["content"] = (
                base_system + "\n\n" + str(stage_instruction).strip()
            )
        else:
            request_messages.insert(
                0,
                {"role": "system", "content": str(stage_instruction).strip()},
            )
    tools = G.TOOLS if available_tools is None else available_tools
    token_budget = THINK_BUDGET + 1200 if num_predict is None else int(num_predict)

    audit_payload = {
        "backend": "vllm",
        "base_url": VLLM_BASE_URL,
        "model": model,
        "messages": request_messages,
        "stream": bool(stream),
        "think": bool(enable_thinking),
        "options": {
            "temperature": float(temperature),
            "num_predict": token_budget,
            "top_p": top_p,
            "top_k": top_k,
        },
        "tools": tools,
    }

    generation_id = ER.before_chat(audit_payload)

    try:
        result = chat_vllm(
            request_messages,
            model,
            tools=tools,
            stream=stream,
            temperature=float(temperature),
            max_tokens=token_budget,
            reasoning_effort=("low" if enable_thinking else "none"),
            enable_thinking=bool(enable_thinking),
            top_p=top_p,
            top_k=top_k,
            timeout=900.0,
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            print_reasoning=True,
            dim=DIM,
            reset=RESET,
        )
    except Exception as exc:
        raise SystemExit(
            "Cannot complete vLLM chat request.\n"
            f"  endpoint: {VLLM_BASE_URL}\n"
            f"  model:    {model}\n"
            "Make sure the vLLM server is running with Qwen3.5 reasoning "
            "and tool-call parsers enabled.\n"
            f"  error: {exc}"
        ) from exc

    metadata = {
        "done_reason": result.get("finish_reason"),
        "eval_count": result.get("completion_tokens"),
        "prompt_eval_count": result.get("prompt_tokens"),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}

    response = {
        "thinking": result.get("thinking", ""),
        "content": result.get("content", ""),
        "tool_calls": result.get("tool_calls", []),
        "generation": metadata,
    }
    ER.after_chat(generation_id, response)
    return response


def _extract_explicit_candidate_choice(
    out: dict, eligible_ids: list[str]
) -> str | None:
    eligible = {str(x).strip().upper() for x in eligible_ids}
    if not eligible:
        return None
    text = "\n".join(
        part
        for part in (str(out.get("thinking") or ""), str(out.get("content") or ""))
        if part
    )
    text = re.sub(r"[*_`]", "", text)
    patterns = (
        r"\bI(?:'ll|\s+will)?\s+(?:choose|select|commit)\s+(?:Candidate\s+)?([ABCD])\b",
        r"\bmy\s+(?:choice|selection|decision)\s+is\s+(?:Candidate\s+)?([ABCD])\b",
        r"\b(?:choose|select|commit)\s+(?:Candidate\s+)?([ABCD])\b",
    )
    choices = []
    for pattern in patterns:
        choices.extend(
            m.group(1).upper() for m in re.finditer(pattern, text, flags=re.I)
        )
    valid = [choice for choice in choices if choice in eligible]
    if not valid or len(set(valid)) != 1:
        return None
    return valid[-1]


def brief(name: str, result: dict) -> str:
    if "error" in result:
        return f"ERROR: {result['error']}"

    if name == "scan_day":
        return (
            f"peak hour {result['peak_hour']}, "
            f"{result['n_base_case_problem_hours']}/24 base-case hours have problems"
        )

    if name == "analyse":
        sec = (result.get("security5") or {}).get("vector", [])
        sec_text = (
            "[" + ", ".join(f"{float(x):.4f}" for x in sec) + "]"
            if len(sec) == 5
            else str(sec)
        )
        return (
            f"whole day Security5={sec_text} | "
            f"{result.get('n_base_case_problem_hours', 0)}/24 base-case hours with diagnostic violations"
        )
    if name == "recall_experience":
        if not result.get("found"):
            return result.get(
                "message",
                "no prior experience found",
            )

        retrieval = result["retrieval"]
        outcome = result["historical_outcome"]

        distance = retrieval.get("l2_distance")
        metric = retrieval.get(
            "metric",
            "similarity",
        )

        if distance is None:
            distance_text = "distance=n/a"
        else:
            distance_text = f"distance={float(distance):.6f}"

        return (
            f"rank {retrieval['rank']}: "
            f"{retrieval['profile']} "
            f"({retrieval['episode_id']}, "
            f"metric={metric}, "
            f"{distance_text}, "
            f"solved={outcome['solved']})"
        )

    if name == "test_staged_candidate":
        if result.get("stage_verified"):
            pf = result.get("full_plan_pf") or {}
            vector = (pf.get("security5") or {}).get("vector") or []
            sec_text = (
                "[" + ", ".join(f"{float(x):.4f}" for x in vector) + "]"
                if len(vector) == 5
                else "unavailable"
            )
            return (
                f"Candidate {result['staged_candidate_id']} FULL PLAN AC PF "
                f"Security5={sec_text}, "
                f"cost=${pf.get('plan_cost_musd', 'n/a')}M "
                f"(added=${pf.get('added_cost_musd', 'n/a')}M), "
                f"solved={pf.get('solved', 'n/a')}, "
                f"relation={pf.get('security5_relation_vs_parent', 'n/a')}, "
                f"base_problem_hours={pf.get('n_base_case_problem_hours', 'n/a')}, "
                f"N-1_failures={pf.get('n_failed_n1_outage_tests', 'n/a')} "
                f"({result.get('action_verifier_evals', 0)} independent action PF + "
                "1 full-plan PF); parent unchanged; evidence saved for D"
            )

    if name in (
        "test_staged_candidate",
        "commit_candidate",
        "test_crossover_plan",
    ):
        tag = (
            "*** SOLVED ***"
            if result["solved"]
            else (
                "REVERTED"
                if result["search_decision"] == "reverted"
                else result["outcome_label"].upper()
            )
        )
        selected = ""

        if name in ("test_staged_candidate", "test_crossover_plan"):
            if result.get("crossover_required"):
                selected = " | D crossover required; none committed"

            elif result.get("selection_required"):
                selected = (
                    " | agent selection required among "
                    f"{result.get('eligible_candidate_ids', [])}; "
                    "none committed"
                )

            else:
                cid = result.get("selected_candidate_id", "?")
                committed = result.get("selected_candidate_committed", False)
                selected = (
                    f" | selected={cid} "
                    f"({'committed' if committed else 'not committed'})"
                )

        elif name == "commit_candidate":
            cid = result.get("selected_candidate_id", "?")
            selected = f" | agent chose {cid} (committed)"
        return (
            f"{tag}{selected} | plan: {result['n_plan_actions']} actions, "
            f"${result['plan_cost_musd']}M"
            + (
                " | Security5=["
                + ", ".join(
                    f"{float(x):.4f}"
                    for x in (result.get("security5") or {}).get("vector", [])
                )
                + "]"
                if len((result.get("security5") or {}).get("vector", [])) == 5
                else ""
            )
        )

    return "ok"


def _format_number(value) -> str:
    try:
        f = float(value)
        return str(int(f)) if f.is_integer() else f"{f:g}"
    except Exception:
        return str(value)


def format_verified_findings(result: dict) -> str:
    plan = result.get("final_plan") or []
    counts = Counter(item.get("action", "unknown") for item in plan)

    lines = [
        "**SOLVED — simulator verified**",
        "",
        f"Final plan cost: **${result['plan_cost_musd']}M**",
        f"Final plan actions: **{len(plan)}**",
        "",
        "Action counts:",
        f"- Line rating changes: {counts.get('upgrade_line', 0)}",
        f"- Transformer rating changes: {counts.get('upgrade_transformer', 0)}",
        f"- New lines: {counts.get('new_line', 0)}",
        f"- Bus reactive-support setpoints: {counts.get('set_bus_reactive_support', 0)}",
        "",
        "Final simulator state:",
        "- Security5: [0, 0, 0, 0, 0]",
        "- All five Security5 security components are zero",
    ]

    if plan:
        lines += ["", "Final plan transitions:"]
        for item in plan:
            kind = item.get("action")
            if kind == "upgrade_line":
                lines.append(
                    f"- {item.get('line')}: {_format_number(item.get('from_capacity_mva'))} "
                    f"→ {_format_number(item.get('to_capacity_mva'))} MVA"
                )
            elif kind == "upgrade_transformer":
                lines.append(
                    f"- {item.get('transformer')}: {_format_number(item.get('from_sn_mva'))} "
                    f"→ {_format_number(item.get('to_sn_mva'))} MVA"
                )
            elif kind == "new_line":
                lines.append(
                    f"- New line Bus{item.get('from_bus')}-Bus{item.get('to_bus')}: "
                    f"{_format_number(item.get('to_capacity_mva'))} MVA"
                )
            elif kind == "set_bus_reactive_support":
                lines.append(
                    f"- {item.get('bus')} reactive support: "
                    f"{_format_number(item.get('from_reactive_support_mvar'))} → "
                    f"{_format_number(item.get('to_reactive_support_mvar'))} Mvar"
                )

    lines += ["", f"Episode: `{result.get('episode_file')}`"]
    return "\n".join(lines)


def _normalize_candidate_id(value):
    if not isinstance(value, str):
        return ""

    value = value.strip()

    for _ in range(3):
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            if value[0] == '"':
                try:
                    decoded = json.loads(value)
                except (ValueError, TypeError):
                    return ""

                if not isinstance(decoded, str):
                    return ""

                value = decoded.strip()
            else:
                value = value[1:-1].strip()
        else:
            break

    return value.upper()


def _prepare_pending_commit(out, profile, eligible_ids):
    calls = out.get("tool_calls") or []

    if not isinstance(calls, list) or len(calls) > 1:
        return [], (
            "Submit exactly one commit_candidate call. " "Multiple calls are ambiguous."
        )

    args = {}
    original = {}

    if calls:
        original = calls[0]

        if not isinstance(original, dict):
            return [], "Malformed tool call: expected an object."

        fn = original.get("function")
        if fn is None:
            fn = {}

        if not isinstance(fn, dict):
            return [], "Malformed function: expected an object."

        raw_name = fn.get("name")

        if raw_name is not None and not isinstance(raw_name, str):
            return [], "Malformed tool name: expected a string."

        name = (raw_name or "").strip()

        if name not in ("", "commit_candidate"):
            return [], (
                "Selection is pending: call commit_candidate, " f"not {name!r}."
            )

        args = fn.get("arguments", {})

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                return [], "Malformed arguments: provide a JSON object."

        if not isinstance(args, dict):
            return [], "Malformed arguments: provide a JSON object."

    if set(args) - {"profile", "candidate_id"}:
        return [], ("Commit arguments may contain only profile and candidate_id.")

    requested_profile = args.get("profile", profile)

    if not isinstance(requested_profile, str) or not requested_profile.strip():
        return [], "Provide the profile for the pending candidate batch."

    if profile is not None and requested_profile != profile:
        return [], "The supplied profile does not match the current scenario."

    if "candidate_id" in args:
        choice = _normalize_candidate_id(args["candidate_id"])
    else:
        choice = _extract_explicit_candidate_choice(out, eligible_ids)

    if choice not in eligible_ids:
        return [], (
            "No unambiguous eligible candidate ID was supplied. "
            f"Use exactly one of {eligible_ids!r}."
        )

    repaired = dict(original)
    repaired["function"] = {
        "name": "commit_candidate",
        "arguments": {
            "profile": requested_profile,
            "candidate_id": choice,
        },
    }

    changed = not calls or repaired != original
    note = f"repaired explicit commit request for Candidate {choice}" if changed else ""

    return [repaired], note


def _decode_tool_arguments(call: dict) -> tuple[dict | None, str | None]:
    if not isinstance(call, dict):
        return None, "tool call is not an object"
    fn = call.get("function")
    if not isinstance(fn, dict):
        return None, "tool call has no function object"

    raw = fn.get("arguments", {})
    if isinstance(raw, dict):
        return raw, None
    if not isinstance(raw, str):
        return None, f"tool arguments have unsupported type {type(raw).__name__}"

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"tool arguments are not valid JSON: {exc}"

    if not isinstance(value, dict):
        return None, "tool arguments JSON is not an object"
    return value, None


def run(
    task: str,
    model: str,
    stream: bool,
    profile: str | None = None,
) -> str:
    next_sequential_candidate = "A"
    sequential_retry_feedback = None
    sequential_retry_is_format = False
    G.reset_sequential_candidates()
    if hasattr(G, "set_run_context"):
        G.set_run_context(model=model, task=task, actor="llm_agent")
        ER.set_episode_metadata()

    messages = [
        {"role": "system", "content": DIAGNOSIS_SYSTEM},
        {"role": "user", "content": task},
    ]

    scan_done = False
    analyse_done = False

    recall_done = False

    pending_candidate_selection = False
    pending_crossover = False

    last_test_plan_result = None
    crossover_retry_feedback = None

    stage_attempts = {"A": 0, "B": 0, "C": 0, "D": 0, "selection": 0}

    planning_rounds_used = 0
    termination_reason = None

    for step in range(1, MAX_LLM_TURNS + 1):
        if (
            planning_rounds_used >= MAX_PLANNING_ROUNDS
            and not pending_crossover
            and not pending_candidate_selection
        ):
            termination_reason = "planning_round_budget"
            print(
                f"{DIM}planning-round budget exhausted: "
                f"{planning_rounds_used}/{MAX_PLANNING_ROUNDS}{RESET}",
                flush=True,
            )
            break
        print(f"\n{BOLD}-- model turn {step} · thinking --{RESET}")
        t0 = time.time()

        diagnosis_stage = not analyse_done

        candidate_stage = (
            analyse_done
            and recall_done
            and not pending_candidate_selection
            and not pending_crossover
        )
        recall_stage = analyse_done and not recall_done

        available_tools = None
        commit_repair_note = ""

        crossover_repair_note = ""

        if pending_candidate_selection:
            required_tool = "commit_candidate"
        elif pending_crossover:
            required_tool = "test_crossover_plan"
        elif recall_stage:
            required_tool = "recall_experience"
        elif candidate_stage:
            required_tool = "test_staged_candidate"
        else:
            required_tool = None

        if diagnosis_stage:

            allowed = {"analyse"} if scan_done else {"scan_day", "analyse"}
            available_tools = [
                tool
                for tool in G.TOOLS
                if tool.get("function", {}).get("name") in allowed
            ]
        elif required_tool:
            tool_pool = G.TOOLS
            available_tools = [
                tool
                for tool in tool_pool
                if tool.get("function", {}).get("name") == required_tool
            ]

        chat_messages = messages
        if pending_candidate_selection:
            chat_messages = _selection_stage_messages(profile, last_test_plan_result)
        elif pending_crossover:
            chat_messages = _fusion_build_messages(
                profile, last_test_plan_result, crossover_retry_feedback
            )
        elif candidate_stage:
            chat_messages = _candidate_stage_messages(
                messages,
                profile,
                last_test_plan_result,
                next_sequential_candidate,
                sequential_retry_feedback,
            )

        if candidate_stage and available_tools:
            try:
                stage_packet = json.loads(chat_messages[1]["content"])
            except (IndexError, TypeError, ValueError):
                stage_packet = {}
            available_tools = _sequential_stage_tool(available_tools, stage_packet)

        active_key = (
            "selection"
            if pending_candidate_selection
            else (
                "D"
                if pending_crossover
                else next_sequential_candidate if candidate_stage else None
            )
        )
        limit = (
            MAX_SELECTION_ATTEMPTS
            if active_key == "selection"
            else MAX_CROSSOVER_ATTEMPTS if active_key == "D" else MAX_STAGE_ATTEMPTS
        )
        if (
            pending_crossover
            and not json.loads(chat_messages[1]["content"])["legal_d_options"]
        ):
            recovery = G.recover_pending_crossover(profile, "no_legal_fusion")
            if recovery.get("error"):
                raise RuntimeError(recovery["error"])
            last_test_plan_result = recovery
            pending_crossover = False
            pending_candidate_selection = bool(recovery.get("selection_required"))
            crossover_retry_feedback = None
            stage_attempts["D"] = 0
            continue

        if active_key and stage_attempts[active_key] >= limit:
            G.reset_pending_round()
            termination_reason = f"{active_key}_agent_stage_exhausted"
            break
        if active_key:
            stage_attempts[active_key] += 1

        format_retry = candidate_stage and sequential_retry_is_format
        thinking = not (pending_candidate_selection or format_retry)
        if pending_crossover:
            thinking = stage_attempts["D"] == 1
        instruction, budget = INITIAL_DIAGNOSIS_INSTRUCTION, DIAGNOSIS_NUM_PREDICT
        if recall_stage:
            instruction, budget = RECALL_STAGE_INSTRUCTION, RECALL_NUM_PREDICT
        elif candidate_stage:
            instruction = SEQUENTIAL_REPAIR if format_retry else None
            budget = (
                CANDIDATE_REPAIR_NUM_PREDICT if format_retry else CANDIDATE_NUM_PREDICT
            )
        elif pending_crossover:
            instruction = (
                None if thinking else "Copy one complete legal_d_options entry exactly."
            )
            budget = CROSSOVER_NUM_PREDICT if thinking else CROSSOVER_REPAIR_NUM_PREDICT
        elif pending_candidate_selection:
            instruction, budget = SELECTION_STAGE_INSTRUCTION, SELECTION_NUM_PREDICT
        out = chat(
            chat_messages,
            model,
            stream=stream,
            available_tools=available_tools,
            stage_instruction=instruction,
            num_predict=budget,
            enable_thinking=thinking,
            temperature=0.1 if format_retry else 0.3,
        )
        generation = out.get("generation") or {}
        print(
            f"{DIM}generation: {time.time() - t0:.1f}s | "
            f"stop={generation.get('done_reason', 'unknown')} | "
            f"output_tokens={generation.get('eval_count', 'unknown')} | "
            f"prompt_tokens={generation.get('prompt_eval_count', 'unknown')} | "
            f"tool_calls={len(out['tool_calls'])} | "
            f"thinking={'on' if thinking else 'off'}{RESET}",
            flush=True,
        )

        candidate_call_provenance = "native_tool_call"
        if candidate_stage:
            if _normalise_staged_action_array(out.get("tool_calls") or []):
                print(
                    f"{DIM}controller: decoded complete stringified "
                    f"Candidate {next_sequential_candidate} actions array{RESET}",
                    flush=True,
                )
            problem = _validate_sequential_native_call(
                out.get("tool_calls") or [],
                profile,
                next_sequential_candidate,
                available_tools[0] if available_tools else None,
            )
            if problem:
                recovered = _recover_explicit_sequential_plan(
                    out, profile, next_sequential_candidate
                )
                if recovered:
                    recovered_problem = _validate_sequential_native_call(
                        recovered,
                        profile,
                        next_sequential_candidate,
                        available_tools[0] if available_tools else None,
                    )
                    if recovered_problem:
                        recovered = None
                        problem = recovered_problem
                if recovered:
                    out["tool_calls"] = recovered
                    candidate_call_provenance = "model_authored_complete_json_recovery"
                    sequential_retry_feedback = None
                    sequential_retry_is_format = False
                    print(
                        f"{DIM}controller: recovery_provenance=json_plan "
                        f"candidate={next_sequential_candidate} "
                        f"generation_finish={generation.get('done_reason', 'unknown')}{RESET}",
                        flush=True,
                    )
                    print(
                        f"{DIM}controller: recovered a complete explicit Candidate "
                        f"{next_sequential_candidate} JSON plan; unchanged actions "
                        "sent through normal verifier." + RESET,
                        flush=True,
                    )
                else:
                    out["tool_calls"] = []
                    sequential_retry_feedback = problem
                    sequential_retry_is_format = True
                    print(
                        f"{DIM}controller: rejected Candidate "
                        f"{next_sequential_candidate} model output: {problem}{RESET}",
                        flush=True,
                    )
            else:
                sequential_retry_feedback = None
                sequential_retry_is_format = False

        if pending_crossover:
            out["tool_calls"], crossover_repair_note = _prepare_pending_crossover(
                out,
                profile,
                last_test_plan_result,
            )
            if out["tool_calls"]:
                crossover_retry_feedback = None
            else:
                crossover_retry_feedback = (
                    crossover_repair_note
                    or "The previous crossover response did not contain a usable tool call."
                )
            if crossover_repair_note:
                print(
                    f"{DIM}controller: {crossover_repair_note}{RESET}",
                    flush=True,
                )

        if pending_candidate_selection:
            eligible = (last_test_plan_result or {}).get("eligible_candidate_ids", [])

            out["tool_calls"], commit_repair_note = _prepare_pending_commit(
                out,
                profile,
                eligible,
            )

            if commit_repair_note:
                print(
                    f"{DIM}controller: {commit_repair_note}{RESET}",
                    flush=True,
                )

        if not out["tool_calls"]:

            if last_test_plan_result and last_test_plan_result.get("solved"):
                return format_verified_findings(last_test_plan_result)

            if pending_crossover:

                continue

            if candidate_stage:
                sequential_retry_feedback = sequential_retry_feedback or (
                    "Emit one native test_staged_candidate call, not prose."
                )
                sequential_retry_is_format = True
                continue

            messages.append(
                {
                    "role": "assistant",
                    "content": out["content"],
                    "thinking": out["thinking"],
                }
            )
            follow_up = (
                (
                    "A candidate selection is pending. "
                    "Repair the tool-call format only. "
                    "If you already chose an eligible candidate, "
                    "keep that choice; do not repeat the comparison. "
                    "Call commit_candidate with profile and candidate_id "
                    "as plain strings. " + commit_repair_note
                )
                if pending_candidate_selection
                else (
                    "The last committed simulator result did not report "
                    "solved=true. Do not claim the system is solved. "
                    "Continue using the tools."
                )
            )
            messages.append({"role": "user", "content": follow_up})
            continue

        print(f"{DIM}({time.time() - t0:.0f}s){RESET}")
        messages.append(
            {
                "role": "assistant",
                "content": out["content"],
                "thinking": out["thinking"],
                "tool_calls": out["tool_calls"],
            }
        )

        solved_result = None

        for call in out["tool_calls"]:
            fn = call.get("function", {})
            name = fn.get("name", "")

            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError as exc:
                    result = {
                        "error": (
                            "Malformed native tool-call JSON arguments; "
                            f"tool was not executed: {exc}"
                        )
                    }
                    ER.record_tool(name, {"raw_arguments": args}, result)
                    print(
                        f"  {BOLD}<-{RESET} ERROR: {result['error']}",
                        flush=True,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "name": name,
                            "content": json.dumps(result),
                        }
                    )
                    continue

            print(
                f"  {BOLD}->{RESET} {name}  {DIM}{args.get('profile')}{RESET}",
                flush=True,
            )

            if name == "test_staged_candidate":
                print(
                    f"     evaluating Candidate {args.get('candidate_id')} independently from parent",
                    flush=True,
                )
                for action in args.get("actions") or []:
                    print(f"       - {G.describe_action(action)}", flush=True)
            t1 = time.time()
            if name not in {t["function"]["name"] for t in available_tools}:
                result = {
                    "error": "Only tools offered for the current stage are permitted."
                }
            else:
                result = G.call_tool(name, args)
            if (
                isinstance(result, dict)
                and not result.get("error")
                and name == "test_staged_candidate"
                and args.get("candidate_id") == "C"
            ):
                planning_rounds_used += 1
                print(
                    f"     {DIM}planning round {planning_rounds_used}/"
                    f"{MAX_PLANNING_ROUNDS}{RESET}",
                    flush=True,
                )
            if name == "test_staged_candidate" and isinstance(result, dict):

                result["generation_provenance"] = candidate_call_provenance
            ER.record_tool(name, args, result)
            if (
                name == "test_crossover_plan"
                and isinstance(result, dict)
                and not result.get("error")
            ):
                print(
                    f"[mandatory-d-batch-verified] profile={profile} " "source=qwen",
                    flush=True,
                )
            if (
                name == "test_staged_candidate"
                and isinstance(result, dict)
                and result.get("error")
            ):

                if result.get("error_code") == "stale_staged_parent":
                    next_sequential_candidate = "A"
                sequential_retry_feedback = str(result["error"])[:500]

                sequential_retry_is_format = result.get("error_code") not in {
                    "empty_effective_candidate",
                    "invalid_or_no_effect_candidate_actions",
                    "candidate_not_distinct",
                    "insufficient_candidate_diversity",
                }

            if "error" not in result:
                if name == "test_staged_candidate":
                    stage_attempts[str(args.get("candidate_id", "")).upper()] = 0
                elif name == "test_crossover_plan":
                    stage_attempts["D"] = 0
                elif name == "commit_candidate":
                    stage_attempts["selection"] = 0
                if name == "scan_day":
                    scan_done = True

                elif name == "analyse":
                    analyse_done = True

                elif name == "recall_experience":
                    recall_done = True

                elif name == "test_staged_candidate":
                    if args.get("candidate_id") in {"A", "B"}:
                        next_sequential_candidate = (
                            "B" if args["candidate_id"] == "A" else "C"
                        )
                    else:
                        next_sequential_candidate = "A"

                        pending_crossover = bool(result.get("crossover_required"))
                        pending_candidate_selection = bool(
                            result.get("selection_required")
                        )
                    sequential_retry_feedback = None
                    sequential_retry_is_format = False

                elif name == "test_crossover_plan":

                    pending_crossover = bool(result.get("crossover_required"))
                    pending_candidate_selection = bool(result.get("selection_required"))
                    crossover_retry_feedback = None

                elif name == "commit_candidate":

                    pending_crossover = False
                    pending_candidate_selection = False

            print(
                f"  {BOLD}<-{RESET} {brief(name, result)} "
                f"{DIM}({time.time() - t1:.1f}s){RESET}"
            )

            if (
                name == "test_crossover_plan"
                and isinstance(result, dict)
                and result.get("error")
            ):
                crossover_retry_feedback = str(result.get("error"))

            model_result = result
            if name == "recall_experience" and isinstance(result, dict):

                model_result = _compact_recall_for_model(result)

            messages.append(
                {
                    "role": "tool",
                    "name": name,
                    "content": json.dumps(model_result),
                }
            )

            if (
                name
                not in (
                    "test_staged_candidate",
                    "commit_candidate",
                    "test_crossover_plan",
                )
                or "error" in result
            ):
                continue

            if name == "test_staged_candidate" and args.get("candidate_id") in {
                "A",
                "B",
            }:

                continue
            last_test_plan_result = result

            if (
                result.get("episode_id")
                and result.get("episode_test_plan_call") is not None
            ):
                print(
                    f"     {DIM}episode {result['episode_id']} simulator round "
                    f"{result.get('episode_test_plan_call')} recorded{RESET}"
                )
            if result.get("search_decision") == "reverted":
                print(
                    f"     {DIM}best candidate not committed; retained plan unchanged{RESET}"
                )

            if name in (
                "test_staged_candidate",
                "test_crossover_plan",
            ):
                for candidate in result.get("candidate_results", []):
                    marker = "IMPROVING" if candidate.get("improving") else ""
                    if candidate.get("selected_as_best"):
                        marker = (marker + "+SELECTED").strip("+")
                    if candidate.get("committed"):
                        marker = (marker + "+COMMITTED").strip("+")
                    sec = (candidate.get("security5") or {}).get("vector", [])
                    sec_text = (
                        "[" + ", ".join(f"{float(x):.4f}" for x in sec) + "]"
                        if len(sec) == 5
                        else str(sec)
                    )
                    print(
                        f"       {DIM}{candidate.get('candidate_id')}: "
                        f"Security5={sec_text}, "
                        f"cost=${candidate.get('plan_cost_musd')}M, "
                        f"relation={candidate.get('security5_relation_vs_parent')} "
                        f"{marker}{RESET}"
                    )
                if result.get("selection_required"):
                    print(
                        f"       {BOLD}Qwen must now choose and commit one of "
                        f"{result.get('eligible_candidate_ids', [])}{RESET}"
                    )

            if name == "commit_candidate":
                print(
                    f"       {BOLD}Qwen committed Candidate "
                    f"{result.get('selected_candidate_id')}{RESET}"
                )

            for note in result.get("normalization_notes", [])[:3]:
                print(f"       {DIM}{note}{RESET}")
            for note in result.get("no_effect_or_invalid_action_notes", [])[:3]:
                print(f"       {DIM}{note}{RESET}")

            for item in result.get("base_case_overloaded_elements", [])[:3]:
                print(
                    f"       {DIM}base-case overload: {item['element']} "
                    f"{item['loading_pct']:.1f}% at h{item['hour']} "
                    f"(rating {item['rating_mva']} MVA){RESET}"
                )

            for item in result.get("base_case_voltage_problem_buses", [])[:3]:
                print(
                    f"       {DIM}base-case voltage: {item['bus']} "
                    f"{item['min_vm_pu']}-{item['max_vm_pu']} pu "
                    f"({item['problem']}){RESET}"
                )

            for item in result.get("n1_overloaded_elements", [])[:3]:
                print(
                    f"       {DIM}N-1 overload: {item['element']} overloaded in "
                    f"{item['overloaded_in_n_outage_tests']} outage test(s); "
                    f"worst {item['worst_loading_pct']:.1f}% "
                    f"(rating {item['capacity_mva']} MVA){RESET}"
                )

            for item in result.get("n1_voltage_problem_buses", [])[:2]:
                example = (
                    item["outages"][0] if item.get("outages") else "unknown outage"
                )
                print(
                    f"       {DIM}N-1 voltage: {item['bus']} had unacceptable voltage in "
                    f"{item['voltage_problem_in_n_outage_tests']} outage test(s); "
                    f"worst voltage {item['worst_vm_pu']} pu; "
                    f"example outage: {example}{RESET}"
                )

            if result.get("solved"):
                solved_result = result

        if solved_result is not None:
            return format_verified_findings(solved_result)
    else:
        termination_reason = "llm_turn_safety_cap"

    if profile and hasattr(G, "finish_episode"):
        G.finish_episode(profile, status="stopped")

    if last_test_plan_result:
        sec = (last_test_plan_result.get("security5") or {}).get("vector", [])
        sec_text = (
            ("[" + ", ".join(f"{float(x):.6f}" for x in sec) + "]")
            if len(sec) == 5
            else str(sec)
        )
        return (
            "**NOT SOLVED — "
            + (
                f"maximum physical planning rounds reached "
                f"({planning_rounds_used}/{MAX_PLANNING_ROUNDS})"
                if termination_reason == "planning_round_budget"
                else (
                    f"LLM-turn safety cap reached ({MAX_LLM_TURNS}) before "
                    f"the {MAX_PLANNING_ROUNDS}-round planning budget was exhausted"
                    if termination_reason == "llm_turn_safety_cap"
                    else "controller stopped before a solved terminal state"
                )
            )
            + ".**\n\n"
        ) + f"Latest committed Security5: {sec_text}."
    return "**NOT SOLVED — no successful planning verification result was produced.**"


def _configure_memory(args):
    out = (
        Path(args.output)
        if args.output
        else RUNS_DIR / f"{args.profile}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    (out / "memory").mkdir(parents=True, exist_ok=True)
    G.config.MEMORY = (out / "memory").resolve()
    print(f"run folder: {out}")
    folds = json.loads((MEMORY_DIR / "folds.json").read_text())
    fold = next((f for f, v in folds.items() if args.profile in v["eval"]), None)
    if fold is None and not (args.episode_memory and args.strategies):
        raise SystemExit(
            f"{args.profile} is in no evaluation group of memory/folds.json; "
            "give --episode-memory and --strategies, provide both memory paths"
        )
    args.episode_memory = args.episode_memory or str(MEMORY_DIR / fold / "episodic")
    ER.set_strategy_bank(args.strategies or str(MEMORY_DIR / fold / "strategies.json"))
    print(f"memory: {fold or 'custom'} (episodic: {args.episode_memory})")


def main():
    parser = argparse.ArgumentParser()
    available = sorted(G.load_grid()[1])
    default_scenario = "normal_001" if "normal_001" in available else available[0]
    parser.add_argument(
        "--scenario",
        "--profile",
        dest="profile",
        default=default_scenario,
        help=("exact scenario name, " "e.g. summer_001"),
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--task", default=None)
    parser.add_argument(
        "--stream", action="store_true", help="stream model output to the terminal"
    )
    parser.add_argument(
        "--episode-memory",
        default=None,
        help="episodic memory folder (default: memory/<fold>/episodic for the scenario's fold)",
    )
    parser.add_argument(
        "--strategies",
        default=None,
        help="procedural strategy bank JSON (default: memory/<fold>/strategies.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="folder for this run's trajectory and episode (default: runs/<scenario>_<time>)",
    )
    args = parser.parse_args()
    _configure_memory(args)

    task = args.task or (
        f"Plan reinforcements for the {args.profile} scenario. "
        "Base-case security is evaluated across all 24 hours, while N-1 security "
        "is evaluated at the selected maximum-total-load hour. "
        "Continue until a committed planning result reports solved=true (Security5=[0,0,0,0,0])."
    )
    with ER.PlanningSession(
        G,
        args.profile,
        args.model,
        episode_memory=args.episode_memory,
    ):
        print(
            f"{BOLD}model:{RESET} {args.model}\n"
            f"{BOLD}backend:{RESET} vLLM @ {VLLM_BASE_URL}\n"
            f"{BOLD}task: {RESET} {task}"
        )

        answer = run(
            task,
            args.model,
            stream=args.stream,
            profile=args.profile,
        )

        print(f"\n{BOLD}{'=' * 70}\nFINDINGS\n{'=' * 70}{RESET}")
        print(answer, "\n")


def _prepare_pending_crossover(out, profile, pending_result):
    calls = out.get("tool_calls") or []
    if (
        len(calls) != 1
        or calls[0].get("function", {}).get("name") != "test_crossover_plan"
    ):
        return [], "Submit exactly one test_crossover_plan call."
    args, error = _decode_tool_arguments(calls[0])
    if error or args not in _fd_legal_options({**pending_result, "profile": profile}):
        return [], "Copy one complete legal_d_options entry exactly."
    problem = _fusion_validate_choice(args, pending_result)
    if problem:
        return [], "v3.0D gene constraint error: " + problem

    return calls, ""


if __name__ == "__main__":
    main()
