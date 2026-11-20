from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import episode_store as E

from . import actions as A
from . import config as C
from . import measurement as M
from . import network as N
from . import state as S


def _episode_read_memory():
    return Path(os.environ.get("EPISODE_MEMORY_READ", str(C.MEMORY))).resolve()


def set_run_context(model: str = "", task: str = "", actor: str = "agent"):
    S.RUN_CONTEXT["model"] = str(model or "")
    S.RUN_CONTEXT["task"] = str(task or "")
    S.RUN_CONTEXT["actor"] = str(actor or "agent")


def _profile_vector(
    profile: str,
    initial_state: dict | None = None,
) -> list[float]:
    """Return initial Security5, never the current reinforced state."""
    if initial_state is None:
        key = ("analyse", profile)
        initial_state = S.DAY_CACHE.get(key)

        if initial_state is None:
            initial_state = M._measure_day(profile, [])
            S.DAY_CACHE[key] = initial_state

    return E.security5_scenario_vector(initial_state)


def _find_similar_security5_runtime(
    memory_dir,
    query_vector: list[float],
    *,
    top_k: int,
    solved_only: bool = False,
    completed_only: bool = True,
    exclude_episode_id: str | None = None,
    exclude_profile: str | None = None,
) -> list[dict]:
    q = [float(x) for x in query_vector]
    if len(q) != 5:
        raise ValueError("Security5 retrieval requires five query components.")

    ranked: list[dict] = []
    index = E.load_index(memory_dir)

    for item in index.get("episodes", []):
        episode_id = item.get("episode_id")
        if exclude_episode_id and episode_id == exclude_episode_id:
            continue
        if exclude_profile and str(item.get("profile") or "") == str(exclude_profile):
            continue
        if completed_only and item.get("status") == "in_progress":
            continue
        if solved_only and not item.get("solved"):
            continue

        try:
            prior = E.load_episode(memory_dir, episode_id)
            prior_vector = E.security5_scenario_vector(prior.get("initial_state") or {})
        except (FileNotFoundError, KeyError, TypeError, ValueError):

            continue

        distance = math.sqrt(
            sum((float(a) - float(b)) ** 2 for a, b in zip(q, prior_vector))
        )

        ranked.append(
            {
                "episode_id": episode_id,
                "profile": item.get("profile", prior.get("profile")),
                "status": item.get("status", prior.get("status")),
                "solved": item.get("solved"),
                "test_plan_calls": item.get("test_plan_calls"),
                "n_kept_plan_updates": item.get("n_kept_plan_updates"),
                "total_cost_musd": item.get("total_cost_musd"),
                "l2_distance": round(float(distance), 8),
                "file": item.get("file"),
            }
        )

    ranked.sort(key=lambda x: x["l2_distance"])
    return ranked[: max(0, int(top_k))]


def _episode_state(state: dict) -> dict:
    return {
        "n_base_case_problem_hours": int(state.get("n_base_case_problem_hours", 0)),
        "nonconverged_base_case_hours": list(
            state.get("nonconverged_base_case_hours", []) or []
        ),
        "n_failed_n1_outage_tests": int(state.get("n_failed_n1_outage_tests", 0) or 0),
        "n_n1_outage_tests": int(state.get("n_n1_outage_tests", 0) or 0),
        "security5": copy.deepcopy(state.get("security5") or {}),
        "base_case_overloaded_elements": list(
            state.get("base_case_overloaded_elements", []) or []
        ),
        "base_case_voltage_problem_buses": list(
            state.get("base_case_voltage_problem_buses", []) or []
        ),
        "n1_overloaded_elements": list(state.get("n1_overloaded_elements", []) or []),
        "n1_voltage_problem_buses": list(
            state.get("n1_voltage_problem_buses", []) or []
        ),
        "base_case_worst_hours": list(state.get("base_case_worst_hours", []) or []),
    }


def _ids(items: list[dict], field: str) -> set[str]:
    out = set()
    for item in items or []:
        value = item.get(field)
        if value is not None:
            out.add(str(value))
    return out


def _episode_consequence(before: dict, after: dict) -> dict:
    b_over = _ids(before.get("base_case_overloaded_elements", []), "element")
    a_over = _ids(after.get("base_case_overloaded_elements", []), "element")
    b_voltage = _ids(before.get("base_case_voltage_problem_buses", []), "bus")
    a_voltage = _ids(after.get("base_case_voltage_problem_buses", []), "bus")
    b_n1_over = _ids(before.get("n1_overloaded_elements", []), "element")
    a_n1_over = _ids(after.get("n1_overloaded_elements", []), "element")
    b_n1_voltage = _ids(before.get("n1_voltage_problem_buses", []), "bus")
    a_n1_voltage = _ids(after.get("n1_voltage_problem_buses", []), "bus")

    return {
        "security5_before": copy.deepcopy(before.get("security5") or {}),
        "security5_after": copy.deepcopy(after.get("security5") or {}),
        "security5_change": {
            name: round(
                float((after.get("security5") or {}).get("features", {}).get(name, 0.0))
                - float(
                    (before.get("security5") or {}).get("features", {}).get(name, 0.0)
                ),
                4,
            )
            for name in (
                "base_thermal_severity",
                "base_voltage_severity",
                "n1_thermal_severity",
                "n1_voltage_severity",
                "n1_failure_fraction",
            )
        },
        "resolved_base_case_overloaded_elements": sorted(b_over - a_over),
        "new_base_case_overloaded_elements": sorted(a_over - b_over),
        "resolved_base_case_voltage_problem_buses": sorted(b_voltage - a_voltage),
        "new_base_case_voltage_problem_buses": sorted(a_voltage - b_voltage),
        "resolved_n1_overloaded_elements": sorted(b_n1_over - a_n1_over),
        "new_n1_overloaded_elements": sorted(a_n1_over - b_n1_over),
        "resolved_n1_voltage_problem_buses": sorted(b_n1_voltage - a_n1_voltage),
        "new_n1_voltage_problem_buses": sorted(a_n1_voltage - b_n1_voltage),
    }


def _plan_cost(profile: str, actions: list[dict]) -> float:
    if not actions:
        return 0.0
    base_net, _ = N._setup(profile, N.peak_hour_of(profile))
    trial = copy.deepcopy(base_net)
    cost, _, _ = A._apply(trial, actions)
    return float(cost)


def _record_final_plan(profile: str, actions: list[dict]) -> list[dict]:
    """Convert the final plan to original-baseline -> final-target transitions."""
    base_net, _ = N._setup(profile, N.peak_hour_of(profile))
    records, _ = A.record_action_transitions(copy.deepcopy(base_net), actions)
    return records


def _start_episode_if_needed(
    profile: str,
    initial_state: dict,
) -> str:
    if S.ACTIVE_EPISODE_ID is not None and S.ACTIVE_EPISODE_PROFILE == profile:
        return S.ACTIVE_EPISODE_ID

    scenario_vector = _profile_vector(profile, initial_state)

    S.ACTIVE_EPISODE_ID = E.start_episode(
        C.MEMORY,
        profile,
        scenario_vector,
        _episode_state(initial_state),
        network=C.TOPOLOGY,
        metadata=dict(S.RUN_CONTEXT),
        scenario_representation="security5",
    )

    S.ACTIVE_EPISODE_PROFILE = profile
    return S.ACTIVE_EPISODE_ID


def finish_episode(profile: str | None = None, status: str = "stopped"):
    if S.ACTIVE_EPISODE_ID is None:
        return None
    if profile is not None and S.ACTIVE_EPISODE_PROFILE != profile:
        return None

    current_profile = S.ACTIVE_EPISODE_PROFILE
    actions = list(S.PLAN.get("actions", []))

    try:
        state = M._measure_day(current_profile, actions)
        cost = _plan_cost(current_profile, actions)
        vector = [float(x) for x in (state.get("security5") or {}).get("vector", [])]
        solved = (
            len(vector) == 5
            and not state.get("nonconverged_base_case_hours")
            and int(state.get("n_base_case_problem_hours", 0) or 0) == 0
            and int(state.get("n_failed_n1_outage_tests", 0) or 0) == 0
            and all(abs(x) <= 1e-9 for x in vector)
        )
        E.finish_episode(
            C.MEMORY,
            S.ACTIVE_EPISODE_ID,
            solved=solved,
            final_plan=_record_final_plan(current_profile, actions),
            total_cost_musd=cost,
            final_state=_episode_state(state),
            n_kept_plan_updates=int(S.PLAN.get("kept_updates", 0)),
            status="solved" if solved else status,
        )
    finally:
        old = S.ACTIVE_EPISODE_ID
        S.ACTIVE_EPISODE_ID = None
        S.ACTIVE_EPISODE_PROFILE = None
        from .tools.test_plan import reset_pending_round

        reset_pending_round()

    return old


def recall_experience(
    profile: str,
    rank: int = 1,
) -> dict:
    rank = int(rank)

    if rank < 1:
        raise ValueError("rank must be 1 or greater")

    initial_state = S.DAY_CACHE.get(("analyse", profile))

    if initial_state is None:
        initial_state = M._measure_day(profile, [])

    current_episode_id = _start_episode_if_needed(
        profile,
        initial_state,
    )

    similarity_method = "security5"
    matches = _find_similar_security5_runtime(
        _episode_read_memory(),
        E.security5_scenario_vector(initial_state),
        top_k=rank,
        solved_only=False,
        completed_only=True,
        exclude_episode_id=current_episode_id,
        exclude_profile=profile,
    )

    if len(matches) < rank:
        E.append_memory_event(
            C.MEMORY,
            current_episode_id,
            {
                "type": "recall_experience",
                "query_profile": profile,
                "rank": rank,
                "found": False,
            },
        )

        return {
            "found": False,
            "rank": rank,
            "message": (
                "No completed prior experience is available "
                f"at similarity rank {rank}."
            ),
        }

    match = matches[rank - 1]

    prior = E.load_episode(
        _episode_read_memory(),
        match["episode_id"],
    )

    final = prior.get("final") or {}
    initial = prior.get("initial_state") or {}
    final_state = final.get("final_state") or {}

    trajectory = []

    for step in prior.get("steps", []):
        consequence = step.get("simulator_consequence") or {}
        trajectory.append(
            {
                "step": step.get("step"),
                "search_decision": step.get(
                    "search_decision",
                    step.get("verdict"),
                ),
                "outcome_label": step.get(
                    "outcome_label",
                    step.get("tool_verdict"),
                ),
                "security5_change": consequence.get("security5_change"),
            }
        )

    result = {
        "found": True,
        "retrieval": {
            "metric": similarity_method,
            "rank": rank,
            "episode_id": prior.get("episode_id"),
            "profile": prior.get("profile"),
            "l2_distance": match["l2_distance"],
        },
        "historical_outcome": {
            "status": prior.get("status"),
            "solved": bool(final.get("solved")),
            "final_cost_musd": final.get("total_cost_musd"),
        },
        "historical_initial_state": {
            "security5": copy.deepcopy(initial.get("security5") or {}),
            "base_case_overloaded_elements": initial.get(
                "base_case_overloaded_elements", []
            ),
            "base_case_voltage_problem_buses": initial.get(
                "base_case_voltage_problem_buses", []
            ),
            "n1_overloaded_elements": initial.get("n1_overloaded_elements", []),
            "n1_voltage_problem_buses": initial.get("n1_voltage_problem_buses", []),
        },
        "historical_final_state": {
            "security5": copy.deepcopy(final_state.get("security5") or {}),
            "base_case_overloaded_elements": final_state.get(
                "base_case_overloaded_elements", []
            ),
            "base_case_voltage_problem_buses": final_state.get(
                "base_case_voltage_problem_buses", []
            ),
            "n1_overloaded_elements": final_state.get("n1_overloaded_elements", []),
            "n1_voltage_problem_buses": final_state.get("n1_voltage_problem_buses", []),
        },
        "historical_final_plan": final.get("final_plan") or [],
        "historical_trajectory": trajectory,
        "note": (
            "This is historical evidence from another episode. "
            "from_* values describe that historical state and to_* "
            "values describe targets tested there. Any reused or revised "
            "target must be verified on the current scenario with test_staged_candidate."
        ),
    }

    E.append_memory_event(
        C.MEMORY,
        current_episode_id,
        {
            "type": "recall_experience",
            "current_profile": profile,
            "rank": rank,
            "found": True,
            "retrieved_episode_id": prior.get("episode_id"),
            "retrieved_profile": prior.get("profile"),
            "l2_distance": match["l2_distance"],
            "retrieved_solved": bool(final.get("solved")),
        },
    )

    return result
