from __future__ import annotations

import copy
import json
import math
import uuid

import episode_store as E

_D_SOURCES = ("A", "B", "C")


def _d_signature(catalog, refs):
    return tuple(
        sorted(
            json.dumps(
                [catalog[r]["equipment_key"], catalog[r]["action"]],
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            for r in refs
        )
    )


def _d_strict_superior(d, abc, score_improves):
    siblings = [r for r in abc if r.get("candidate_id") in _D_SOURCES]
    if len(siblings) != 3:
        return False
    secure = [r for r in siblings if r.get("solved")]
    if secure:
        if not d.get("solved"):
            return False
        if float(d["cost_musd"]) >= min(float(r["cost_musd"]) for r in secure) - 1e-9:
            return False
    return all(score_improves(d["score"], r["score"]) for r in siblings)


from .. import actions as A
from .. import config as C
from .. import episodes as EP
from .. import measurement as M
from .. import network as N
from .. import state as S

_PENDING_CANDIDATE_SELECTION: dict | None = None
_PENDING_SEQUENTIAL_CANDIDATES: dict | None = None


def _unique(items: list[str]) -> list[str]:
    """Preserve order while removing duplicate messages."""
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def _security5_vector(state: dict) -> tuple[float, float, float, float, float]:
    security5 = state.get("security5") or {}
    source = security5.get("vector", [])
    vector = tuple(float(x) for x in source)
    if len(vector) != 5:
        raise ValueError(
            "Planner comparison requires state['security5']['vector'] with exactly "
            "five values [T_B, V_B, T_N1, V_N1, F_N1]."
        )
    if not all(math.isfinite(x) for x in vector):
        raise ValueError("Security5 contains a non-finite value.")
    return vector


def _security5_solved(state: dict, atol: float = 1e-9) -> bool:
    if state.get("nonconverged_base_case_hours"):
        return False
    if int(state.get("n_base_case_problem_hours", 0) or 0) != 0:
        return False
    if int(state.get("n_failed_n1_outage_tests", 0) or 0) != 0:
        return False
    return all(abs(float(x)) <= atol for x in _security5_vector(state))


_SECURITY5_ATOL = 1e-9
_SECURITY5_NAMES = (
    "base_thermal_severity",
    "base_voltage_severity",
    "n1_thermal_severity",
    "n1_voltage_severity",
    "n1_failure_fraction",
)


def _selection_score(
    state: dict,
    cost_musd: float,
) -> tuple[tuple[float, float, float, float, float], float]:
    return (_security5_vector(state), float(cost_musd))


def _score_record(
    score: tuple[tuple[float, float, float, float, float], float],
) -> dict:
    vector, cost = score
    return {
        "security5": {
            "vector": [float(x) for x in vector],
            "features": {
                name: float(vector[i]) for i, name in enumerate(_SECURITY5_NAMES)
            },
        },
        "plan_cost_musd": round(float(cost), 4),
    }


def _security5_equal(
    left: tuple[float, ...],
    right: tuple[float, ...],
    atol: float = _SECURITY5_ATOL,
) -> bool:
    return all(abs(float(a) - float(b)) <= atol for a, b in zip(left, right))


def _security5_dominates(
    left: tuple[float, ...],
    right: tuple[float, ...],
    atol: float = _SECURITY5_ATOL,
) -> bool:
    """True when left is no worse in all five components and better in at least one."""
    no_worse = all(float(a) <= float(b) + atol for a, b in zip(left, right))
    strictly_better = any(float(a) < float(b) - atol for a, b in zip(left, right))
    return bool(no_worse and strictly_better)


def _score_improves(
    candidate_score: tuple[tuple[float, ...], float],
    parent_score: tuple[tuple[float, ...], float],
) -> bool:
    c_sec, c_cost = candidate_score
    p_sec, p_cost = parent_score
    if _security5_dominates(c_sec, p_sec):
        return True
    if _security5_equal(c_sec, p_sec) and c_cost < p_cost - _SECURITY5_ATOL:
        return True
    return False


def _feasibility_aware_improvement(item, parent_state, parent_score):
    if bool(item.get("solved", False)) and not _security5_solved(parent_state):
        return True

    return _score_improves(item["score"], parent_score)


def _score_relation(
    candidate_score: tuple[tuple[float, ...], float],
    reference_score: tuple[tuple[float, ...], float],
) -> str:
    c_sec, c_cost = candidate_score
    r_sec, r_cost = reference_score
    if _security5_dominates(c_sec, r_sec):
        return "security5_dominates"
    if _security5_dominates(r_sec, c_sec):
        return "security5_dominated"
    if _security5_equal(c_sec, r_sec):
        if c_cost < r_cost - _SECURITY5_ATOL:
            return "security5_equal_lower_cost"
        if c_cost > r_cost + _SECURITY5_ATOL:
            return "security5_equal_higher_cost"
        return "security5_equal"
    return "security5_tradeoff"


def _security5_change(candidate_state: dict, parent_state: dict) -> dict:
    c = _security5_vector(candidate_state)
    p = _security5_vector(parent_state)
    return {
        name: round(float(c[i] - p[i]), 4) for i, name in enumerate(_SECURITY5_NAMES)
    }


def _evaluate_candidate_from_parent(
    profile: str,
    previous: list[dict],
    state_before: dict,
    base_net,
    actions: list[dict],
) -> dict:
    new_actions, batch_normalization_notes = A._dedupe(actions)

    record_net = copy.deepcopy(base_net)
    if previous:
        A._apply(record_net, previous)

    recorded_actions, action_notes = A.record_action_transitions(
        record_net,
        new_actions,
    )

    state_changing_new_actions = [
        action
        for action, record in zip(new_actions, recorded_actions)
        if record.get("simulator_applied")
    ]

    combined_raw, plan_normalization_notes = A._dedupe(
        previous + state_changing_new_actions
    )
    combined = A._applicable(base_net, combined_raw)

    attempted_after = M._measure_day(profile, combined)

    trial = copy.deepcopy(base_net)
    attempted_cost, plan_descriptions, plan_apply_notes = A._apply(
        trial,
        combined,
    )

    candidate_score = _selection_score(attempted_after, attempted_cost)
    parent_score = _selection_score(state_before, EP._plan_cost(profile, previous))
    relation = _score_relation(candidate_score, parent_score)
    outcome_label = (
        "better"
        if _score_improves(candidate_score, parent_score)
        else (
            "worse"
            if relation in {"security5_dominated", "security5_equal_higher_cost"}
            else "mixed" if relation == "security5_tradeoff" else "no_change"
        )
    )

    solved = _security5_solved(attempted_after)
    if solved:
        outcome_label = "solved"

    normalization_notes = _unique(batch_normalization_notes + plan_normalization_notes)
    invalid_notes = _unique(action_notes + plan_apply_notes)

    return {
        "submitted_actions": list(actions),
        "recorded_actions": recorded_actions,
        "full_plan": list(combined),
        "state": attempted_after,
        "cost_musd": float(attempted_cost),
        "plan_descriptions": plan_descriptions,
        "normalization_notes": normalization_notes,
        "no_effect_or_invalid_action_notes": invalid_notes,
        "outcome_label": outcome_label,
        "solved": solved,
        "score": candidate_score,
    }


def _individual_action_evaluations(
    profile: str,
    previous: list[dict],
    state_before: dict,
    base_net,
    catalog: dict,
    previous_cost: float,
) -> list[dict]:
    parent_vector = _security5_vector(state_before)
    rows = []
    for ref, entry in catalog.items():

        action = copy.deepcopy(entry["action"])
        try:
            observation = _evaluate_candidate_from_parent(
                profile, previous, state_before, base_net, [action]
            )
        except Exception as exc:

            trial_cost = None
            try:
                trial = copy.deepcopy(base_net)
                plan_raw, _ = A._dedupe(previous + [action])
                cost_plan = A._applicable(base_net, plan_raw)
                trial_cost = float(A._apply(trial, cost_plan)[0])
            except Exception:
                pass
            rows.append(
                {
                    "action_ref": ref,
                    "source_candidate_id": entry["candidate_id"],
                    "action": action,
                    "equipment_key": copy.deepcopy(entry["equipment_key"]),
                    "evaluation_status": "error",
                    "evaluation_error": f"{type(exc).__name__}: {exc}"[:500],
                    "security5": None,
                    "security5_change_vs_parent": None,
                    "security5_relation_vs_parent": None,
                    "parent_plan_cost_musd": round(float(previous_cost), 4),
                    "plan_cost_musd": (
                        round(trial_cost, 4) if trial_cost is not None else None
                    ),
                    "added_cost_musd": (
                        round(trial_cost - previous_cost, 4)
                        if trial_cost is not None
                        else None
                    ),
                    "all_base_case_converged": None,
                    "nonconverged_base_case_hours": None,
                    "n_failed_n1_outage_tests": None,
                    "n_base_case_problem_hours": None,
                    "solved": False,
                    "no_effect_or_invalid_action_notes": [],
                }
            )
            continue

        vector = _security5_vector(observation["state"])
        trial_cost = float(observation["cost_musd"])
        st = observation["state"]
        rows.append(
            {
                "action_ref": ref,
                "source_candidate_id": entry["candidate_id"],
                "action": action,
                "equipment_key": copy.deepcopy(entry["equipment_key"]),
                "evaluation_status": "verified",
                "evaluation_error": None,
                "security5": list(vector),
                "security5_change_vs_parent": [
                    round(float(v - p), 4) for v, p in zip(vector, parent_vector)
                ],
                "security5_relation_vs_parent": _score_relation(
                    observation["score"], _selection_score(state_before, previous_cost)
                ),
                "parent_plan_cost_musd": round(float(previous_cost), 4),
                "plan_cost_musd": round(trial_cost, 4),
                "added_cost_musd": round(trial_cost - previous_cost, 4),
                "all_base_case_converged": not bool(
                    st.get("nonconverged_base_case_hours", [])
                ),
                "nonconverged_base_case_hours": list(
                    st.get("nonconverged_base_case_hours", [])
                ),
                "n_failed_n1_outage_tests": int(st.get("n_failed_n1_outage_tests", 0)),
                "n_base_case_problem_hours": int(
                    st.get("n_base_case_problem_hours", 0)
                ),
                "solved": bool(observation["solved"]),
                "no_effect_or_invalid_action_notes": list(
                    observation["no_effect_or_invalid_action_notes"]
                ),
            }
        )
    return rows


def _action_evaluation_summary(ctx: dict) -> dict:
    rows = ctx.get("action_evaluations", [])
    return {
        "action_evaluation_mode": ctx.get("action_evaluation_mode", "off"),
        "action_evaluations": copy.deepcopy(rows),
        "verification_counts": {
            "individual_action": len(rows),
            "sibling_full_plan": 3,
            "crossover_full_plan": int(
                any(item.get("candidate_id") == "D" for item in ctx["evaluated"])
            ),
            "fusion_batch_full_plan": int(bool(ctx.get("fusion_batch_evaluation"))),
        },
        "fusion_batch_evaluation": copy.deepcopy(ctx.get("fusion_batch_evaluation")),
        "fusion_superior_to_all_siblings": bool(
            ctx.get("fusion_superior_to_all_siblings", False)
        ),
        "fusion_source_action_refs": list(ctx.get("fusion_source_action_refs", [])),
    }


def _model_candidate_result(
    item: dict, *, selected: bool = False, committed: bool = False
) -> dict:
    st = item["state"]
    return {
        "candidate_id": item["candidate_id"],
        "strategy": item.get("strategy", ""),
        "improving": bool(item.get("improving", False)),
        "selected_as_best": bool(selected),
        "committed": bool(committed),
        "solved": bool(item["solved"]),
        "outcome_label": item["outcome_label"],
        "security5_relation_vs_parent": item.get("security5_relation_vs_parent"),
        "security5_change_vs_parent": item.get("security5_change_vs_parent", {}),
        "security5": st.get("security5", {}),
        "score": _score_record(item["score"]),
        "n_base_case_problem_hours": st["n_base_case_problem_hours"],
        "plan_cost_musd": round(float(item["cost_musd"]), 1),
        "n1_overloaded_elements": st["n1_overloaded_elements"][:8],
        "n1_voltage_problem_buses": st["n1_voltage_problem_buses"][:6],
        "base_case_overloaded_elements": st["base_case_overloaded_elements"][:8],
        "base_case_voltage_problem_buses": st["base_case_voltage_problem_buses"][:6],
        "normalization_notes": item["normalization_notes"],
        "no_effect_or_invalid_action_notes": item["no_effect_or_invalid_action_notes"],
        "fusion_superior_to_all_siblings": item.get("fusion_superior_to_all_siblings"),
        "fusion_source_action_refs": item.get("source_action_refs", []),
        "fusion_batch_verified": bool(item.get("fusion_batch_evaluation")),
        "fusion_comparison": copy.deepcopy(item.get("fusion_comparison", {})),
    }


def _candidate_event_record(
    profile: str,
    parent_score: tuple[tuple[float, float, float, float, float], float],
    evaluated: list[dict],
    *,
    status: str,
    eligible_ids: list[str],
    selected_candidate_id: str | None,
    selected_by: str | None,
    committed: bool,
) -> dict:
    event = {
        "type": "candidate_plan_round",
        "profile": profile,
        "selection_rule": (
            "strict parent improvement uses Security5 [T_B,V_B,T_N1,V_N1,F_N1]: "
            "candidate must be no worse in all five and better in at least one; "
            "if Security5 is effectively equal, lower total plan cost may break the tie. "
            "Exactly one improving sibling auto-commits; if two or more improve, "
            "the LLM must choose one using verified Security5 and cost; physical diagnostics are explanatory only."
        ),
        "parent_score": _score_record(parent_score),
        "status": status,
        "eligible_candidate_ids": list(eligible_ids),
        "selected_candidate_id": selected_candidate_id,
        "selected_by": selected_by,
        "selected_candidate_committed": bool(committed),
        "candidates": [],
    }
    for item in evaluated:
        event["candidates"].append(
            {
                "candidate_id": item["candidate_id"],
                "strategy": item.get("strategy", ""),
                **(
                    {
                        "crossover_decision_rationale": item.get("strategy", ""),
                        "crossover_edit": copy.deepcopy(item.get("crossover_edit")),
                        "crossover_is_anchor_copy": bool(
                            item.get("crossover_is_anchor_copy", False)
                        ),
                        "source_action_refs": list(item.get("source_action_refs", [])),
                    }
                    if item.get("candidate_id") == "D"
                    else {}
                ),
                "improving": bool(item.get("improving", False)),
                "submitted_actions": item["submitted_actions"],
                "recorded_actions": item["recorded_actions"],
                "full_plan": item["full_plan"],
                "score": _score_record(item["score"]),
                "security5_relation_vs_parent": item.get(
                    "security5_relation_vs_parent"
                ),
                "security5_change_vs_parent": item.get(
                    "security5_change_vs_parent", {}
                ),
                "solved": bool(item["solved"]),
                "outcome_label": item["outcome_label"],
                "state": EP._episode_state(item["state"]),
                "normalization_notes": item["normalization_notes"],
                "no_effect_or_invalid_action_notes": item[
                    "no_effect_or_invalid_action_notes"
                ],
            }
        )
    return event


def _commit_evaluated_candidate(
    *,
    profile: str,
    chosen: dict,
    previous: list[dict],
    state_before: dict,
    previous_cost: float,
    original: dict,
    episode_id: str,
    evaluated: list[dict],
    parent_score: tuple[tuple[float, float, float, float, float], float],
    selected_by: str,
) -> dict:
    effective_plan = list(chosen["full_plan"])
    effective_after = chosen["state"]
    effective_cost = float(chosen["cost_musd"])

    plan_changed = effective_plan != previous
    search_decision = "kept" if plan_changed else "unchanged"
    if plan_changed:
        S.PLAN["kept_updates"] += 1
    S.PLAN["actions"] = list(effective_plan)

    base_net, _ = N._setup(profile, N.peak_hour_of(profile))
    retained_trial = copy.deepcopy(base_net)
    _, retained_plan_descriptions, _ = A._apply(retained_trial, effective_plan)

    consequence = EP._episode_consequence(state_before, effective_after)
    test_plan_call = E.append_step(
        C.MEMORY,
        episode_id,
        state_before=EP._episode_state(state_before),
        attempted_actions=chosen["recorded_actions"],
        attempted_state=EP._episode_state(effective_after),
        consequence=consequence,
        costs={
            "plan_before_cost_musd": round(previous_cost, 4),
            "attempted_plan_cost_musd": round(effective_cost, 4),
            "attempted_plan_cost_delta_musd": round(effective_cost - previous_cost, 4),
            "realized_plan_cost_musd": round(effective_cost, 4),
            "realized_plan_cost_delta_musd": round(effective_cost - previous_cost, 4),
        },
        search_decision=search_decision,
        outcome_label=chosen["outcome_label"],
        normalization_notes=chosen["normalization_notes"],
        no_effect_or_invalid_action_notes=chosen["no_effect_or_invalid_action_notes"],
    )

    selection_event = E.append_memory_event(
        C.MEMORY,
        episode_id,
        {
            "type": "candidate_plan_selection",
            "round_id": chosen.get("round_id"),
            "source_action_refs": chosen.get("source_action_refs", []),
            "profile": profile,
            "selected_candidate_id": chosen["candidate_id"],
            "selected_by": selected_by,
            "eligible_candidate_ids": [
                item["candidate_id"] for item in evaluated if item.get("improving")
            ],
            "parent_score": _score_record(parent_score),
            "selected_score": _score_record(chosen["score"]),
            "selected_plan_cost_musd": round(effective_cost, 4),
        },
    )

    solved = _security5_solved(effective_after)
    final_plan = None
    if solved:
        final_plan = EP._record_final_plan(profile, effective_plan)
        E.finish_episode(
            C.MEMORY,
            episode_id,
            solved=True,
            final_plan=final_plan,
            total_cost_musd=effective_cost,
            final_state=EP._episode_state(effective_after),
            n_kept_plan_updates=int(S.PLAN.get("kept_updates", 0)),
            status="solved",
        )
        S.ACTIVE_EPISODE_ID = None
        S.ACTIVE_EPISODE_PROFILE = None

    note = (
        f"Candidate {chosen['candidate_id']} was committed by {selected_by}. "
        "The returned Security5 and physical diagnostics are the authoritative CURRENT retained state. "
        "Continue the next Best-of-3 round from this Security5 state and these diagnostics; do not call analyse again "
        "to refresh the post-commit state."
    )
    if solved:
        note = (
            f"Candidate {chosen['candidate_id']} was committed by {selected_by} and solved the scenario: "
            "all five Security5 components are zero."
        )

    return {
        "solved": solved,
        "search_decision": search_decision,
        "outcome_label": "solved" if solved else chosen["outcome_label"],
        "selection_required": False,
        "selected_candidate_id": chosen["candidate_id"],
        "selected_candidate_committed": True,
        "selected_by": selected_by,
        "selection_rule": (
            "Security5-first parent improvement; single improving sibling auto-commits; "
            "multiple improving siblings require explicit LLM selection using verified "
            "Security5 and total plan cost; physical diagnostics are explanatory only"
        ),
        "parent_score": _score_record(parent_score),
        "selected_score": _score_record(chosen["score"]),
        "candidate_results": [
            _model_candidate_result(
                item,
                selected=(item["candidate_id"] == chosen["candidate_id"]),
                committed=(item["candidate_id"] == chosen["candidate_id"]),
            )
            for item in evaluated
        ],
        "candidate_selection_memory_event": selection_event,
        "n_plan_actions": len(effective_plan),
        "plan_cost_musd": round(effective_cost, 1),
        "security5": effective_after.get("security5", {}),
        "n_base_case_problem_hours": effective_after["n_base_case_problem_hours"],
        "retained_current_plan_descriptions": retained_plan_descriptions,
        "base_case_overloaded_elements": effective_after[
            "base_case_overloaded_elements"
        ][: C.LIST_LIMIT],
        "base_case_voltage_problem_buses": effective_after[
            "base_case_voltage_problem_buses"
        ][: C.LIST_LIMIT],
        "n1_overloaded_elements": effective_after["n1_overloaded_elements"][
            : C.LIST_LIMIT
        ],
        "n1_voltage_problem_buses": effective_after["n1_voltage_problem_buses"][:12],
        "base_case_worst_hours": effective_after["base_case_worst_hours"][:3],
        "current_plan": list(effective_plan),
        "attempted_plan_cost_musd": round(effective_cost, 1),
        "attempted_security5": effective_after.get("security5", {}),
        "attempted_plan_descriptions": chosen["plan_descriptions"],
        "normalization_notes": chosen["normalization_notes"],
        "no_effect_or_invalid_action_notes": chosen[
            "no_effect_or_invalid_action_notes"
        ],
        "final_plan": final_plan,
        "episode_id": episode_id,
        "episode_test_plan_call": test_plan_call,
        "episode_file": f"memory/episodes/{episode_id}.json",
        "note": note,
    }


def _is_exact_parent_noop(action, record):
    if record.get("simulator_applied") is not False:
        return False

    kind = action.get("action")
    if kind != record.get("action"):
        return False

    fields = {
        "upgrade_line": (
            "target_capacity_mva",
            "from_capacity_mva",
            "to_capacity_mva",
        ),
        "upgrade_transformer": (
            "target_capacity_mva",
            "from_sn_mva",
            "to_sn_mva",
        ),
        "set_bus_reactive_support": (
            "target_mvar",
            "from_reactive_support_mvar",
            "to_reactive_support_mvar",
        ),
    }

    if kind not in fields:
        return False

    if kind == "upgrade_line":
        if not record.get("line") or action.get("line") != record["line"]:
            return False

    elif kind == "upgrade_transformer":
        if (
            not record.get("transformer")
            or action.get("transformer") != record["transformer"]
        ):
            return False

    elif kind == "set_bus_reactive_support":
        try:
            if N._busnum(action["bus"]) != N._busnum(record["bus"]):
                return False
        except (KeyError, ValueError, TypeError):
            return False

    requested_field, before_field, after_field = fields[kind]

    try:
        values = [
            float(action[requested_field]),
            float(record[before_field]),
            float(record[after_field]),
        ]
    except (KeyError, ValueError, TypeError, OverflowError):
        return False

    if not all(math.isfinite(value) for value in values):
        return False

    requested, before, after = values
    return math.isclose(requested, before, rel_tol=0, abs_tol=1e-9) and math.isclose(
        before, after, rel_tol=0, abs_tol=1e-9
    )


def _candidate_action_validation_error(base_net, previous, candidates):
    parent_net = copy.deepcopy(base_net)
    if previous:
        A._apply(parent_net, previous)

    errors = []
    for candidate in candidates:
        cid = candidate["candidate_id"]
        actions, normalization_notes = A._dedupe(candidate["actions"])
        records, action_notes = A.record_action_transitions(parent_net, actions)

        invalid_actions = [
            A.describe_action(action)
            for action, record in zip(actions, records)
            if not record.get("simulator_applied")
            and not _is_exact_parent_noop(action, record)
        ]

        exact_noops = [
            A.describe_action(action)
            for action, record in zip(actions, records)
            if _is_exact_parent_noop(action, record)
        ]
        if exact_noops:
            print(
                "[candidate validation] Ignoring exact parent-equivalent "
                f"targets in candidate {cid}: {exact_noops}",
                flush=True,
            )

        if invalid_actions:
            errors.append(
                {
                    "candidate_id": cid,
                    "invalid_or_no_effect_actions": invalid_actions,
                    "notes": _unique(action_notes),
                    "normalization_notes": _unique(normalization_notes),
                }
            )

    if not errors:
        return None

    summary = []
    for item in errors:
        detail = "; ".join(item["invalid_or_no_effect_actions"])
        notes = "; ".join(item["notes"])
        text = f"Candidate {item['candidate_id']}: {detail}"
        if notes:
            text += f" ({notes})"
        summary.append(text)

    return {
        "error": (
            "Invalid or no-effect candidate actions. No siblings were tested and "
            "the retained parent is unchanged. "
            + " | ".join(summary)
            + ". Omit unchanged parent targets; use only valid new or revised targets."
        ),
        "error_code": "invalid_or_no_effect_candidate_actions",
        "candidate_action_errors": errors,
        "parent_unchanged": True,
    }


def _candidate_diversity_error(base_net, previous, candidates):
    parent_net = copy.deepcopy(base_net)
    if previous:
        A._apply(parent_net, previous)

    transition_fields = {
        "upgrade_line": ("from_capacity_mva", "to_capacity_mva"),
        "upgrade_transformer": ("from_sn_mva", "to_sn_mva"),
        "new_line": ("from_capacity_mva", "to_capacity_mva"),
        "set_bus_reactive_support": (
            "from_reactive_support_mvar",
            "to_reactive_support_mvar",
        ),
    }

    footprints = {}

    for candidate in candidates:
        actions, _ = A._dedupe(candidate["actions"])
        records, _ = A.record_action_transitions(parent_net, actions)

        transitions = {}

        for record in records:
            if not record.get("simulator_applied"):
                continue

            kind = record["action"]

            if kind == "upgrade_line":
                key = (kind, record["line"])

            elif kind == "upgrade_transformer":
                key = (kind, record["transformer"])

            elif kind == "set_bus_reactive_support":
                key = (kind, N._busnum(record["bus"]))

            elif kind == "new_line":
                endpoints = sorted(
                    (
                        N._busnum(record["from_bus"]),
                        N._busnum(record["to_bus"]),
                    )
                )
                key = (kind, *endpoints)

            else:
                continue

            before_key, after_key = transition_fields[kind]
            before = float(record[before_key])
            after = float(record[after_key])

            first = transitions.get(key, (before, before))[0]
            transitions[key] = (first, after)

        cid = candidate["candidate_id"]

        footprints[cid] = frozenset(
            (*key, round(float(after), 9))
            for key, (before, after) in transitions.items()
            if not math.isclose(before, after, abs_tol=1e-9)
        )

        if not footprints[cid]:
            return {
                "error": (
                    f"Candidate {cid} makes no effective change "
                    "to the retained parent. Revise its actions."
                ),
                "error_code": "empty_effective_candidate",
                "candidate_id": cid,
            }

    ids = list(footprints)

    conflicts = [
        [left, right]
        for i, left in enumerate(ids)
        for right in ids[i + 1 :]
        if footprints[left] == footprints[right]
    ]

    if conflicts:
        return {
            "error": "Each sibling must differ in equipment, action type, or target.",
            "error_code": "candidate_not_distinct",
        }

    return None


def _decorate_evaluated_candidate(result, candidate, state_before, parent_score):
    result["candidate_id"] = candidate["candidate_id"]
    result["strategy"] = candidate["strategy"]
    result["improving"] = _feasibility_aware_improvement(
        result, state_before, parent_score
    )
    relation = _score_relation(result["score"], parent_score)
    result["security5_relation_vs_parent"] = relation
    result["security5_change_vs_parent"] = _security5_change(
        result["state"], state_before
    )
    if result["solved"]:
        result["outcome_label"] = "solved"
    elif result["improving"]:
        result["outcome_label"] = "better"
    elif relation in {"security5_dominated", "security5_equal_higher_cost"}:
        result["outcome_label"] = "worse"
    elif relation == "security5_tradeoff":
        result["outcome_label"] = "mixed"
    else:
        result["outcome_label"] = "no_change"
    return result


def _finish_staged_round(profile, candidates, _verified_stage):
    global _PENDING_CANDIDATE_SELECTION

    if _PENDING_CANDIDATE_SELECTION is not None:
        return {
            "error": (
                "An unfinished candidate round exists. Complete its "
                + _PENDING_CANDIDATE_SELECTION.get("phase", "selection")
                + " stage before submitting another A/B/C set."
            )
        }

    if isinstance(candidates, str):
        try:
            candidates = json.loads(candidates)
        except json.JSONDecodeError:
            return {
                "error": "'candidates' must be a JSON array of exactly 3 candidate objects."
            }

    if not isinstance(candidates, list) or len(candidates) != 3:
        return {
            "error": "A verified round requires exactly 3 candidate objects: A, B, and C."
        }
    if any(not isinstance(candidate, dict) for candidate in candidates):
        return {"error": "Every candidate must be one JSON object."}

    parsed = []
    seen_ids = set()
    for i, candidate in enumerate(candidates):
        cid = str(candidate.get("candidate_id", chr(ord("A") + i))).strip().upper()
        actions = candidate.get("actions")
        if cid not in {"A", "B", "C"}:
            return {
                "error": f"Invalid candidate_id '{cid}'. Use A, B, and C exactly once."
            }
        if cid in seen_ids:
            return {
                "error": f"Duplicate candidate_id '{cid}'. Use A, B, and C exactly once."
            }
        if not isinstance(actions, list) or not actions:
            return {
                "error": f"Candidate {cid} must contain a non-empty 'actions' array."
            }
        if any(not isinstance(action, dict) for action in actions):
            return {
                "error": f"Every action in candidate {cid} must be one JSON object."
            }
        seen_ids.add(cid)
        parsed.append(
            {
                "candidate_id": cid,
                "strategy": str(candidate.get("strategy", "")).strip(),
                "actions": actions,
            }
        )
    if seen_ids != {"A", "B", "C"}:
        return {
            "error": "Candidates must contain candidate_id A, B, and C exactly once."
        }
    parsed.sort(key=lambda item: item["candidate_id"])

    parent_actions = list(S.PLAN["actions"]) if S.PLAN["key"] == profile else []

    diversity_base_net, _ = N._setup(
        profile,
        N.peak_hour_of(profile),
    )

    validation_error = _candidate_action_validation_error(
        diversity_base_net,
        parent_actions,
        parsed,
    )
    if validation_error is not None:
        return validation_error

    diversity_error = _candidate_diversity_error(
        diversity_base_net,
        parent_actions,
        parsed,
    )

    if diversity_error is not None:
        return diversity_error

    if S.PLAN["key"] != profile:
        if S.ACTIVE_EPISODE_ID is not None:
            different_profile = S.ACTIVE_EPISODE_PROFILE != profile
            if different_profile:
                EP.finish_episode(S.ACTIVE_EPISODE_PROFILE, status="replaced")
        S.PLAN["key"] = profile
        S.PLAN["actions"] = []
        S.PLAN["kept_updates"] = 0

    previous = list(S.PLAN["actions"])
    original = copy.deepcopy(_verified_stage["original"])
    state_before = copy.deepcopy(_verified_stage["state_before"])
    previous_cost = float(_verified_stage["previous_cost"])
    S.DAY_CACHE.setdefault(("analyse", profile), copy.deepcopy(original))
    parent_score = _selection_score(state_before, previous_cost)
    episode_id = EP._start_episode_if_needed(profile, original)
    base_net, _ = N._setup(profile, N.peak_hour_of(profile))

    parent_net = copy.deepcopy(base_net)
    A._apply(parent_net, previous)
    catalog = {}
    for candidate in parsed:
        for index, (equipment_key, action) in enumerate(
            _effective_actions(parent_net, candidate["actions"]), 1
        ):
            ref = f"{candidate['candidate_id']}:{index}"
            catalog[ref] = {
                "candidate_id": candidate["candidate_id"],
                "equipment_key": list(equipment_key),
                "action": action,
            }

    if (
        _verified_stage.get("profile") != profile
        or _verified_stage.get("previous") != previous
        or _verified_stage.get("candidates") != parsed
        or len(_verified_stage.get("evaluated", [])) != 3
        or _verified_stage.get("action_catalog") != catalog
    ):
        return {"error": "Sequential verification became stale; no plan was committed."}

    action_evaluations = copy.deepcopy(_verified_stage["action_evaluations"])
    evaluated = copy.deepcopy(_verified_stage["evaluated"])

    context = {
        "profile": profile,
        "round_id": uuid.uuid4().hex,
        "phase": "crossover",
        "previous": copy.deepcopy(previous),
        "state_before": copy.deepcopy(state_before),
        "previous_cost": float(previous_cost),
        "original": copy.deepcopy(original),
        "episode_id": episode_id,
        "parent_score": parent_score,
        "evaluated": copy.deepcopy(evaluated),
        "action_catalog": catalog,
        "action_evaluation_mode": "individual",
        "action_evaluations": action_evaluations,
    }

    for item in context["evaluated"]:
        item["round_id"] = context["round_id"]

    _PENDING_CANDIDATE_SELECTION = context
    event_id = _record_round(context, "awaiting_crossover")

    result = _round_view(context)
    result.update(
        crossover_required=True,
        search_decision="awaiting_crossover",
        outcome_label="awaiting_crossover",
        action_catalog=catalog,
        candidate_memory_event=event_id,
        note=(
            "A/B/C full plans were tested; available independent single-action "
            "AC results are in action_evaluations. No candidate is committed. "
            "Construct D using exact action_refs from at least two "
            "source candidates. Choose one target per equipment. "
            "Call test_crossover_plan with this profile and round_id."
        ),
    )
    return result


def reset_sequential_candidates() -> None:
    global _PENDING_SEQUENTIAL_CANDIDATES
    _PENDING_SEQUENTIAL_CANDIDATES = None


def test_staged_candidate(
    profile: str = "normal_001",
    candidate_id: str = "A",
    strategy: str = "",
    actions: list | str | None = None,
) -> dict:
    global _PENDING_SEQUENTIAL_CANDIDATES
    if _PENDING_CANDIDATE_SELECTION is not None:
        return {
            "error": "Finish the pending crossover/commit before another candidate."
        }
    if isinstance(actions, str):
        try:
            actions = json.loads(actions)
        except (TypeError, ValueError):
            return {"error": "actions must be a JSON array"}
    if (
        not isinstance(actions, list)
        or not actions
        or not all(isinstance(a, dict) for a in actions)
    ):
        return {"error": "Each candidate requires a nonempty action-object list."}
    cid = str(candidate_id).strip().upper()
    if cid != "C" and any(a.get("action") == "new_line" for a in actions):
        return {"error": "Only Candidate C may propose a new line."}
    staged = _PENDING_SEQUENTIAL_CANDIDATES
    expected = "ABC"[len(staged["candidates"])] if staged is not None else "A"
    if cid != expected:
        return {"error": f"Expected Candidate {expected} in the sequential stage."}
    if staged is not None and (
        staged["profile"] != profile
        or S.PLAN["key"] != profile
        or S.PLAN["actions"] != staged["previous"]
    ):
        _PENDING_SEQUENTIAL_CANDIDATES = None
        return {
            "error": "Stale retained parent or scenario: sequential round discarded. Start again from Candidate A.",
            "error_code": "stale_staged_parent",
            "next_candidate_id": "A",
        }
    base_net, _ = N._setup(profile, N.peak_hour_of(profile))
    previous = (
        copy.deepcopy(staged["previous"])
        if staged
        else (list(S.PLAN["actions"]) if S.PLAN["key"] == profile else [])
    )
    candidate = {
        "candidate_id": cid,
        "strategy": str(strategy).strip(),
        "actions": actions,
    }
    validation = _candidate_action_validation_error(base_net, previous, [candidate])
    if validation:
        return validation

    effective_check = _candidate_diversity_error(base_net, previous, [candidate])
    if (
        effective_check
        and effective_check.get("error_code") == "empty_effective_candidate"
    ):
        return {
            "error": (
                f"Candidate {cid} makes no effective physical change "
                "to the current retained parent. Do not repeat "
                "already-committed targets. Propose at least one "
                "valid new or revised equipment setting."
            ),
            "error_code": "empty_effective_candidate",
            "parent_unchanged": True,
        }

    if staged is not None:
        diverse = _candidate_diversity_error(
            base_net, previous, staged["candidates"] + [candidate]
        )
        if diverse:

            return {
                "error": "This proposed plan has the same effective intervention footprint as an existing plan. Select a different effective intervention strategy.",
                "error_code": "candidate_not_distinct",
                "parent_unchanged": True,
            }
    if staged is None:
        original = S.DAY_CACHE.get(("analyse", profile))
        if original is None:
            original = M._measure_day(profile, [])
        before = M._measure_day(profile, previous) if previous else original
        previous_cost = EP._plan_cost(profile, previous)
        staged = {
            "profile": profile,
            "previous": previous,
            "original": copy.deepcopy(original),
            "candidates": [],
            "evaluated": [],
            "action_catalog": {},
            "action_evaluations": [],
            "state_before": before,
            "previous_cost": previous_cost,
        }
    parent_net = copy.deepcopy(base_net)
    if previous:
        A._apply(parent_net, previous)
    candidate_catalog = {
        f"{cid}:{i}": {
            "candidate_id": cid,
            "equipment_key": list(key),
            "action": action,
        }
        for i, (key, action) in enumerate(_effective_actions(parent_net, actions), 1)
    }
    try:
        observations = _individual_action_evaluations(
            profile,
            previous,
            staged["state_before"],
            base_net,
            candidate_catalog,
            staged["previous_cost"],
        )
        complete = _evaluate_candidate_from_parent(
            profile, previous, staged["state_before"], base_net, actions
        )
    except Exception as exc:
        return {
            "error": f"Candidate {cid} AC verification failed: {type(exc).__name__}: {exc}"[
                :600
            ]
        }
    full = _decorate_evaluated_candidate(
        complete,
        candidate,
        staged["state_before"],
        _selection_score(staged["state_before"], staged["previous_cost"]),
    )

    if staged is not None and not staged["candidates"] and S.PLAN["key"] != profile:
        if S.ACTIVE_EPISODE_ID is not None and S.ACTIVE_EPISODE_PROFILE != profile:
            EP.finish_episode(S.ACTIVE_EPISODE_PROFILE, status="replaced")
        S.PLAN["key"] = profile
        S.PLAN["actions"] = []
        S.PLAN["kept_updates"] = 0

    staged["candidates"].append(copy.deepcopy(candidate))
    staged["evaluated"].append(copy.deepcopy(full))
    staged["action_catalog"].update(candidate_catalog)
    staged["action_evaluations"].extend(observations)
    if cid != "C":
        _PENDING_SEQUENTIAL_CANDIDATES = staged

        st = full["state"]
        full_plan_pf = {
            "security5": copy.deepcopy(st.get("security5", {})),
            "security5_change_vs_parent": copy.deepcopy(
                full.get("security5_change_vs_parent", {})
            ),
            "security5_relation_vs_parent": full.get("security5_relation_vs_parent"),
            "plan_cost_musd": round(float(full["cost_musd"]), 4),
            "added_cost_musd": round(
                float(full["cost_musd"]) - float(staged["previous_cost"]), 4
            ),
            "solved": bool(full["solved"]),
            "outcome_label": full["outcome_label"],
            "n_base_case_problem_hours": st.get("n_base_case_problem_hours"),
            "nonconverged_base_case_hours": copy.deepcopy(
                st.get("nonconverged_base_case_hours", [])
            ),
            "n_failed_n1_outage_tests": st.get("n_failed_n1_outage_tests"),
        }
        return {
            "staged_candidate_id": cid,
            "stage_verified": True,
            "next_candidate_id": "B" if cid == "A" else "C",
            "parent_unchanged": True,
            "action_verifier_evals": len(observations),
            "full_plan_verifier_evals": 1,
            "full_plan_pf": full_plan_pf,
            "evidence_visibility": "controller_only_until_D",
        }

    _PENDING_SEQUENTIAL_CANDIDATES = None
    final = _finish_staged_round(
        profile=profile,
        candidates=staged["candidates"],
        _verified_stage=staged,
    )
    if final.get("error"):

        staged["candidates"].pop()
        staged["evaluated"].pop()
        for ref in candidate_catalog:
            staged["action_catalog"].pop(ref, None)
        staged["action_evaluations"] = [
            row
            for row in staged["action_evaluations"]
            if row.get("source_candidate_id") != "C"
        ]
        _PENDING_SEQUENTIAL_CANDIDATES = staged
    return final


def _normalize_candidate_id(value):
    """Remove matching quote wrappers without guessing a candidate."""
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


def commit_candidate(
    profile: str = "normal_001",
    candidate_id: str | None = None,
    *,
    selected_by: str = "llm_reasoned_selection",
) -> dict:
    global _PENDING_CANDIDATE_SELECTION

    pending = _PENDING_CANDIDATE_SELECTION
    if pending is None:
        return {
            "error": (
                "No candidate selection is pending. Complete the staged siblings first. "
                "commit_candidate is only valid after 2 or more siblings strictly improve the same parent."
            )
        }
    if pending.get("profile") != profile:
        return {
            "error": (
                f"Pending candidate selection belongs to profile {pending.get('profile')}, not {profile}."
            )
        }
    if pending.get("phase") != "selection":
        return {"error": "Test crossover D before committing a candidate."}

    if S.PLAN["key"] != profile or S.PLAN["actions"] != pending["previous"]:
        return {"error": "The retained parent changed. Refusing a stale commit."}

    cid = _normalize_candidate_id(candidate_id)
    eligible = list(pending.get("eligible_candidate_ids", []))

    if cid not in eligible:
        return {
            "error": (
                f"Candidate input {candidate_id!r} "
                f"(normalized {cid!r}) is not eligible for commit. "
                f"Verified improving candidates are {eligible}. "
                "Choose one of those IDs."
            )
        }

    chosen = next(item for item in pending["evaluated"] if item["candidate_id"] == cid)

    result = _commit_evaluated_candidate(
        profile=profile,
        chosen=chosen,
        previous=pending["previous"],
        state_before=pending["state_before"],
        previous_cost=pending["previous_cost"],
        original=pending["original"],
        episode_id=pending["episode_id"],
        evaluated=pending["evaluated"],
        parent_score=pending["parent_score"],
        selected_by=selected_by,
    )
    result.update(_action_evaluation_summary(pending))
    result["eligible_candidate_ids"] = eligible
    result["prior_candidate_memory_event"] = pending.get("candidate_memory_event")
    result["note"] = (
        f"Candidate {cid} selected by {selected_by} from verified improving candidates {eligible}. "
        + result["note"]
    )
    _PENDING_CANDIDATE_SELECTION = None
    return result


def _effective_actions(parent_net, actions):
    records, _ = A.record_action_transitions(parent_net, actions)
    folded = {}

    for rec in records:
        if not rec.get("simulator_applied"):
            continue

        kind = rec["action"]

        if kind == "upgrade_line":
            key = (kind, rec["line"])
            before = rec["from_capacity_mva"]
            after = rec["to_capacity_mva"]
            action = {
                "action": kind,
                "line": rec["line"],
                "target_capacity_mva": after,
            }

        elif kind == "upgrade_transformer":
            key = (kind, rec["transformer"])
            before = rec["from_sn_mva"]
            after = rec["to_sn_mva"]
            action = {
                "action": kind,
                "transformer": rec["transformer"],
                "target_sn_mva": after,
            }

        elif kind == "new_line":
            fb, tb = sorted((rec["from_bus"], rec["to_bus"]))
            key = (kind, fb, tb)
            before = rec["from_capacity_mva"]
            after = rec["to_capacity_mva"]
            action = {
                "action": kind,
                "from_bus": fb,
                "to_bus": tb,
                "capacity_mva": after,
            }

        elif kind == "set_bus_reactive_support":
            key = (kind, rec["bus"])
            before = rec["from_reactive_support_mvar"]
            after = rec["to_reactive_support_mvar"]
            action = {
                "action": kind,
                "bus": rec["bus"],
                "target_mvar": after,
            }

        else:
            continue

        original = folded[key][0] if key in folded else before
        folded[key] = (original, after, action)

    return [
        (key, action)
        for key, (before, after, action) in folded.items()
        if not math.isclose(
            float(before),
            float(after),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ]


def _round_view(ctx):
    st = ctx["state_before"]

    return {
        "solved": False,
        "crossover_required": False,
        "selection_required": False,
        "selected_candidate_id": None,
        "selected_candidate_committed": False,
        "eligible_candidate_ids": [],
        "round_id": ctx["round_id"],
        "parent_score": _score_record(ctx["parent_score"]),
        "security5": st.get("security5", {}),
        "candidate_results": [
            _model_candidate_result(item) for item in ctx["evaluated"]
        ],
        **_action_evaluation_summary(ctx),
        "current_plan": copy.deepcopy(ctx["previous"]),
        "n_plan_actions": len(ctx["previous"]),
        "plan_cost_musd": round(ctx["previous_cost"], 4),
        "episode_id": ctx["episode_id"],
        "episode_file": f"memory/episodes/{ctx['episode_id']}.json",
        "n_base_case_problem_hours": st.get("n_base_case_problem_hours", 0),
        **{
            key: st.get(key, [])[: C.LIST_LIMIT]
            for key in (
                "base_case_overloaded_elements",
                "base_case_voltage_problem_buses",
                "n1_overloaded_elements",
                "n1_voltage_problem_buses",
                "base_case_worst_hours",
            )
        },
    }


def _record_round(ctx, status):
    eligible = [
        item["candidate_id"]
        for item in ctx["evaluated"]
        if _feasibility_aware_improvement(
            item, ctx["state_before"], ctx["parent_score"]
        )
        and (
            item["candidate_id"] != "D"
            or item.get("fusion_superior_to_all_siblings", False)
        )
    ]

    event = _candidate_event_record(
        ctx["profile"],
        ctx["parent_score"],
        ctx["evaluated"],
        status=status,
        eligible_ids=eligible,
        selected_candidate_id=None,
        selected_by=None,
        committed=False,
    )

    event.update(
        round_id=ctx["round_id"],
        parent_plan=copy.deepcopy(ctx["previous"]),
        state_before=EP._episode_state(ctx["state_before"]),
        action_catalog=copy.deepcopy(ctx.get("action_catalog", {})),
        **_action_evaluation_summary(ctx),
        crossover_skipped=bool(ctx.get("crossover_skipped", False)),
        crossover_skip_reason=ctx.get("crossover_skip_reason"),
    )

    for record, item in zip(event["candidates"], ctx["evaluated"]):
        record["source_action_refs"] = item.get("source_action_refs", [])

    return E.append_memory_event(
        C.MEMORY,
        ctx["episode_id"],
        event,
    )


def _complete_candidate_round(ctx):
    global _PENDING_CANDIDATE_SELECTION

    eligible = [
        item["candidate_id"]
        for item in ctx["evaluated"]
        if _feasibility_aware_improvement(
            item, ctx["state_before"], ctx["parent_score"]
        )
        and (
            item["candidate_id"] != "D"
            or item.get("fusion_superior_to_all_siblings", False)
        )
    ]

    if "D" in eligible:
        d = next(row for row in ctx["evaluated"] if row["candidate_id"] == "D")
        if not _d_strict_superior(d, ctx["evaluated"], _score_improves):
            eligible.remove("D")

    status = "awaiting_agent_selection" if eligible else "no_improving_candidate"

    event_id = _record_round(ctx, status)
    result = _round_view(ctx)
    result["candidate_memory_event"] = event_id

    if not eligible:
        _PENDING_CANDIDATE_SELECTION = None
        result.update(
            selection_required=False,
            search_decision="reverted",
            outcome_label="no_improving_candidate",
            note=(
                "No verified candidate strictly improves the parent. "
                "No commitment permitted; retained plan unchanged."
            ),
        )
        return result

    ctx.update(
        phase="selection",
        eligible_candidate_ids=eligible,
        candidate_memory_event=event_id,
    )
    _PENDING_CANDIDATE_SELECTION = copy.deepcopy(ctx)

    result.update(
        selection_required=True,
        eligible_candidate_ids=eligible,
        selected_candidate_id=None,
        selected_candidate_committed=False,
        search_decision="awaiting_agent_selection",
        outcome_label="awaiting_agent_selection",
        selection_rule=(
            "Agent selects among ALL verifier-approved parent-improving "
            "candidates. Security5 must never worsen. The agent may "
            "trade greater immediate security progress for lower cost. "
            "Only explicit commit_candidate can change the parent. "
            "Fusion D is eligible only if it strictly outperforms EACH A/B/C "
            "by componentwise Security5 dominance or equal-security lower cost."
        ),
        note=(
            f"Nothing has been committed. The agent must choose from "
            f"{eligible} and explicitly call commit_candidate."
        ),
    )
    return result


def _fusion_action_signature(catalog: dict, refs: list[str]) -> frozenset:
    """Canonical, physical D footprint. Exact ratings are part of the key."""
    return frozenset(
        (
            tuple(catalog[ref]["equipment_key"]),
            json.dumps(catalog[ref]["action"], sort_keys=True, allow_nan=False),
        )
        for ref in refs
    )


def _fusion_superiority(result: dict, siblings: list[dict]) -> dict:
    rows = []
    for item in siblings:
        relation = _score_relation(result["score"], item["score"])
        rows.append(
            {
                "candidate_id": item["candidate_id"],
                "relation": relation,
                "sibling_cost_musd": round(float(item["cost_musd"]), 4),
                "fusion_cost_musd": round(float(result["cost_musd"]), 4),
                "strictly_better": relation
                in {"security5_dominates", "security5_equal_lower_cost"},
            }
        )

    feasibility_preserved = not any(
        bool(item.get("solved")) and not bool(result.get("solved")) for item in siblings
    )
    return {
        "superior": (
            len(rows) == 3
            and all(row["strictly_better"] for row in rows)
            and feasibility_preserved
        ),
        "feasibility_preserved": feasibility_preserved,
        "comparisons": rows,
    }


def _evaluate_fusion_as_batch(profile, pending, base_net, action_refs, actions):
    measured = _evaluate_candidate_from_parent(
        profile,
        pending["previous"],
        pending["state_before"],
        base_net,
        [copy.deepcopy(action) for action in actions],
    )

    records = measured["recorded_actions"]

    if len(records) != len(actions) or any(
        not record.get("simulator_applied") for record in records
    ):
        raise ValueError("Fusion contains an invalid or ineffective action")

    if measured["no_effect_or_invalid_action_notes"]:
        raise ValueError(
            "Fusion contains invalid/no-effect actions: "
            + str(measured["no_effect_or_invalid_action_notes"])
        )

    audit = {
        "verification_mode": "single_batch_24h_AC_PF_plus_peak_N_minus_1",
        "action_refs": list(action_refs),
        "source_candidate_ids": [
            pending["action_catalog"][ref]["candidate_id"] for ref in action_refs
        ],
        "actions": copy.deepcopy(actions),
        "action_count": len(actions),
        "security5": list(_security5_vector(measured["state"])),
        "plan_cost_musd": round(float(measured["cost_musd"]), 4),
        "solved": bool(measured["solved"]),
        "n_base_case_problem_hours": int(
            measured["state"].get("n_base_case_problem_hours", 0)
        ),
        "n_failed_n1_outage_tests": int(
            measured["state"].get("n_failed_n1_outage_tests", 0)
        ),
        "nonconverged_base_case_hours": list(
            measured["state"].get("nonconverged_base_case_hours", [])
        ),
    }

    return measured, audit


def test_crossover_plan(
    profile: str = "normal_001",
    round_id: str | None = None,
    action_refs: list[str] | None = None,
    strategy: str = "",
) -> dict:
    pending = _PENDING_CANDIDATE_SELECTION

    if pending is None or pending.get("phase") != "crossover":
        return {"error": "No crossover is pending. Test A/B/C first."}

    if profile != pending["profile"] or round_id != pending["round_id"]:
        return {"error": "profile or round_id does not match the pending round."}

    if S.PLAN["key"] != profile or S.PLAN["actions"] != pending["previous"]:
        return {"error": "The retained parent changed. Refusing a stale crossover."}

    if (
        not isinstance(action_refs, list)
        or not action_refs
        or any(not isinstance(ref, str) for ref in action_refs)
    ):
        return {"error": "action_refs must be a nonempty list of catalog keys."}

    catalog = pending["action_catalog"]
    unknown = [ref for ref in action_refs if ref not in catalog]

    if unknown:
        return {"error": f"Unknown action_refs: {unknown}."}
    if len(action_refs) != len(set(action_refs)):
        return {
            "error": "Repeated action_refs are forbidden in fusion.",
            "error_code": "duplicate_fusion_refs",
        }

    entries = [catalog[ref] for ref in action_refs]
    sources = sorted({entry["candidate_id"] for entry in entries})
    if len(sources) < 2:
        return {
            "error": "D must contain unchanged source actions from at least two of A/B/C.",
            "error_code": "not_genuine_fusion",
        }

    equipment = [tuple(entry["equipment_key"]) for entry in entries]

    if len(set(equipment)) != len(equipment):
        by_equipment = {}
        for ref, key in zip(action_refs, equipment):
            by_equipment.setdefault(key, []).append(ref)
        collisions = [
            {"equipment_key": list(key), "conflicting_refs": refs}
            for key, refs in by_equipment.items()
            if len(refs) > 1
        ]
        return {
            "error": (
                "Repeated or conflicting equipment in D: "
                f"{collisions}. Use replace_refs for alternate settings."
            ),
            "error_code": "duplicate_crossover_equipment",
            "conflicts": collisions,
        }

    actions = [copy.deepcopy(entry["action"]) for entry in entries]

    d_signature = _fusion_action_signature(catalog, action_refs)
    for cid in ("A", "B", "C"):
        sibling_refs = [
            ref for ref, entry in catalog.items() if entry["candidate_id"] == cid
        ]
        if d_signature == _fusion_action_signature(catalog, sibling_refs):
            return {
                "error": f"D copies Candidate {cid} without a new physical combination.",
                "error_code": "fusion_copies_sibling",
            }

    if len({catalog[ref]["candidate_id"] for ref in action_refs}) < 2:
        return {
            "error": "D must use at least two source siblings.",
            "error_code": "d_single_source",
        }
    proposed = _d_signature(catalog, action_refs)
    for source in ("A", "B", "C"):
        original = [
            ref for ref, entry in catalog.items() if entry["candidate_id"] == source
        ]
        if proposed == _d_signature(catalog, original):
            return {
                "error": f"D is physically identical to Candidate {source}.",
                "error_code": "d_physical_copy",
            }

    base_net, _ = N._setup(profile, N.peak_hour_of(profile))
    try:
        result, batch_evaluation = _evaluate_fusion_as_batch(
            profile, pending, base_net, action_refs, actions
        )
    except Exception as exc:
        return {
            "error": f"Fusion batch verification failed: {type(exc).__name__}: {exc}"[
                :600
            ],
            "error_code": "fusion_batch_verification_failed",
            "parent_unchanged": True,
        }

    superiority = _fusion_superiority(result, pending["evaluated"])

    result.update(
        candidate_id="D",
        strategy=str(strategy),
        round_id=round_id,
        source_action_refs=list(action_refs),
        fusion_batch_evaluation=copy.deepcopy(batch_evaluation),
        fusion_superior_to_all_siblings=bool(superiority["superior"]),
        fusion_comparison=copy.deepcopy(superiority["comparisons"]),
    )

    result["improving"] = _feasibility_aware_improvement(
        result,
        pending["state_before"],
        pending["parent_score"],
    )
    result["security5_relation_vs_parent"] = _score_relation(
        result["score"],
        pending["parent_score"],
    )
    result["security5_change_vs_parent"] = _security5_change(
        result["state"],
        pending["state_before"],
    )

    context = copy.deepcopy(pending)
    context["evaluated"].append(result)
    context["fusion_batch_evaluation"] = copy.deepcopy(batch_evaluation)
    context["fusion_superior_to_all_siblings"] = bool(superiority["superior"])
    context["fusion_source_action_refs"] = list(action_refs)

    print(
        f"[fusion-v1-batch] profile={profile} batch_verifications=1 actions={len(actions)} "
        f"superior_to_ABC={int(superiority['superior'])} "
        f"fusion_cost_musd={float(result['cost_musd']):.4f}",
        flush=True,
    )
    final = _complete_candidate_round(context)
    final.update(
        crossover_required=False,
        round_id=round_id,
        crossover_source_action_refs=list(action_refs),
        fusion_superior_to_all_siblings=bool(superiority["superior"]),
        fusion_comparison=copy.deepcopy(superiority["comparisons"]),
        fusion_batch_evaluation=copy.deepcopy(batch_evaluation),
        fusion_source_action_refs=list(action_refs),
        fusion_ineligible_reason=(
            None
            if superiority["superior"]
            else "D did not strictly outperform all A/B/C siblings; D cannot be committed."
        ),
    )
    return final


def recover_pending_crossover(
    profile: str, reason: str = "model_output_exhausted"
) -> dict:
    pending = _PENDING_CANDIDATE_SELECTION
    if (
        not pending
        or pending.get("phase") != "crossover"
        or pending["profile"] != profile
    ):
        return {"error": "No crossover round available for recovery"}
    if S.PLAN["key"] != profile or S.PLAN["actions"] != pending["previous"]:
        return {"error": "Retained parent changed: refusing stale crossover recovery"}
    ctx = copy.deepcopy(pending)
    ctx["crossover_skipped"] = True
    ctx["crossover_skip_reason"] = reason
    result = _complete_candidate_round(ctx)
    result.update(
        crossover_skipped=True, crossover_skip_reason=reason, degraded_round=True
    )
    return result


def pending_round_tool():
    pending = _PENDING_CANDIDATE_SELECTION

    if pending is None:
        return None

    if pending.get("phase") == "crossover":
        return "test_crossover_plan"

    return "commit_candidate"


def reset_pending_round():
    global _PENDING_CANDIDATE_SELECTION, _PENDING_SEQUENTIAL_CANDIDATES
    _PENDING_CANDIDATE_SELECTION = None
    _PENDING_SEQUENTIAL_CANDIDATES = None
