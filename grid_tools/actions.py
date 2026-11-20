from __future__ import annotations

import copy
import json
import math

import numpy as np
import pandapower as pp

from . import config as C
from . import network as N

_EPS = 1e-9


def _signed_support_from_legacy(a: dict) -> float:
    mvar = abs(float(a.get("mvar", 0.0)))
    return -mvar if str(a.get("mode", "raise")).lower() == "lower" else mvar


def canonicalize_action(a: dict) -> dict:
    if not isinstance(a, dict):
        return {"action": "invalid", "raw": a}

    kind = str(a.get("action", "")).strip()

    if kind == "upgrade_line":
        target = a.get("target_capacity_mva", a.get("capacity_mva"))
        return {
            "action": "upgrade_line",
            "line": a.get("line"),
            "target_capacity_mva": target,
        }

    if kind == "upgrade_transformer":
        target = a.get("target_sn_mva", a.get("sn_mva"))
        return {
            "action": "upgrade_transformer",
            "transformer": a.get("transformer"),
            "target_sn_mva": target,
        }

    if kind == "new_line":
        return {
            "action": "new_line",
            "from_bus": a.get("from_bus"),
            "to_bus": a.get("to_bus"),
            "capacity_mva": a.get("capacity_mva", 200.0),
        }

    if kind == "set_bus_reactive_support":
        return {
            "action": "set_bus_reactive_support",
            "bus": a.get("bus"),
            "target_mvar": a.get("target_mvar", 0.0),
        }

    if kind == "upgrade_bus":
        return {
            "action": "set_bus_reactive_support",
            "bus": a.get("bus"),
            "target_mvar": _signed_support_from_legacy(a),
        }

    return dict(a)


def _dedupe(actions: list[dict]) -> tuple[list[dict], list[str]]:
    keyed: dict[tuple, dict] = {}
    order: list[tuple] = []
    notes: list[str] = []

    for raw in actions or []:
        a = canonicalize_action(raw)
        kind = a.get("action")

        try:
            if kind == "upgrade_line":
                key = ("line", str(a.get("line")))
            elif kind == "upgrade_transformer":
                key = ("transformer", str(a.get("transformer")))
            elif kind == "set_bus_reactive_support":
                key = ("bus_support", N._busnum(a.get("bus")))
                a["bus"] = key[1]
            elif kind == "new_line":
                fb, tb = N._busnum(a.get("from_bus")), N._busnum(a.get("to_bus"))
                a["from_bus"], a["to_bus"] = fb, tb
                key = ("new_line", min(fb, tb), max(fb, tb))
            else:
                key = ("other", json.dumps(a, sort_keys=True, default=str))
        except Exception:
            key = ("other", json.dumps(a, sort_keys=True, default=str))

        if key in keyed:
            notes.append(
                f"plan target replaced: {describe_action(keyed[key])} -> {describe_action(a)}"
            )
            keyed[key] = a
        else:
            keyed[key] = a
            order.append(key)

    return [keyed[k] for k in order], notes


def _planning_shunt_name(bus_number: int) -> str:
    return f"PLANNING_SHUNT_Bus{bus_number}"


def _current_bus_support_mvar(net, bus_number: int) -> float:
    name = _planning_shunt_name(bus_number)
    hits = net.shunt.index[net.shunt["name"] == name].tolist() if len(net.shunt) else []
    if not hits:
        return 0.0

    return -float(net.shunt.at[hits[0], "q_mvar"])


def _planning_new_line_name(from_bus: int, to_bus: int) -> str:
    a, b = sorted((from_bus, to_bus))
    return f"PLANNING_NEW_Bus{a}_Bus{b}"


def _find_planning_new_line(net, from_bus: int, to_bus: int):
    name = _planning_new_line_name(from_bus, to_bus)
    hits = net.line.index[net.line["name"] == name].tolist()
    return int(hits[0]) if hits else None


def _applicable(net, actions: list[dict]) -> list[dict]:
    """Keep only plan entries that change an untouched baseline network."""
    trial = copy.deepcopy(net)
    keep = []
    for a in actions or []:
        _, done, _ = _apply(trial, [a])
        if done:
            keep.append(canonicalize_action(a))
    return keep


def _apply(net, actions: list[dict]) -> tuple[float, list[str], list[str]]:
    actions, _ = _dedupe(actions)
    cost = 0.0
    done: list[str] = []
    notes: list[str] = []

    for a in actions:
        kind = a.get("action")
        try:
            if kind == "upgrade_line":
                i = N._find(net, a.get("line"))
                old = float(net.line.at[i, "capacity_mva"])
                base = (
                    float(net.line.at[i, "base_capacity_mva"])
                    if "base_capacity_mva" in net.line.columns
                    else old
                )
                target = float(a.get("target_capacity_mva"))

                if target < base - _EPS:
                    notes.append(
                        f"{N._name(net, i)}: target rating {target:.0f} MVA is below "
                        f"the original rating {base:.0f} MVA; action is invalid."
                    )
                    continue
                if math.isclose(target, old, abs_tol=_EPS):
                    notes.append(
                        f"{N._name(net, i)}: target rating {target:.0f} MVA equals "
                        f"the current planned rating; action has no effect."
                    )
                    continue

                ratio = target / old
                net.line.at[i, "r_ohm_per_km"] = (
                    float(net.line.at[i, "r_ohm_per_km"]) / ratio
                )
                net.line.at[i, "x_ohm_per_km"] = (
                    float(net.line.at[i, "x_ohm_per_km"]) / ratio**0.3
                )
                net.line.at[i, "max_i_ka"] = float(net.line.at[i, "max_i_ka"]) * ratio
                net.line.at[i, "capacity_mva"] = target
                done.append(
                    f"set {N._name(net, i)} rating: {old:.0f} -> {target:.0f} MVA"
                )

                if target > base + _EPS:
                    cost += C.COST_LINE_FIXED + C.COST_LINE_PER_MVA * (target - base)

            elif kind == "upgrade_transformer":
                ref = str(a.get("transformer", "")).strip()
                i = (
                    int(ref.rsplit("#T", 1)[1])
                    if "#T" in ref
                    else int(net.trafo.index[net.trafo["name"] == ref][0])
                )
                old = float(net.trafo.at[i, "sn_mva"])
                base = (
                    float(net.trafo.at[i, "base_sn_mva"])
                    if "base_sn_mva" in net.trafo.columns
                    else old
                )
                target = float(a.get("target_sn_mva"))

                if target < base - _EPS:
                    notes.append(
                        f"{N._tname(net, i)}: target rating {target:.0f} MVA is below "
                        f"the original rating {base:.0f} MVA; action is invalid."
                    )
                    continue
                if math.isclose(target, old, abs_tol=_EPS):
                    notes.append(
                        f"{N._tname(net, i)}: target rating {target:.0f} MVA equals "
                        f"the current planned rating; action has no effect."
                    )
                    continue

                ratio = target / old
                net.trafo.at[i, "vk_percent"] = (
                    float(net.trafo.at[i, "vk_percent"]) * ratio
                )
                net.trafo.at[i, "vkr_percent"] = (
                    float(net.trafo.at[i, "vkr_percent"]) * ratio
                )
                net.trafo.at[i, "sn_mva"] = target
                done.append(
                    f"set {N._tname(net, i)} rating: {old:.0f} -> {target:.0f} MVA"
                )

                if target > base + _EPS:
                    cost += C.COST_TRAFO_FIXED + C.COST_TRAFO_PER_MVA * (target - base)

            elif kind == "new_line":
                fb = N._busnum(a["from_bus"])
                tb = N._busnum(a["to_bus"])
                fb_idx, tb_idx = fb - 1, tb - 1
                if fb_idx not in net.bus.index or tb_idx not in net.bus.index:
                    notes.append(f"Bus{fb} or Bus{tb} does not exist")
                    continue

                target = float(a.get("capacity_mva", 200.0))
                existing = _find_planning_new_line(net, fb, tb)
                old = (
                    float(net.line.at[existing, "capacity_mva"])
                    if existing is not None
                    else 0.0
                )

                if target < -_EPS:
                    notes.append(
                        f"new line Bus{fb}-Bus{tb}: capacity {target:.0f} MVA is invalid."
                    )
                    continue
                if math.isclose(target, old, abs_tol=_EPS):
                    notes.append(
                        f"new line Bus{fb}-Bus{tb}: target capacity {target:.0f} MVA "
                        f"equals the current planned capacity; action has no effect."
                    )
                    continue

                if target <= _EPS:
                    if existing is not None:
                        net.line.drop(index=existing, inplace=True)
                        done.append(f"remove planned new line Bus{fb}-Bus{tb}")
                    continue

                if existing is not None:
                    vn = float(net.bus.at[fb_idx, "vn_kv"])
                    net.line.at[existing, "max_i_ka"] = target / (np.sqrt(3) * vn)
                    net.line.at[existing, "capacity_mva"] = target
                    done.append(
                        f"set planned new line Bus{fb}-Bus{tb}: {old:.0f} -> {target:.0f} MVA"
                    )
                else:
                    same = net.line[
                        ((net.line.from_bus == fb_idx) & (net.line.to_bus == tb_idx))
                        | ((net.line.from_bus == tb_idx) & (net.line.to_bus == fb_idx))
                    ]
                    vn = float(net.bus.at[fb_idx, "vn_kv"])
                    if len(same):
                        src = same.index[0]
                        r = float(net.line.at[src, "r_ohm_per_km"])
                        x = float(net.line.at[src, "x_ohm_per_km"])
                    else:
                        r = 3.0 * (vn / 138.0) ** 2
                        x = 12.0 * (vn / 138.0) ** 2

                    i = pp.create_line_from_parameters(
                        net,
                        from_bus=fb_idx,
                        to_bus=tb_idx,
                        length_km=1.0,
                        r_ohm_per_km=r,
                        x_ohm_per_km=x,
                        c_nf_per_km=0.0,
                        max_i_ka=target / (np.sqrt(3) * vn),
                        name=_planning_new_line_name(fb, tb),
                    )
                    net.line.at[i, "capacity_mva"] = target
                    net.line.at[i, "base_capacity_mva"] = 0.0
                    done.append(f"add new line Bus{fb}-Bus{tb} at {target:.0f} MVA")

                cost += C.COST_NEW_FIXED + C.COST_NEW_PER_MVA * target

            elif kind == "set_bus_reactive_support":
                bus_number = N._busnum(a["bus"])
                b = bus_number - 1
                if b not in net.bus.index:
                    notes.append(f"Bus{bus_number} does not exist")
                    continue

                target = float(a.get("target_mvar", 0.0))
                old = _current_bus_support_mvar(net, bus_number)
                if math.isclose(target, old, abs_tol=_EPS):
                    notes.append(
                        f"Bus{bus_number}: target reactive support {target:+.0f} Mvar "
                        f"equals the current planned target; action has no effect."
                    )
                    continue

                name = _planning_shunt_name(bus_number)
                hits = (
                    net.shunt.index[net.shunt["name"] == name].tolist()
                    if len(net.shunt)
                    else []
                )
                if abs(target) <= _EPS:
                    if hits:
                        net.shunt.drop(index=hits, inplace=True)
                    done.append(
                        f"set Bus{bus_number} reactive support: {old:+.0f} -> 0 Mvar"
                    )
                else:
                    q = -target
                    if hits:
                        net.shunt.at[hits[0], "q_mvar"] = q
                    else:
                        pp.create_shunt(
                            net,
                            bus=b,
                            q_mvar=q,
                            p_mw=0.0,
                            name=name,
                        )
                    done.append(
                        f"set Bus{bus_number} reactive support: {old:+.0f} -> {target:+.0f} Mvar"
                    )
                    cost += C.COST_BUS_FIXED + C.COST_BUS_PER_MVAR * abs(target)

            else:
                notes.append(
                    f"unknown action '{kind}' -- use upgrade_line, upgrade_transformer, "
                    f"new_line, or set_bus_reactive_support"
                )

        except Exception as e:
            notes.append(f"{describe_action(a)}: {e}")

    return float(cost), done, notes


def record_action_transitions(net, actions: list[dict]) -> tuple[list[dict], list[str]]:
    trial = copy.deepcopy(net)
    actions, _ = _dedupe(actions)
    records: list[dict] = []
    notes: list[str] = []

    for a in actions:
        kind = a.get("action")
        rec: dict = {"action": kind, "simulator_applied": False}

        try:
            if kind == "upgrade_line":
                i = N._find(trial, a.get("line"))
                old = float(trial.line.at[i, "capacity_mva"])
                base = (
                    float(trial.line.at[i, "base_capacity_mva"])
                    if "base_capacity_mva" in trial.line.columns
                    else old
                )
                target = float(a.get("target_capacity_mva"))
                valid = target >= base - _EPS and not math.isclose(
                    target, old, abs_tol=_EPS
                )
                actual_to = target if valid else old
                rec.update(
                    {
                        "line": N._name(trial, i),
                        "from_capacity_mva": old,
                        "to_capacity_mva": actual_to,
                        "capacity_change_mva": actual_to - old,
                        "simulator_applied": valid,
                    }
                )

            elif kind == "upgrade_transformer":
                ref = str(a.get("transformer", "")).strip()
                i = (
                    int(ref.rsplit("#T", 1)[1])
                    if "#T" in ref
                    else int(trial.trafo.index[trial.trafo["name"] == ref][0])
                )
                old = float(trial.trafo.at[i, "sn_mva"])
                base = (
                    float(trial.trafo.at[i, "base_sn_mva"])
                    if "base_sn_mva" in trial.trafo.columns
                    else old
                )
                target = float(a.get("target_sn_mva"))
                valid = target >= base - _EPS and not math.isclose(
                    target, old, abs_tol=_EPS
                )
                actual_to = target if valid else old
                rec.update(
                    {
                        "transformer": N._tname(trial, i),
                        "from_sn_mva": old,
                        "to_sn_mva": actual_to,
                        "sn_mva_change": actual_to - old,
                        "simulator_applied": valid,
                    }
                )

            elif kind == "new_line":
                fb, tb = N._busnum(a["from_bus"]), N._busnum(a["to_bus"])
                existing = _find_planning_new_line(trial, fb, tb)
                old = (
                    float(trial.line.at[existing, "capacity_mva"])
                    if existing is not None
                    else 0.0
                )
                target = float(a.get("capacity_mva", 200.0))
                valid = target >= -_EPS and not math.isclose(target, old, abs_tol=_EPS)
                actual_to = max(target, 0.0) if valid else old
                rec.update(
                    {
                        "from_bus": fb,
                        "to_bus": tb,
                        "from_capacity_mva": old,
                        "to_capacity_mva": actual_to,
                        "capacity_change_mva": actual_to - old,
                        "simulator_applied": valid,
                    }
                )

            elif kind == "set_bus_reactive_support":
                bus_number = N._busnum(a["bus"])
                old = _current_bus_support_mvar(trial, bus_number)
                target = float(a.get("target_mvar", 0.0))
                valid = not math.isclose(target, old, abs_tol=_EPS)
                actual_to = target if valid else old
                rec.update(
                    {
                        "bus": f"Bus{bus_number}",
                        "from_reactive_support_mvar": old,
                        "to_reactive_support_mvar": actual_to,
                        "reactive_support_change_mvar": actual_to - old,
                        "simulator_applied": valid,
                    }
                )

            else:
                rec.update({"raw_action": a})

            _, applied_descriptions, action_notes = _apply(trial, [a])
            if not applied_descriptions:
                rec["simulator_applied"] = False
            if action_notes:
                notes.extend(action_notes)

        except Exception as e:
            rec["simulator_applied"] = False
            notes.append(f"{describe_action(a)}: {e}")

        records.append(rec)

    return records, notes


def describe_action(a: dict) -> str:
    a = canonicalize_action(a)
    kind = a.get("action")

    if kind == "upgrade_line":
        return f"set {a.get('line')} rating -> {a.get('target_capacity_mva')} MVA"
    if kind == "upgrade_transformer":
        return f"set {a.get('transformer')} rating -> {a.get('target_sn_mva')} MVA"
    if kind == "new_line":
        return (
            f"new line Bus{a.get('from_bus')}-Bus{a.get('to_bus')} "
            f"at {a.get('capacity_mva', 200)} MVA"
        )
    if kind == "set_bus_reactive_support":
        try:
            bus = N._busnum(a.get("bus"))
        except Exception:
            bus = a.get("bus")
        target = a.get("target_mvar", 0)
        try:
            target_text = f"{float(target):+.0f} Mvar"
        except Exception:
            target_text = f"{target} Mvar"
        return f"set Bus{bus} reactive support -> {target_text}"
    return str(a)
