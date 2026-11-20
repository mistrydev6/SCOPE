from __future__ import annotations

import copy
import math

import pandapower as pp

from . import actions as A
from . import config as C
from . import network as N


def _rms(values) -> float:
    finite = [float(x) for x in values if math.isfinite(float(x))]
    if not finite:
        return 0.0
    return math.sqrt(sum(x * x for x in finite) / len(finite))


def _security5_record(
    base_thermal: float,
    base_voltage: float,
    n1_thermal: float,
    n1_voltage: float,
    n1_failure_fraction: float,
) -> dict:

    vector = [
        round(float(base_thermal), 4),
        round(float(base_voltage), 4),
        round(float(n1_thermal), 4),
        round(float(n1_voltage), 4),
        round(float(n1_failure_fraction), 4),
    ]
    names = (
        "base_thermal_severity",
        "base_voltage_severity",
        "n1_thermal_severity",
        "n1_voltage_severity",
        "n1_failure_fraction",
    )
    return {
        "vector": vector,
        "features": {name: vector[i] for i, name in enumerate(names)},
    }


def _measure_day(profile: str, plan: list[dict] | None = None) -> dict:
    """Return neutral simulator observations for the full 24-hour profile."""
    plan = plan or []
    peak = N.peak_hour_of(profile)

    worst_loading: dict[str, dict] = {}
    lowest_voltage: dict[str, float] = {}
    highest_voltage: dict[str, float] = {}
    problem_hours: list[dict] = []
    losses_mwh = 0.0
    nonconverged_hours: list[int] = []
    base_thermal_excess: list[float] = []
    base_voltage_excess: list[float] = []

    for hour in range(24):
        base_net, hour_info = N._setup(profile, hour)
        trial = copy.deepcopy(base_net)
        if plan:
            A._apply(trial, plan)

        try:
            pp.runpp(trial, numba=False)
        except Exception:
            nonconverged_hours.append(hour)
            continue

        line_loading = trial.res_line["loading_percent"]
        trafo_loading = trial.res_trafo["loading_percent"]
        vm = trial.res_bus["vm_pu"]
        losses_mwh += float(trial.res_line["pl_mw"].sum())

        for value in line_loading.tolist() + trafo_loading.tolist():
            fv = float(value)
            if math.isfinite(fv):
                base_thermal_excess.append(max(fv / float(C.THERMAL_MAX) - 1.0, 0.0))

        for value in vm.tolist():
            fv = float(value)
            if not math.isfinite(fv):
                continue
            low = max((float(C.V_MIN) - fv) / float(C.V_MIN), 0.0)
            high = max((fv - float(C.V_MAX)) / float(C.V_MAX), 0.0)
            base_voltage_excess.append(max(low, high))

        for i, value in line_loading.items():
            if value <= C.THERMAL_MAX:
                continue
            name = N._name(trial, i)
            item = {
                "element": name,
                "element_type": "line",
                "loading_pct": round(float(value), 1),
                "rating_mva": round(float(trial.line.at[i, "capacity_mva"])),
                "hour": hour,
            }
            if name not in worst_loading or value > worst_loading[name]["loading_pct"]:
                worst_loading[name] = item

        for i, value in trafo_loading.items():
            if value <= C.THERMAL_MAX:
                continue
            name = N._tname(trial, i)
            item = {
                "element": name,
                "element_type": "transformer",
                "loading_pct": round(float(value), 1),
                "rating_mva": round(float(trial.trafo.at[i, "sn_mva"])),
                "hour": hour,
            }
            if name not in worst_loading or value > worst_loading[name]["loading_pct"]:
                worst_loading[name] = item

        for i, value in vm.items():
            bus = f"Bus{int(i) + 1}"
            fv = float(value)
            lowest_voltage[bus] = min(lowest_voltage.get(bus, fv), fv)
            highest_voltage[bus] = max(highest_voltage.get(bus, fv), fv)

        n_overloaded = int((line_loading > C.THERMAL_MAX).sum()) + int(
            (trafo_loading > C.THERMAL_MAX).sum()
        )
        n_voltage_problem = int((vm < C.V_MIN).sum()) + int((vm > C.V_MAX).sum())

        if n_overloaded or n_voltage_problem:
            problem_hours.append(
                {
                    "hour": hour,
                    "total_load_p_mw": round(float(hour_info["total_load_p_mw"]), 1),
                    "total_renewable_p_mw": round(
                        float(hour_info["total_renewable_p_mw"]), 1
                    ),
                    "net_load_p_mw": round(float(hour_info["net_load_p_mw"]), 1),
                    "n_overloaded_elements": n_overloaded,
                    "n_voltage_problem_buses": n_voltage_problem,
                    "max_loading_pct": round(
                        max(float(line_loading.max()), float(trafo_loading.max())), 1
                    ),
                    "min_vm_pu": round(float(vm.min()), 3),
                    "max_vm_pu": round(float(vm.max()), 3),
                }
            )

    base_case_voltage_problem_buses = []
    for bus in sorted(set(lowest_voltage) | set(highest_voltage)):
        lo = lowest_voltage.get(bus, 1.0)
        hi = highest_voltage.get(bus, 1.0)
        if lo >= C.V_MIN and hi <= C.V_MAX:
            continue
        problem = (
            "both_low_and_high"
            if lo < C.V_MIN and hi > C.V_MAX
            else "low" if lo < C.V_MIN else "high"
        )
        base_case_voltage_problem_buses.append(
            {
                "bus": bus,
                "min_vm_pu": round(lo, 3),
                "max_vm_pu": round(hi, 3),
                "problem": problem,
            }
        )

    base_case_voltage_problem_buses.sort(
        key=lambda d: (
            d["problem"] != "both_low_and_high",
            (
                -(C.V_MIN - d["min_vm_pu"])
                if d["min_vm_pu"] < C.V_MIN
                else -(d["max_vm_pu"] - C.V_MAX)
            ),
        )
    )

    peak_base_net, _ = N._setup(profile, peak)
    peak_trial = copy.deepcopy(peak_base_net)
    if plan:
        A._apply(peak_trial, plan)
    n1 = _screen_n1(peak_trial, peak)

    base_case_overloaded_elements = sorted(
        worst_loading.values(), key=lambda d: -d["loading_pct"]
    )

    security5 = _security5_record(
        _rms(base_thermal_excess),
        _rms(base_voltage_excess),
        n1["thermal_severity"],
        n1["voltage_severity"],
        n1["failure_fraction"],
    )

    return {
        "profile": profile,
        "scenario_family": str(
            N.load_scenario(profile).get("meta", {}).get("subcategory", "")
        ),
        "peak_hour_basis": "maximum_total_load_mw",
        "peak_hour": peak,
        "hours_checked": 24,
        "security5": security5,
        "n_base_case_problem_hours": len(problem_hours),
        "losses_mwh_day": round(losses_mwh, 1),
        "nonconverged_base_case_hours": nonconverged_hours,
        "n_failed_n1_outage_tests": int(n1["n_failed_outage_tests"]),
        "n_n1_outage_tests": int(n1["n_outage_tests"]),
        "base_case_overloaded_elements": base_case_overloaded_elements[: C.LIST_LIMIT],
        "base_case_voltage_problem_buses": base_case_voltage_problem_buses[
            : C.LIST_LIMIT
        ],
        "n1_overloaded_elements": n1["overloaded_elements"][: C.LIST_LIMIT],
        "n1_voltage_problem_buses": n1["voltage_problem_buses"][:12],
        "base_case_worst_hours": sorted(
            problem_hours,
            key=lambda d: -(d["n_overloaded_elements"] + d["n_voltage_problem_buses"]),
        )[:5],
    }


def _screen_n1(net, hour: int) -> dict:
    """N-1 over every in-service branch. Only violations CAUSED by the outage."""
    pp.runpp(net, numba=False)

    targets = [
        ("line", int(i)) for i in net.line.index if net.line.at[i, "in_service"]
    ] + [("trafo", int(i)) for i in net.trafo.index if net.trafo.at[i, "in_service"]]

    n_bad = 0
    weak = {}
    vweak = {}
    thermal_excess: list[float] = []
    voltage_excess: list[float] = []

    for kind, i in targets:
        tbl = net.line if kind == "line" else net.trafo
        tbl.at[i, "in_service"] = False

        try:
            pp.runpp(net, numba=False)

            p2 = net.res_line["loading_percent"]
            t2 = net.res_trafo["loading_percent"]

            for j, value in p2.items():
                if kind == "line" and int(j) == i:
                    continue
                fv = float(value)
                if math.isfinite(fv):
                    thermal_excess.append(max(fv / float(C.THERMAL_MAX_N1) - 1.0, 0.0))
            for j, value in t2.items():
                if kind == "trafo" and int(j) == i:
                    continue
                fv = float(value)
                if math.isfinite(fv):
                    thermal_excess.append(max(fv / float(C.THERMAL_MAX_N1) - 1.0, 0.0))

            new = [
                ("line", j)
                for j in p2[p2 > C.THERMAL_MAX_N1].index
                if not (kind == "line" and j == i)
            ]

            new += [
                ("trafo", j)
                for j in t2[t2 > C.THERMAL_MAX_N1].index
                if not (kind == "trafo" and j == i)
            ]

            finite_vm = [
                float(value)
                for value in net.res_bus["vm_pu"].tolist()
                if math.isfinite(float(value))
            ]
            for fv in finite_vm:
                voltage_excess.append(
                    max(
                        (float(C.V_MIN_N1) - fv) / float(C.V_MIN_N1),
                        0.0,
                    )
                )
            vmin = min(finite_vm) if finite_vm else float("nan")

            if new or vmin < C.V_MIN_N1:
                n_bad += 1

                outage = N._name(net, i) if kind == "line" else N._tname(net, i)

                for k, j in new:
                    nm = N._name(net, j) if k == "line" else N._tname(net, j)

                    cap = (
                        float(net.line.at[j, "capacity_mva"])
                        if k == "line"
                        else float(net.trafo.at[j, "sn_mva"])
                    )

                    load = float(p2[j]) if k == "line" else float(t2[j])

                    w = weak.setdefault(
                        nm,
                        {
                            "element": nm,
                            "overloaded_in_n_outage_tests": 0,
                            "worst_loading_pct": 0.0,
                            "capacity_mva": round(cap),
                        },
                    )

                    w["overloaded_in_n_outage_tests"] += 1

                    w["worst_loading_pct"] = round(
                        max(
                            w["worst_loading_pct"],
                            load,
                        ),
                        1,
                    )

                if not new:
                    b = int(net.res_bus["vm_pu"].idxmin())
                    bn = f"Bus{b + 1}"

                    v = vweak.setdefault(
                        bn,
                        {
                            "bus": bn,
                            "voltage_problem_in_n_outage_tests": 0,
                            "worst_vm_pu": 1.0,
                            "outages": [],
                        },
                    )

                    v["voltage_problem_in_n_outage_tests"] += 1

                    v["worst_vm_pu"] = round(
                        min(
                            v["worst_vm_pu"],
                            vmin,
                        ),
                        3,
                    )

                    if len(v["outages"]) < 3:
                        v["outages"].append(outage)

        except Exception:
            n_bad += 1

        finally:
            tbl.at[i, "in_service"] = True

    pp.runpp(net, numba=False)

    return {
        "n_failed_outage_tests": n_bad,
        "n_outage_tests": len(targets),
        "thermal_severity": _rms(thermal_excess),
        "voltage_severity": _rms(voltage_excess),
        "failure_fraction": (float(n_bad) / len(targets)) if targets else 0.0,
        "overloaded_elements": sorted(
            weak.values(),
            key=lambda d: -d["overloaded_in_n_outage_tests"],
        ),
        "voltage_problem_buses": sorted(
            vweak.values(),
            key=lambda d: -d["voltage_problem_in_n_outage_tests"],
        ),
    }
