from __future__ import annotations


from .. import config as C
from .. import network as N


def scan_day(profile: str = "normal_001") -> dict:
    rows = []
    for hour in range(24):
        net, hour_info = N._setup(
            profile,
            hour,
        )

        line_loading = net.res_line["loading_percent"]
        trafo_loading = net.res_trafo["loading_percent"]
        vm = net.res_bus["vm_pu"]

        rows.append(
            {
                "hour": hour,
                "total_load_p_mw": round(float(hour_info["total_load_p_mw"]), 1),
                "total_renewable_p_mw": round(
                    float(hour_info["total_renewable_p_mw"]), 1
                ),
                "net_load_p_mw": round(float(hour_info["net_load_p_mw"]), 1),
                "n_overloaded_elements": int((line_loading > C.THERMAL_MAX).sum())
                + int((trafo_loading > C.THERMAL_MAX).sum()),
                "max_loading_pct": round(
                    max(float(line_loading.max()), float(trafo_loading.max())), 1
                ),
                "n_voltage_problem_buses": int((vm < C.V_MIN).sum())
                + int((vm > C.V_MAX).sum()),
                "min_vm_pu": round(float(vm.min()), 3),
                "max_vm_pu": round(float(vm.max()), 3),
            }
        )

    meta = N.load_scenario(profile).get("meta", {})

    return {
        "profile": profile,
        "scenario_family": meta.get("subcategory"),
        "peak_hour": N.peak_hour_of(profile),
        "peak_hour_basis": "maximum_total_load_mw",
        "n_base_case_problem_hours": sum(
            1
            for row in rows
            if row["n_overloaded_elements"] or row["n_voltage_problem_buses"]
        ),
        "hourly": rows,
        "note": (
            "This tool contains base-case measurements only. "
            "Call analyse for whole-day base-case problems and peak-hour N-1 outage tests."
        ),
    }
