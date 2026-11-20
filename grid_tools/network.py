from __future__ import annotations
import json
import numpy as np
import pandapower as pp
from . import config as C
from . import state as S

import copy
from pathlib import Path


def _b(x) -> int:
    return int(str(x).strip().lower().replace("bus", "")) - 1


def _busnum(x) -> int:
    import re

    if isinstance(x, str):
        match = re.fullmatch(r"(?:bus[ _-]*)?([0-9]+)", x.strip(), re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid bus identifier {x!r}: use 94 or Bus94")
        number = int(match.group(1))
    else:
        number = int(x)
    if number <= 0:
        raise ValueError(f"Bus index must be positive: {x!r}")
    return number


def _scenario_index() -> dict[str, Path]:
    if not C.SCENARIO_RECORDS.exists():
        raise FileNotFoundError(f"Scenario directory not found: {C.SCENARIO_RECORDS}")

    index = {p.stem: p for p in sorted(C.SCENARIO_RECORDS.glob("*.json"))}

    if not index:
        raise FileNotFoundError(
            f"No scenario JSON records found in {C.SCENARIO_RECORDS}"
        )

    return index


def load_grid():
    if S.NET is not None and S.SCENARIOS is not None:
        return S.NET, S.SCENARIOS

    d = json.loads((C.DATA / C.TOPOLOGY).read_text())
    mapping = _load_mapping()
    net = pp.create_empty_network(sn_mva=100.0)

    for b in sorted(d["buses"], key=lambda x: x["bus_idx"]):
        pp.create_bus(
            net, vn_kv=float(b["vn_kv"]), index=int(b["bus_idx"]), name=b["id"]
        )
    for ld in d["loads"]:
        pp.create_load(
            net,
            bus=_b(ld["bus_id"]),
            p_mw=float(ld["p_mw"]),
            q_mvar=float(ld["q_mvar"]),
            name=ld["id"],
        )

    scenario_gen_map = []

    gen_base_p = []
    gen_max_p = []
    gen_min_p = []
    gen_vm = []
    gen_bus = []

    for g in d["generators"]:
        bus = int(g.get("bus_idx", _b(g["bus_id"])))
        vm = float(g.get("vm_pu", 1.0))

        gen_base_p.append(float(g["p_mw"]))
        gen_max_p.append(float(g["max_p_mw"]))
        gen_min_p.append(float(g["min_p_mw"]))
        gen_vm.append(vm)
        gen_bus.append(bus)

        if g["type"] == "slack":
            i = pp.create_ext_grid(
                net,
                bus=bus,
                vm_pu=vm,
                name=g["id"],
            )
            scenario_gen_map.append(("ext_grid", int(i)))

        else:
            i = pp.create_gen(
                net,
                bus=bus,
                p_mw=float(g["p_mw"]),
                vm_pu=vm,
                max_p_mw=float(g["max_p_mw"]),
                min_p_mw=float(g["min_p_mw"]),
                name=g["id"],
            )
            scenario_gen_map.append(("gen", int(i)))

    for ln in d["lines"]:

        i = pp.create_line_from_parameters(
            net,
            from_bus=_b(ln["from_bus"]),
            to_bus=_b(ln["to_bus"]),
            length_km=1.0,
            r_ohm_per_km=float(ln["r_ohm"]),
            x_ohm_per_km=float(ln["x_ohm"]),
            c_nf_per_km=0.0,
            max_i_ka=float(ln["max_i_ka"]),
            name=ln["id"],
        )
        rating = float(ln["capacity_mva"])
        net.line.at[i, "capacity_mva"] = rating
        net.line.at[i, "base_capacity_mva"] = rating

    for t in d["transformers"]:
        hv, lv = _b(t["from_bus"]), _b(t["to_bus"])
        pp.create_transformer_from_parameters(
            net,
            hv_bus=hv,
            lv_bus=lv,
            sn_mva=float(t["sn_mva"]),
            vn_hv_kv=float(net.bus.at[hv, "vn_kv"]),
            vn_lv_kv=float(net.bus.at[lv, "vn_kv"]),
            vk_percent=float(t["vk_percent"]),
            vkr_percent=float(t["vkr_percent"]),
            pfe_kw=0.0,
            i0_percent=0.0,
            name=t["id"],
        )
        i = int(net.trafo.index[-1])
        net.trafo.at[i, "base_sn_mva"] = float(t["sn_mva"])

    for sh in d.get("shunts", []):
        pp.create_shunt(
            net,
            bus=int(sh["bus_idx"]),
            q_mvar=float(sh["q_mvar"]),
            p_mw=0.0,
            name=sh["id"],
        )

    ren_buses = mapping.get("renewable_buses_zero_based", [])
    ren_kinds = mapping.get(
        "renewable_kind",
        ["renewable"] * len(ren_buses),
    )

    if len(ren_buses) != len(ren_kinds):
        raise ValueError("renewable bus/kind mapping length mismatch")

    scenario_ren_map = []

    for j, (bus, kind) in enumerate(zip(ren_buses, ren_kinds)):
        i = pp.create_sgen(
            net,
            bus=int(bus),
            p_mw=0.0,
            q_mvar=0.0,
            name=f"ScenarioRenewable_{kind}_{j}",
        )

        scenario_ren_map.append(int(i))

    net["p_nom"] = net.load["p_mw"].values.copy()
    net["q_nom"] = net.load["q_mvar"].values.copy()

    net["scenario_gen_map"] = scenario_gen_map
    net["scenario_gen_base_p"] = np.asarray(gen_base_p, dtype=float)
    net["scenario_gen_max_p"] = np.asarray(gen_max_p, dtype=float)
    net["scenario_gen_min_p"] = np.asarray(gen_min_p, dtype=float)
    net["scenario_gen_vm"] = np.asarray(gen_vm, dtype=float)
    net["scenario_gen_bus"] = np.asarray(gen_bus, dtype=int)

    net["scenario_ren_map"] = scenario_ren_map

    scenarios = _scenario_index()

    S.NET = net
    S.SCENARIOS = scenarios

    return net, scenarios


def load_scenario(profile: str) -> dict:
    _, scenarios = load_grid()

    key = str(profile).removesuffix(".json")

    if key not in scenarios:
        sample = ", ".join(list(sorted(scenarios))[:8])
        raise KeyError(f"Unknown scenario {profile!r}. Examples: {sample}")

    if key not in S.SCENARIO_CACHE:
        S.SCENARIO_CACHE[key] = json.loads(scenarios[key].read_text())

    return S.SCENARIO_CACHE[key]


def _load_mapping() -> dict:
    if not C.SCENARIO_MAPPING.exists():
        raise FileNotFoundError(
            f"Scenario mapping report not found: {C.SCENARIO_MAPPING}"
        )

    mapping = json.loads(C.SCENARIO_MAPPING.read_text())

    if not mapping.get("load_order_exact"):
        raise ValueError("Unsafe scenario mapping: load order does not match topology")

    if not mapping.get("generator_order_exact"):
        raise ValueError(
            "Unsafe scenario mapping: generator order does not match topology"
        )

    if not mapping.get("branch_endpoint_multiset_match"):
        raise ValueError(
            "Unsafe scenario mapping: physical branch endpoint set does not match"
        )

    return mapping


def _configure_scenario_hour(net, record: dict, hour: int) -> dict:
    inputs = record["inputs"]

    p_row = np.asarray(
        inputs["load_p_mw"][hour],
        dtype=float,
    )

    ren_row = np.asarray(
        inputs["renewable_avail_mw"][hour],
        dtype=float,
    )

    avail = np.asarray(
        inputs["gen_availability"][hour],
        dtype=float,
    )

    if len(p_row) != len(net.load):
        raise ValueError(
            f"load vector length {len(p_row)} " f"!= topology loads {len(net.load)}"
        )

    if len(ren_row) != len(net["scenario_ren_map"]):
        raise ValueError(
            f"renewable vector length {len(ren_row)} "
            f"!= mapped renewables "
            f"{len(net['scenario_ren_map'])}"
        )

    if len(avail) != len(net["scenario_gen_map"]):
        raise ValueError(
            f"generator availability length {len(avail)} "
            f"!= topology generators "
            f"{len(net['scenario_gen_map'])}"
        )

    ratio = np.divide(
        np.asarray(net["q_nom"], dtype=float),
        np.asarray(net["p_nom"], dtype=float),
        out=np.zeros_like(np.asarray(net["q_nom"], dtype=float)),
        where=np.abs(np.asarray(net["p_nom"], dtype=float)) > 1e-12,
    )

    net.load.loc[:, "p_mw"] = p_row
    net.load.loc[:, "q_mvar"] = p_row * ratio

    for j, pp_idx in enumerate(net["scenario_ren_map"]):
        net.sgen.at[pp_idx, "p_mw"] = float(ren_row[j])
        net.sgen.at[pp_idx, "q_mvar"] = 0.0
        net.sgen.at[pp_idx, "in_service"] = True

    for j, (kind, pp_idx) in enumerate(net["scenario_gen_map"]):
        online = bool(avail[j] > 1e-9)

        if kind == "ext_grid":
            net.ext_grid.at[pp_idx, "in_service"] = online

        else:
            net.gen.at[pp_idx, "in_service"] = online

            if not online:
                net.gen.at[pp_idx, "p_mw"] = 0.0

    total_load = float(p_row.sum())
    total_renewable = float(ren_row.sum())

    net_load = total_load - total_renewable

    conventional_target = max(
        net_load,
        0.0,
    )

    active_nominal_sum = 0.0

    for j, (kind, _) in enumerate(net["scenario_gen_map"]):
        if kind == "gen" and avail[j] > 1e-9:
            active_nominal_sum += max(
                float(net["scenario_gen_base_p"][j]),
                0.0,
            )

    if active_nominal_sum > 1e-12:

        for j, (kind, pp_idx) in enumerate(net["scenario_gen_map"]):

            if kind != "gen" or avail[j] <= 1e-9:
                continue

            nominal = max(
                float(net["scenario_gen_base_p"][j]),
                0.0,
            )

            p = conventional_target * nominal / active_nominal_sum

            p = np.clip(
                p,
                float(net["scenario_gen_min_p"][j]),
                float(net["scenario_gen_max_p"][j]),
            )

            net.gen.at[pp_idx, "p_mw"] = float(p)

    if inputs.get("line_out"):
        raise ValueError(
            "This scenario contains an embedded "
            "transmission outage. "
            "Use the final outage-free 900 dataset."
        )

    if any(
        c.get("element_type") == "line"
        for c in inputs.get(
            "contingencies",
            [],
        )
    ):
        raise ValueError("This scenario contains an embedded " "line contingency.")

    return {
        "total_load_p_mw": total_load,
        "total_renewable_p_mw": total_renewable,
        "net_load_p_mw": net_load,
    }


def _setup(profile: str, hour: int):
    base_net, _ = load_grid()

    record = load_scenario(profile)

    horizon = int(
        record.get(
            "meta",
            {},
        ).get(
            "horizon_steps",
            24,
        )
    )

    if hour < 0 or hour >= horizon:
        raise IndexError(f"hour {hour} outside scenario " f"horizon 0..{horizon - 1}")

    net = copy.deepcopy(base_net)

    hour_info = _configure_scenario_hour(
        net,
        record,
        hour,
    )

    pp.runpp(
        net,
        algorithm="nr",
        init="auto",
        calculate_voltage_angles=True,
        numba=False,
        max_iteration=50,
        tolerance_mva=1e-8,
    )

    return net, hour_info


def scenario_hour_table(
    profile: str,
) -> list[dict]:

    record = load_scenario(profile)

    inputs = record["inputs"]

    horizon = int(
        record.get(
            "meta",
            {},
        ).get(
            "horizon_steps",
            24,
        )
    )

    rows = []

    for hour in range(horizon):

        total_load = float(sum(float(x) for x in inputs["load_p_mw"][hour]))

        total_renewable = float(
            sum(float(x) for x in inputs["renewable_avail_mw"][hour])
        )

        rows.append(
            {
                "hour": hour,
                "total_load_p_mw": total_load,
                "total_renewable_p_mw": total_renewable,
                "net_load_p_mw": total_load - total_renewable,
            }
        )

    return rows


def peak_hour_of(profile: str) -> int:
    rows = scenario_hour_table(profile)

    return int(
        max(
            rows,
            key=lambda x: x["total_load_p_mw"],
        )["hour"]
    )


def _name(net, i) -> str:
    """Parallel circuits share a name, so the index is part of the identity."""
    return f"{net.line.at[i, 'name']}#{i}"


def _tname(net, i) -> str:
    return f"{net.trafo.at[i, 'name']}#T{i}"


def _find(net, ref) -> int:
    ref = str(ref).strip()
    if "#" in ref:
        return int(ref.rsplit("#", 1)[1])
    if ref.isdigit():
        return int(ref)
    hits = net.line.index[net.line["name"] == ref].tolist()
    if not hits:
        raise ValueError(f"no line named '{ref}'")
    return int(hits[0])
