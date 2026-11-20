from __future__ import annotations
import argparse, glob, hashlib, json, math, os, re, statistics
from collections import Counter

from prompts import CONSOLIDATION_EPISODE_SYSTEM, CONSOLIDATION_MERGE_SYSTEM


def sha(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def _events(run_dir):
    out = []
    for line in open(os.path.join(run_dir, "events.jsonl"), errors="replace"):
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def _vec(x):
    if isinstance(x, dict):
        x = x.get("vector")
    return [round(float(v), 4) for v in x] if isinstance(x, list) else None


def load_trajectories(source_memory_dir):
    idx = json.load(open(os.path.join(source_memory_dir, "episode_index.json")))
    eps = {}
    for x in idx["episodes"]:
        e = json.load(open(os.path.join(source_memory_dir, x["file"])))
        eps[e["episode_id"]] = e
    runs = {}

    pool = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(source_memory_dir))),
        "source_pool_memory",
        "evolution",
        "runs",
    )
    for d in glob.glob(os.path.join(source_memory_dir, "evolution", "runs", "*")):
        for cand in (d, os.path.join(pool, os.path.basename(d))):
            try:
                m = json.load(open(os.path.join(cand, "manifest.json")))
                if os.path.getsize(os.path.join(cand, "events.jsonl")) == 0:
                    continue
            except Exception:
                continue
            if m.get("episode_id") in eps:
                runs[m["episode_id"]] = cand
                break
    return {eps[i]["profile"]: (eps[i], runs.get(i)) for i in eps}


def _bus(x):
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, (int, float)) and float(x).is_integer():
        return f"Bus{int(x)}"
    if isinstance(x, str) and x.strip().isdigit():
        return f"Bus{int(x.strip())}"
    return x


def ledger(episode, run_dir):
    steps = episode.get("steps") or []
    sibs, commits = [], []
    if run_dir:
        for ev in _events(run_dir):
            if ev.get("kind") != "tool_result":
                continue
            if ev.get("name") == "test_staged_candidate":
                sibs.append(ev)
            elif ev.get("name") == "commit_candidate":
                commits.append((len(sibs), (ev.get("args") or {}).get("candidate_id")))

    rounds_raw, start = [], 0
    for end, cid in commits:
        rounds_raw.append((sibs[start:end], cid))
        start = end
    if start < len(sibs):
        rounds_raw.append((sibs[start:], None))

    rounds, equipment = [], set()
    initial = episode.get("initial_state") or {}
    for r, (cands, chosen) in enumerate(rounds_raw):
        parent = (steps[r].get("state_before") if r < len(steps) else None) or (
            initial if r == 0 else {}
        )
        needs = line_needs(parent) if parent else {}
        volt = bus_depths(parent) if parent else {}
        equipment |= set(needs) | set(volt)
        rd = {
            "key": f"R{r + 1}",
            "parent_security5": _vec(parent.get("security5")) if parent else None,
            "parent_violated_lines": [
                {
                    "line": l,
                    "worst_loading_pct": round(n["pct"], 1),
                    "binding_limit": "N-1 110%" if n["kind"] == "n1" else "base 100%",
                    "rating_mva": n["rating"],
                    "rating_needed_for_observed_flow_mva": round(n["need"], 1),
                }
                for l, n in sorted(
                    needs.items(), key=lambda kv: -kv[1]["need"] / kv[1]["rating"]
                )
            ],
            "parent_voltage_buses": sorted(volt),
            "candidates": [],
        }
        for ev in cands:
            a = ev.get("args") or {}
            cid = a.get("candidate_id")
            res = ev.get("result") or {}
            pf = res.get("full_plan_pf") or {}
            if not pf.get("security5"):

                for y in res.get("candidate_results") or []:
                    if isinstance(y, dict) and y.get("candidate_id") == cid:
                        pf = y
                        break
            acts = []
            for x in a.get("actions") or []:
                if x.get("action") == "upgrade_line":
                    l = x.get("line")
                    t = x.get("target_capacity_mva")
                    equipment.add(l)
                    item = {"line": l, "target_mva": t}
                    if l in needs and t:
                        item["target_over_needed_ratio"] = round(
                            float(t) / needs[l]["need"], 3
                        )
                    else:
                        item["note"] = "line not violated in parent"
                    acts.append(item)
                elif x.get("action") == "set_bus_reactive_support":
                    b = _bus(x.get("bus"))
                    equipment.add(b)
                    acts.append(
                        {
                            "bus": b,
                            "target_mvar": x.get("target_mvar"),
                            "bus_had_voltage_problem": b in volt,
                        }
                    )
                else:
                    acts.append({k: v for k, v in x.items() if k != "strategy"})
            rd["candidates"].append(
                {
                    "key": f"R{r + 1}.{cid}",
                    "candidate_id": cid,
                    "committed": cid == chosen,
                    "actions": acts,
                    "verified_security5": _vec(pf.get("security5")),
                    **({} if pf.get("security5") else {"not_verified": True}),
                    "solved": pf.get("solved"),
                    "plan_cost_musd": pf.get("plan_cost_musd"),
                    "relation_vs_parent": pf.get("security5_relation_vs_parent"),
                }
            )
        if r < len(steps):
            ch = (steps[r].get("simulator_consequence") or {}).get("change") or {}
            rd["after_commit"] = {
                "resolved_n1_overloads": ch.get("resolved_n1_overloaded_elements")
                or [],
                "new_n1_overloads_caused": ch.get("new_n1_overloaded_elements") or [],
                "new_base_overloads_caused": ch.get("new_base_case_overloaded_elements")
                or [],
                "new_voltage_problems_caused": (
                    ch.get("new_n1_voltage_problem_buses") or []
                )
                + (ch.get("new_base_case_voltage_problem_buses") or []),
            }
            for k in ("new_n1_overloads_caused", "new_base_overloads_caused"):
                equipment |= set(rd["after_commit"][k])
        rounds.append(rd)

    fin = episode.get("final") or {}
    n0 = line_needs(initial)
    final_plan = []
    for x in fin.get("final_plan") or []:
        if x.get("action") == "upgrade_line":
            l = x.get("line")
            t = x.get("to_capacity_mva")
            equipment.add(l)
            item = {
                "line": l,
                "final_mva": t,
                "original_mva": x.get("from_capacity_mva"),
            }
            if l in n0 and t:
                item["final_over_initially_needed_ratio"] = round(
                    float(t) / n0[l]["need"], 3
                )
            else:
                item["note"] = "not violated at start (knock-on or preemptive)"
            final_plan.append(item)
        elif x.get("action") == "set_bus_reactive_support":
            equipment.add(_bus(x.get("bus")))
            final_plan.append(
                {
                    "bus": _bus(x.get("bus")),
                    "mvar": x.get("to_reactive_support_mvar", x.get("target_mvar")),
                }
            )
    out = {
        "profile": episode.get("profile"),
        "episode_id": episode.get("episode_id"),
        "family": str(episode.get("profile", "")).rsplit("_", 1)[0],
        "rounds": rounds,
        "final": {
            "key": "FINAL",
            "solved": fin.get("solved"),
            "total_cost_musd": fin.get("total_cost_musd"),
            "plan": final_plan,
        },
    }
    out["equipment"] = sorted(str(e) for e in equipment if e)
    out["keys"] = sorted(
        {"FINAL"}
        | {r["key"] for r in rounds}
        | {c["key"] for r in rounds for c in r["candidates"]}
    )
    return out


BASE_LIMIT, N1_LIMIT = 100.0, 110.0


V_MIN, V_MIN_N1 = 0.95, 0.92


def line_needs(state: dict) -> dict:
    out = {}
    for it in state.get("base_case_overloaded_elements") or []:
        if it.get("element_type", "line") != "line":
            continue
        pct = it.get("loading_pct", it.get("worst_loading_pct"))
        cap = it.get("rating_mva", it.get("capacity_mva"))
        if pct is None or cap is None:
            continue
        out[it["element"]] = {
            "rating": float(cap),
            "pct": float(pct),
            "kind": "base",
            "need": float(cap) * float(pct) / BASE_LIMIT,
        }
    for it in state.get("n1_overloaded_elements") or []:
        name = it.get("element")
        if not name or not str(name).startswith("Line_"):
            continue
        pct = it.get("worst_loading_pct", it.get("loading_pct"))
        cap = it.get("capacity_mva", it.get("rating_mva"))
        if pct is None or cap is None:
            continue
        need = float(cap) * float(pct) / N1_LIMIT
        if name not in out or need > out[name]["need"]:
            out[name] = {
                "rating": float(cap),
                "pct": float(pct),
                "kind": "n1",
                "need": need,
            }
    return out


def bus_depths(state: dict) -> dict:
    out = {}
    for it in state.get("base_case_voltage_problem_buses") or []:
        vm = it.get("min_vm_pu", it.get("worst_vm_pu"))
        if it.get("bus") and vm is not None and float(vm) < V_MIN:
            out[it["bus"]] = max(out.get(it["bus"], 0.0), V_MIN - float(vm))
    for it in state.get("n1_voltage_problem_buses") or []:
        vm = it.get("worst_vm_pu")
        if it.get("bus") and vm is not None:
            out[it["bus"]] = max(out.get(it["bus"], 0.0), V_MIN_N1 - float(vm), 0.001)
    return out


SCHEMA = "scope-generic-strategies-v1"
BUILDER = "generic-trajectory-v1"
MAX_PER_EPISODE = 2
MAX_MERGED = 8
STAGES = ["first_plan", "residual_repair", "any"]
VIOLATIONS = ["n1_thermal", "base_thermal", "voltage", "mixed", "any"]
SEVERITIES = ["mild", "moderate", "severe", "any"]
EXTENTS = ["few", "several", "many", "any"]


def extent_of(n_violated_lines):
    return (
        "few"
        if n_violated_lines <= 6
        else "several" if n_violated_lines <= 12 else "many"
    )


APPLIES = ["n1_bound", "base_bound", "any", "none"]

REAL_NAME = re.compile(r"Line_Bus\d+_Bus\d+#\d+|(?<![A-Za-z_])Bus\d+(?![_\d])")
LOCAL_LABEL = re.compile(r"\b[LB]\d+\b")
STOCK = re.compile(r"current (physical )?diagnostics to target the residual", re.I)


BANDS = [
    (0.0, 1.00, "<1.00"),
    (1.00, 1.03, "1.00-1.03"),
    (1.03, 1.05, "1.03-1.05"),
    (1.05, 1.07, "1.05-1.07"),
    (1.07, 1.10, "1.07-1.10"),
    (1.10, 1.20, "1.10-1.20"),
    (1.20, 9.0, ">=1.20"),
]


def severity_of(over_limit):
    return (
        "mild" if over_limit < 1.05 else "moderate" if over_limit <= 1.30 else "severe"
    )


SEVS = ["mild", "moderate", "severe"]


def committed_outcomes(episode):
    out = []
    for st in episode.get("steps") or []:
        need = line_needs(st.get("state_before") or {})
        post = line_needs(
            ((st.get("simulator_consequence") or {}).get("attempted_state")) or {}
        )
        rows = []
        for a in st.get("attempted_actions") or []:
            if a.get("action") != "upgrade_line" or a.get("line") not in need:
                continue
            r = float(a["to_capacity_mva"]) / need[a["line"]]["need"]
            row = {
                "line": a["line"],
                "rating_over_needed_ratio": round(r, 3),
                "severity": severity_of(
                    need[a["line"]]["need"] / need[a["line"]]["rating"]
                ),
                "extent": extent_of(len(need)),
                "binding_limit": (
                    "N-1 110%" if need[a["line"]]["kind"] == "n1" else "base 100%"
                ),
                "resolved": a["line"] not in post,
            }
            if a["line"] in post:
                row["still_worst_loading_pct"] = round(post[a["line"]]["pct"], 1)
                row["rating_actually_needed_over_static_need"] = round(
                    post[a["line"]]["need"] / need[a["line"]]["need"], 3
                )
            rows.append(row)
        out.append(rows)
    return out


def anonymise(L):
    m = {}

    def lab(name, kind):
        if name not in m:
            m[name] = (
                f"{'L' if kind == 'line' else 'B'}{1 + sum(1 for v in m.values() if v[0] == ('L' if kind == 'line' else 'B'))}"
            )
        return m[name]

    out = {"family": L["family"], "rounds": [], "final": None}
    for r in L["rounds"]:
        rr = {
            "key": r["key"],
            "parent_security5": r["parent_security5"],
            "n_overloaded_lines": len(r["parent_violated_lines"]),
            "extent": extent_of(len(r["parent_violated_lines"])),
            "parent_violated_lines": [
                {
                    **{k: v for k, v in x.items() if k != "line"},
                    "label": lab(x["line"], "line"),
                    "overload_over_limit": round(
                        x["rating_needed_for_observed_flow_mva"] / x["rating_mva"], 3
                    ),
                    "severity": severity_of(
                        x["rating_needed_for_observed_flow_mva"] / x["rating_mva"]
                    ),
                }
                for x in r["parent_violated_lines"]
            ],
            "parent_voltage_buses": [lab(b, "bus") for b in r["parent_voltage_buses"]],
            "candidates": [],
        }
        for c in r["candidates"]:
            acts = []
            for a in c["actions"]:
                a = dict(a)
                if "line" in a:
                    a["label"] = lab(a.pop("line"), "line")
                if "bus" in a:
                    a["label"] = lab(a.pop("bus"), "bus")
                acts.append(a)
            rr["candidates"].append(
                {**{k: v for k, v in c.items() if k != "actions"}, "actions": acts}
            )
        if r.get("committed_line_outcomes"):
            rr["committed_line_outcomes"] = [
                {
                    **{k: v for k, v in x.items() if k != "line"},
                    "label": lab(x["line"], "line"),
                }
                for x in r["committed_line_outcomes"]
            ]
        if "after_commit" in r:
            ac = r["after_commit"]
            rr["after_commit"] = {
                k: [lab(x, "bus" if "voltage" in k else "line") for x in v]
                for k, v in ac.items()
            }
        out["rounds"].append(rr)
    fin = dict(L["final"])
    plan = []
    for a in fin.get("plan") or []:
        a = dict(a)
        if "line" in a:
            a["label"] = lab(a.pop("line"), "line")
        if "bus" in a:
            a["label"] = lab(a.pop("bus"), "bus")
        plan.append(a)
    fin["plan"] = plan
    out["final"] = fin
    out["keys"] = L["keys"]
    return out


def calibration(ledgers):
    by = {
        "n1_bound": {"solved": [], "unsolved": [], "committed_final": []},
        "base_bound": {"solved": [], "unsolved": [], "committed_final": []},
    }
    kind_of = {}
    for L in ledgers:
        for r in L["rounds"]:
            lim = {
                v["line"]: (
                    "n1_bound" if v["binding_limit"].startswith("N-1") else "base_bound"
                )
                for v in r["parent_violated_lines"]
            }
            kind_of.update(lim)
            for c in r["candidates"]:
                if c.get("not_verified"):
                    continue
                for a in c["actions"]:
                    if "target_over_needed_ratio" in a and a.get("line") in lim:
                        by[lim[a["line"]]][
                            "solved" if c.get("solved") else "unsolved"
                        ].append(a["target_over_needed_ratio"])
        for a in L["final"]["plan"]:
            if "final_over_initially_needed_ratio" in a and a["line"] in kind_of:
                by[kind_of[a["line"]]]["committed_final"].append(
                    a["final_over_initially_needed_ratio"]
                )

    def summ(v):
        if not v:
            return None
        s = sorted(v)
        q = lambda f: round(s[int(f * (len(s) - 1))], 3)
        bands = Counter(
            (
                "<1.00"
                if x < 1.0
                else (
                    "1.00-1.05"
                    if x < 1.05
                    else (
                        "1.05-1.10"
                        if x < 1.10
                        else "1.10-1.20" if x < 1.20 else ">=1.20"
                    )
                )
            )
            for x in s
        )
        return {
            "n": len(s),
            "min": round(s[0], 3),
            "p10": q(0.1),
            "median": q(0.5),
            "p90": q(0.9),
            "bands": dict(sorted(bands.items())),
        }

    ratios = {k: {"committed_final": summ(v["committed_final"])} for k, v in by.items()}
    per_line = {}
    implied = []
    for kind in ("N-1 110%", "base 100%"):
        rows = [
            x
            for L in ledgers
            for r in L["rounds"]
            for x in r.get("committed_line_outcomes") or []
            if x["binding_limit"] == kind
        ]
        tab = []
        for lo, hi, name in BANDS:
            b = [x for x in rows if lo <= x["rating_over_needed_ratio"] < hi]
            if b:
                res = sum(x["resolved"] for x in b)
                tab.append(
                    {
                        "band": name,
                        "upgrades": len(b),
                        "line_resolved": res,
                        "resolved_rate": round(res / len(b), 3),
                    }
                )
        per_line["n1_bound" if kind.startswith("N-1") else "base_bound"] = tab
        implied += [
            x["rating_actually_needed_over_static_need"]
            for x in rows
            if "rating_actually_needed_over_static_need" in x
        ]
    by_sev = {}
    allrows = [
        x
        for L in ledgers
        for r in L["rounds"]
        for x in r.get("committed_line_outcomes") or []
    ]
    for sv in SEVS:
        tab = []
        for lo, hi, name in BANDS:
            b = [
                x
                for x in allrows
                if x.get("severity") == sv and lo <= x["rating_over_needed_ratio"] < hi
            ]
            if b:
                res = sum(x["resolved"] for x in b)
                tab.append(
                    {
                        "band": name,
                        "upgrades": len(b),
                        "line_resolved": res,
                        "resolved_rate": round(res / len(b), 3),
                    }
                )
        by_sev[sv] = tab
    many = {}
    for sv in SEVS:
        tab = []
        for lo, hi, name in BANDS:
            b = [
                x
                for x in allrows
                if x.get("severity") == sv
                and x.get("extent") == "many"
                and lo <= x["rating_over_needed_ratio"] < hi
            ]
            if b:
                res = sum(x["resolved"] for x in b)
                tab.append(
                    {
                        "band": name,
                        "upgrades": len(b),
                        "line_resolved": res,
                        "resolved_rate": round(res / len(b), 3),
                    }
                )
        many[sv] = tab
    tot = sum(L["final"]["total_cost_musd"] or 0 for L in ledgers) or 1.0
    spend = {}
    for m in (1.03, 1.05, 1.07, 1.10):
        ex = sum(
            max(
                0.0,
                a["final_mva"]
                - m * a["final_mva"] / a["final_over_initially_needed_ratio"],
            )
            * 0.05
            for L in ledgers
            for a in L["final"]["plan"]
            if "final_over_initially_needed_ratio" in a
        )
        spend[f"share_of_spend_above_{m:.2f}x_need"] = round(ex / tot, 3)
    multi = [L for L in ledgers if len(L["rounds"]) > 1]
    first_share, repair_cost = [], []
    for L in ledgers:
        if not L["rounds"]:
            continue
        com = [c for c in L["rounds"][0]["candidates"] if c["committed"]]
        fc = L["final"]["total_cost_musd"]
        if com and com[0].get("plan_cost_musd") and fc:
            first_share.append(min(1.0, com[0]["plan_cost_musd"] / fc))
            if len(L["rounds"]) > 1:
                repair_cost.append(max(0.0, fc - com[0]["plan_cost_musd"]))
    react = [
        (a.get("bus_had_voltage_problem"), c["committed"])
        for L in ledgers
        for r in L["rounds"]
        for c in r["candidates"]
        for a in c["actions"]
        if "bus" in a
    ]
    round_targets = [
        a["final_mva"]
        for L in ledgers
        for a in L["final"]["plan"]
        if "final_mva" in a and a["final_mva"]
    ]
    rounded = sum(1 for t in round_targets if float(t) % 10 == 0)
    return {
        "episodes": len(ledgers),
        "per_line_outcome_of_committed_upgrades": per_line,
        "per_line_outcome_by_overload_severity": by_sev,
        "per_line_outcome_when_many_lines_overloaded": many,
        "lines_still_overloaded_after_upgrade_actually_needed": summ(implied),
        "committed_final_rating_over_needed": ratios,
        "upper_bound_spend_above_margin": spend,
        "first_commit_share_of_final_cost_median": (
            round(statistics.median(first_share), 3) if first_share else None
        ),
        "episodes_needing_more_than_one_round": len(multi),
        "cost_added_after_first_commit_musd_median": (
            round(statistics.median(repair_cost), 2) if repair_cost else None
        ),
        "committed_reactive_devices_at_buses_without_voltage_problem": sum(
            1 for v, c in react if c and not v
        ),
        "committed_reactive_devices": sum(1 for _, c in react if c),
        "final_line_ratings_that_are_multiples_of_10_mva": f"{rounded}/{len(round_targets)}",
    }


EPISODE_SYSTEM = CONSOLIDATION_EPISODE_SYSTEM
MERGE_SYSTEM = CONSOLIDATION_MERGE_SYSTEM


def _schema(merged):
    def strings(n, size, mn=1):
        return {
            "type": "array",
            "minItems": mn,
            "maxItems": n,
            "items": {"type": "string", "minLength": 1, "maxLength": size},
        }

    def enum_list(vals):
        return {
            "type": "array",
            "minItems": 1,
            "maxItems": len(vals),
            "items": {"type": "string", "enum": vals},
        }

    props = {
        "name": {"type": "string", "minLength": 1, "maxLength": 120},
        "trigger": {
            "type": "object",
            "properties": {
                "stage": enum_list(STAGES),
                "violation": enum_list(VIOLATIONS),
                "severity": enum_list(SEVERITIES),
                "extent": enum_list(EXTENTS),
            },
            "required": ["stage", "violation", "severity", "extent"],
            "additionalProperties": False,
        },
        "sizing_rule": {
            "type": "object",
            "properties": {
                "applies_to": {"type": "string", "enum": APPLIES},
                "margins": {
                    "type": "object",
                    "properties": {
                        sv: {
                            "type": "object",
                            "properties": {
                                "low": {"type": "number"},
                                "high": {"type": "number"},
                            },
                            "required": ["low", "high"],
                            "additionalProperties": False,
                        }
                        for sv in SEVS
                    },
                    "required": SEVS,
                    "additionalProperties": False,
                },
            },
            "required": ["applies_to", "margins"],
            "additionalProperties": False,
        },
        "applies_when": strings(3, 240),
        "procedure": strings(5, 260),
        "avoid": strings(4, 240),
        "watch_for": strings(3, 240, 0),
    }
    if merged:
        props["supporting_episode_ids"] = {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
        }
    else:
        props["source_keys"] = {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
        }
    item = {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }
    lo, hi = (2, MAX_MERGED) if merged else (1, MAX_PER_EPISODE)
    return {
        "type": "object",
        "properties": {
            "strategies": {
                "type": "array",
                "minItems": lo,
                "maxItems": hi,
                "items": item,
            }
        },
        "required": ["strategies"],
        "additionalProperties": False,
    }


def _check_common(i, s, floor):
    bad = []
    text = json.dumps(
        {
            k: s.get(k)
            for k in ("name", "applies_when", "procedure", "avoid", "watch_for")
        }
    )
    if REAL_NAME.search(text):
        bad.append(f"strategy {i}: names specific equipment; describe roles instead")
    if LOCAL_LABEL.search(text):
        bad.append(
            f"strategy {i}: uses trajectory-local labels (L1, B2 ...); describe roles instead"
        )
    if STOCK.search(text):
        bad.append(
            f"strategy {i}: restates a stock phrase; state the concrete decision rule"
        )
    r = s.get("sizing_rule") or {}
    if r.get("applies_to") != "none":
        for sv in SEVS:
            m = (r.get("margins") or {}).get(sv) or {}
            lo, hi = m.get("low"), m.get("high")
            covers_many = bool(
                {"many", "any"} & set(((s.get("trigger") or {}).get("extent")) or [])
            )
            f = (floor or {}).get(("many_" + sv) if covers_many else sv)
            cap = (floor or {}).get(
                ("cap_many_" + sv) if covers_many else ("cap_" + sv)
            )
            if (
                not isinstance(lo, (int, float))
                or not isinstance(hi, (int, float))
                or not (0.98 <= lo <= hi <= 1.5)
            ):
                bad.append(
                    f"strategy {i}: {sv} margins must satisfy 0.98 <= low <= high <= 1.5"
                )
            elif hi - lo > 0.10:
                bad.append(
                    f"strategy {i}: {sv} margin range is too wide to guide sizing (max 0.10)"
                )
            else:
                if cap is not None and lo > cap + 0.005:
                    bad.append(
                        f"strategy {i}: {sv} low margin {lo} is above {cap}: committed upgrades "
                        f"of {sv} overloads already always resolved below that, so the extra "
                        f"rating buys nothing. Use a low margin between {f} and {cap}."
                    )
                if f is not None and lo < f - 0.005:
                    bad.append(
                        f"strategy {i}: {sv} low margin {lo} is below {f}, the lowest margin the "
                        f"evidence supports for {sv} overloads"
                        + (
                            " when many lines are overloaded at once (every line must resolve)"
                            if covers_many and EXTENT_FLOORS
                            else ""
                        )
                        + "."
                    )
    return bad


def wilson_low(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d


FLOOR_CONFIDENCE = 0.80


EXTENT_FLOORS = os.environ.get("SCOPE_EXTENT_FLOORS", "1") == "1"


def _qualifies(t):
    return wilson_low(t["line_resolved"], t["upgrades"]) >= FLOOR_CONFIDENCE


def _floor(stats):
    edges = {name: lo for lo, _, name in BANDS}
    out = {}
    for k in ("n1_bound", "base_bound"):
        tab = (stats.get("per_line_outcome_of_committed_upgrades") or {}).get(k) or []
        ok = [edges[t["band"]] for t in tab if _qualifies(t)]
        out[k] = round(min(ok), 3) if ok else None
    if out["base_bound"] is None:
        out["base_bound"] = out["n1_bound"]
    glob_floor = out["n1_bound"] or 1.05
    for sv in SEVS:
        tab = (stats.get("per_line_outcome_by_overload_severity") or {}).get(sv) or []
        ok = [edges[t["band"]] for t in tab if _qualifies(t)]

        out[sv] = round(min(ok), 3) if ok else round(max(glob_floor, 1.05), 3)

    for sv in SEVS:
        tab = (stats.get("per_line_outcome_by_overload_severity") or {}).get(sv) or []
        sat = [
            edges[t["band"]]
            for t in tab
            if t["upgrades"] >= 10 and t["line_resolved"] == t["upgrades"]
        ]
        c = min(sat) if sat else out[sv] + 0.05
        out["cap_" + sv] = round(max(c, out[sv]) + 0.02, 3)

    for sv in SEVS:
        tab = (stats.get("per_line_outcome_when_many_lines_overloaded") or {}).get(
            sv
        ) or []
        ok = [
            edges[t["band"]]
            for t in tab
            if t["line_resolved"] == t["upgrades"] and _qualifies(t)
        ]
        base = out[sv]
        if not EXTENT_FLOORS:
            out["many_" + sv] = base
            out["cap_many_" + sv] = out["cap_" + sv]
            continue
        out["many_" + sv] = (
            round(max(min(ok), base), 3) if ok else round(base + 0.02, 3)
        )
        out["cap_many_" + sv] = round(
            max(out["cap_" + sv], out["many_" + sv] + 0.02), 3
        )
    return out


def _shape(st, lo, hi):
    if not isinstance(st, list) or not (lo <= len(st) <= hi):
        raise ValueError(f"Submit between {lo} and {hi} strategies as a list.")
    for i, s in enumerate(st):
        if not isinstance(s, dict):
            raise ValueError(f"strategy {i} must be an object")
        t = s.get("trigger")
        if not isinstance(t, dict) or not all(
            isinstance(t.get(k), list) and t.get(k)
            for k in ("stage", "violation", "severity", "extent")
        ):
            raise ValueError(
                f"strategy {i}: trigger needs non-empty lists stage, violation, severity, extent"
            )
        bad_enum = [
            v
            for k, allowed in (
                ("stage", STAGES),
                ("violation", VIOLATIONS),
                ("severity", SEVERITIES),
                ("extent", EXTENTS),
            )
            for v in t[k]
            if v not in allowed
        ]
        if bad_enum:
            raise ValueError(
                f"strategy {i}: trigger values not allowed: {bad_enum[:3]}"
            )
        r = s.get("sizing_rule")
        if not isinstance(r, dict) or r.get("applies_to") not in APPLIES:
            raise ValueError(
                f"strategy {i}: sizing_rule.applies_to must be one of {APPLIES}"
            )
        m = r.get("margins")
        if not isinstance(m, dict) or not all(
            isinstance(m.get(sv), dict) for sv in SEVS
        ):
            raise ValueError(
                f"strategy {i}: sizing_rule.margins needs objects for {SEVS}"
            )
        for sv in SEVS:
            for k in ("low", "high"):
                v = m[sv].get(k)
                if isinstance(v, str):
                    try:
                        m[sv][k] = float(v)
                    except ValueError:
                        raise ValueError(f"strategy {i}: {sv}.{k} must be a number")
                elif not isinstance(v, (int, float)) or isinstance(v, bool):
                    raise ValueError(f"strategy {i}: {sv}.{k} must be a number")
        for k in ("applies_when", "procedure", "avoid", "watch_for"):
            v = s.get(k)
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise ValueError(f"strategy {i}: {k} must be a list of strings")
        if not isinstance(s.get("name"), str) or not s["name"].strip():
            raise ValueError(f"strategy {i}: name must be a non-empty string")


def episode_validator(keys, floor):
    keys = set(keys)

    def v(args):
        if not isinstance(args, dict):
            raise ValueError("Submit an object with a strategies list.")
        st = args.get("strategies")
        _shape(st, 1, MAX_PER_EPISODE)
        bad = []
        for i, s in enumerate(st):
            bad += _check_common(i, s, floor)
            if not isinstance(s.get("source_keys"), list):
                raise ValueError(f"strategy {i}: source_keys must be a list")
            k = [x for x in s.get("source_keys") or [] if x not in keys]
            if k or not s.get("source_keys"):
                bad.append(
                    f"strategy {i}: unknown source_keys {k[:3]}; use keys such as R1, R1.B, FINAL"
                )
        if bad:
            raise ValueError("Fix and resubmit all strategies: " + " | ".join(bad[:6]))
        return st

    return v


def merge_validator(episode_ids, floor, final=False):
    episode_ids = set(episode_ids)

    def v(args):
        if not isinstance(args, dict):
            raise ValueError("Submit an object with a strategies list.")
        st = args.get("strategies")
        _shape(st, 2, MAX_MERGED)
        bad = []
        for i, s in enumerate(st):
            bad += _check_common(i, s, floor)
            if not isinstance(s.get("supporting_episode_ids"), list):
                raise ValueError(f"strategy {i}: supporting_episode_ids must be a list")
            e = [
                x for x in s.get("supporting_episode_ids") or [] if x not in episode_ids
            ]
            if e or not s.get("supporting_episode_ids"):
                bad.append(f"strategy {i}: unknown supporting_episode_ids {e[:2]}")
        if final:
            if not any(
                (s.get("sizing_rule") or {}).get("applies_to") != "none" for s in st
            ):
                bad.append("the bank needs at least one strategy with a sizing_rule")
            if not any(
                {"first_plan", "any"} & set((s.get("trigger") or {}).get("stage") or [])
                for s in st
            ):
                bad.append(
                    "the bank needs at least one strategy whose stage covers first_plan"
                )
            if not any(
                {"many", "any"} & set((s.get("trigger") or {}).get("extent") or [])
                for s in st
            ):
                bad.append(
                    "the bank needs at least one strategy whose extent covers many overloads"
                )
        if bad:
            raise ValueError("Fix and resubmit all strategies: " + " | ".join(bad[:6]))
        return st

    return v


def _reflect(model, A, floor, cache_dir):
    packet = {
        "trajectory": {k: A[k] for k in ("family", "rounds", "final")},
        "allowed_source_keys": A["keys"],
        "validity_floor": {
            k: v
            for k, v in floor.items()
            if not k.startswith("cap_") and k not in ("n1_bound", "base_bound")
        },
        "waste_ceiling": {k[4:]: v for k, v in floor.items() if k.startswith("cap_")},
    }
    key = sha(
        {
            "builder": BUILDER,
            "system": EPISODE_SYSTEM,
            "packet": packet,
            "floor": floor,
            "settings": getattr(model, "settings", None),
        }
    )
    path = os.path.join(cache_dir, key + ".json")
    if os.path.exists(path):
        return json.load(open(path))["strategies"]
    before = len(getattr(model, "calls", []))
    st = model.ask(
        EPISODE_SYSTEM,
        packet,
        "submit_episode_strategies",
        _schema(False),
        episode_validator(A["keys"], floor),
    )
    os.makedirs(cache_dir, exist_ok=True)
    json.dump(
        {"strategies": st, "calls": getattr(model, "calls", [])[before:]},
        open(path, "w"),
        indent=1,
    )
    return st


def _compact(s, ep_id=None):
    out = {
        k: s[k]
        for k in (
            "name",
            "trigger",
            "sizing_rule",
            "applies_when",
            "procedure",
            "avoid",
            "watch_for",
        )
        if k in s
    }
    out["supporting_episode_ids"] = s.get("supporting_episode_ids") or [ep_id]
    return out


MERGE_BUDGET_CHARS = 48000


def _clamp(s, floor):
    notes = []
    r = s.get("sizing_rule") or {}
    if r.get("applies_to") in (None, "none") or not isinstance(r.get("margins"), dict):
        return notes
    many = bool({"many", "any"} & set(((s.get("trigger") or {}).get("extent")) or []))
    for sv in SEVS:
        m = r["margins"].get(sv)
        if not isinstance(m, dict):
            continue
        f = floor.get(("many_" + sv) if many else sv)
        cap = floor.get(("cap_many_" + sv) if many else ("cap_" + sv))
        try:
            lo, hi = float(m.get("low")), float(m.get("high"))
        except (TypeError, ValueError):
            continue
        nlo = min(max(lo, f if f is not None else lo), cap if cap is not None else lo)
        if hi >= nlo:
            nhi = hi
        else:
            nhi = min(nlo + 0.03, cap) if cap is not None and cap > nlo else nlo + 0.03
        nhi = min(nhi, nlo + 0.10, 1.5)
        if (nlo, nhi) != (lo, hi):
            notes.append(
                {"severity": sv, "from": [lo, hi], "to": [round(nlo, 3), round(nhi, 3)]}
            )
            m["low"], m["high"] = round(nlo, 3), round(nhi, 3)
    return notes


def _salvage(model, validate, validate_each, floor, exc):
    log = getattr(model, "salvage_log", None)
    if log is None:
        log = []
        try:
            model.salvage_log = log
        except Exception:
            pass
    calls = getattr(model, "calls", []) or []
    args = None
    for rec in reversed(calls):
        tc = rec.get("tool_calls") or []
        if tc:
            raw = (tc[0].get("function") or {}).get("arguments")
            try:
                if isinstance(raw, str):
                    t = raw.strip()
                    if t.startswith("```"):
                        t = "\n".join(
                            l for l in t.splitlines() if not l.strip().startswith("```")
                        )
                    raw = json.loads(t)
                args = raw
                break
            except Exception:
                continue
    st = (args or {}).get("strategies") if isinstance(args, dict) else None
    if not isinstance(st, list):
        raise exc
    clamped = []
    for i, x in enumerate(st):
        if isinstance(x, dict):
            n = _clamp(x, floor)
            if n:
                clamped.append({"strategy": i, "changes": n})
    try:
        out = validate({"strategies": st})
        log.append(
            {"action": "clamped", "clamped": clamped, "original_error": str(exc)[:300]}
        )
        return out
    except ValueError:
        pass
    keep = []
    for x in st:
        try:
            validate_each({"strategies": [x, x]})
            keep.append(x)
        except ValueError:
            continue
    if len(keep) >= 2:
        try:
            out = validate({"strategies": keep})
            log.append(
                {
                    "action": "clamped_and_dropped",
                    "clamped": clamped,
                    "kept": len(keep),
                    "submitted": len(st),
                    "original_error": str(exc)[:300],
                }
            )
            return out
        except ValueError:
            pass
    raise exc


def _tight(x):
    y = dict(x)
    for k, n in (("applies_when", 2), ("procedure", 3), ("avoid", 2), ("watch_for", 2)):
        y[k] = [t[:180] for t in (x.get(k) or [])[:n]]
    y["name"] = (x.get("name") or "")[:100]
    return y


def _fits(strategies, base, budget):
    return len(json.dumps({"strategies": strategies, **base})) <= budget


def _merge_bounded(
    model, strategies, stats, ep_ids, floor, final, budget=MERGE_BUDGET_CHARS, depth=0
):
    base = {
        "calibration": stats,
        "validity_floor": {
            k: v
            for k, v in floor.items()
            if not k.startswith("cap_") and k not in ("n1_bound", "base_bound")
        },
        "waste_ceiling": {k[4:]: v for k, v in floor.items() if k.startswith("cap_")},
    }

    def ask(st):
        validate = merge_validator(ep_ids, floor, final=final)
        try:
            return model.ask(
                MERGE_SYSTEM,
                {"strategies": st, **base},
                "submit_merged_strategies",
                _schema(True),
                validate,
            )
        except ValueError as exc:
            return _salvage(
                model, validate, merge_validator(ep_ids, floor, final=False), floor, exc
            )

    if _fits(strategies, base, budget):
        return ask(strategies)
    if depth == 0:
        return _merge_bounded(
            model,
            [_tight(x) for x in strategies],
            stats,
            ep_ids,
            floor,
            final,
            budget,
            1,
        )
    room = max(budget - len(json.dumps(base)) - 200, 4000)
    chunks, cur, size = [], [], 0
    for x in strategies:
        n = len(json.dumps(x))
        if cur and size + n > room:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(x)
        size += n
    if cur:
        chunks.append(cur)
    if len(chunks) > 1 and len(chunks[-1]) < 3:
        last = chunks.pop()
        chunks[-1] = chunks[-1] + last
    if len(chunks) > 1 and depth < 3:
        partial = []
        for ch in chunks:
            if _fits(ch, base, budget):
                partial += [
                    _tight(_compact(x))
                    for x in model.ask(
                        MERGE_SYSTEM,
                        {"strategies": ch, **base},
                        "submit_merged_strategies",
                        _schema(True),
                        merge_validator(ep_ids, floor, final=False),
                    )
                ]
            else:
                partial += ch
        if len(partial) < len(strategies):
            return _merge_bounded(
                model, partial, stats, ep_ids, floor, final, budget, depth + 1
            )
        strategies = partial
    keep, ranked = [], sorted(
        strategies, key=lambda x: -len(x.get("supporting_episode_ids") or [])
    )
    for x in ranked:
        if _fits(keep + [x], base, budget):
            keep.append(x)
    return ask(keep if len(keep) >= 2 else ranked[:2])


def build(source_memory_dir, model, fold, allowed_profiles, cache_dir):
    traj = load_trajectories(source_memory_dir)
    if set(traj) != set(allowed_profiles):
        raise ValueError(
            f"{fold}: staged trajectories differ from the fold's source profiles"
        )
    raw = []
    no_events = []
    for _, (e, d) in sorted(traj.items()):
        L = ledger(e, d)
        outs = committed_outcomes(e)
        if not L["rounds"] and outs:

            no_events.append(L["profile"])
            for i, st in enumerate(e.get("steps") or []):
                before = st.get("state_before") or {}
                needs = line_needs(before)
                L["rounds"].append(
                    {
                        "key": f"R{i + 1}",
                        "parent_security5": _vec(before.get("security5")),
                        "parent_violated_lines": [
                            {
                                "line": l,
                                "worst_loading_pct": round(n["pct"], 1),
                                "binding_limit": (
                                    "N-1 110%" if n["kind"] == "n1" else "base 100%"
                                ),
                                "rating_mva": n["rating"],
                                "rating_needed_for_observed_flow_mva": round(
                                    n["need"], 1
                                ),
                            }
                            for l, n in needs.items()
                        ],
                        "parent_voltage_buses": sorted(bus_depths(before)),
                        "candidates": [],
                        "note": "run events unavailable; committed step only",
                    }
                )
            L["keys"] = sorted(set(L["keys"]) | {r["key"] for r in L["rounds"]})
        for r, rows in zip(L["rounds"], outs):
            r["committed_line_outcomes"] = rows
        raw.append(L)
    stats = calibration(raw)

    floor = _floor(stats)
    per, failed = [], []
    for L in raw:
        A = anonymise(L)
        try:
            for s in _reflect(model, A, floor, cache_dir):
                per.append((L["episode_id"], L["family"], s))
        except ValueError as exc:
            failed.append({"profile": L["profile"], "error": str(exc)[:300]})
    if len(failed) > 0.2 * len(raw):
        raise ValueError(
            f"{fold}: {len(failed)} of {len(raw)} episode reflections failed; "
            f"first error: {failed[0]['error']}"
        )
    ep_ids = [L["episode_id"] for L in raw]
    before = len(getattr(model, "calls", []))
    fam_level = []
    for fam in sorted({f for _, f, _ in per}):
        group = [_compact(s, e) for e, f, s in per if f == fam]
        fam_raw = [L for L in raw if L["family"] == fam]
        fam_stats = calibration(fam_raw)
        out = _merge_bounded(
            model, group, fam_stats, [L["episode_id"] for L in fam_raw], floor, False
        )
        fam_level += [_compact(s) for s in out]
    merged = _merge_bounded(model, fam_level, stats, ep_ids, floor, True)
    for i, s in enumerate(merged):
        s["strategy_id"] = f"{fold}_S{i + 1}"
    return {
        "schema": SCHEMA,
        "builder": BUILDER,
        "fold": fold,
        "profiles": sorted(allowed_profiles),
        "source": "raw trajectories (episode files + run events), equipment anonymised; no simulator runs",
        "calibration": stats,
        "validity_floor": floor,
        "n_episode_strategies": len(per),
        "episodes_without_run_events": no_events,
        "failed_episode_reflections": failed,
        "merge_salvage": getattr(model, "salvage_log", []),
        "prompt_sha": {"episode": sha(EPISODE_SYSTEM), "merge": sha(MERGE_SYSTEM)},
        "model_settings": getattr(model, "settings", None),
        "strategies": merged,
        "merge_calls": getattr(model, "calls", [])[before:],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--source",
        required=True,
        help="memory folder with episodes/ and evolution/ run events",
    )
    ap.add_argument(
        "--fold", required=True, help="evaluation group the bank is for, e.g. fold_1"
    )
    ap.add_argument(
        "--profiles-from",
        default="memory/folds.json",
        help="folds file listing each group's source scenarios",
    )
    ap.add_argument("--out", required=True, help="output strategy bank JSON")
    ap.add_argument("--model", default=os.environ.get("SCOPE_MODEL", "qwen3.5:9b"))
    ap.add_argument("--cache", default="runs/consolidation_cache")
    ap.add_argument(
        "--thinking", action="store_true", help="let the model think (off in the paper)"
    )
    a = ap.parse_args()
    import trajectory_llm as TL

    folds = json.load(open(a.profiles_from))
    if not a.thinking:

        _orig = TL.chat_vllm

        def _no_think(*args, **kw):
            kw.update(
                enable_thinking=False, reasoning_effort="none", print_reasoning=False
            )
            return _orig(*args, **kw)

        TL.chat_vllm = _no_think
    model = TL.Model(name=a.model, num_ctx=32768, num_predict=6000, stream=False)
    model.settings["thinking"] = a.thinking
    bank = build(a.source, model, a.fold, folds[a.fold]["source"], a.cache)
    with open(a.out, "w") as f:
        json.dump(bank, f, indent=1)
    print(f"{a.fold}: {len(bank['strategies'])} strategies -> {a.out}")


if __name__ == "__main__":
    main()
