from __future__ import annotations

from .. import state as S
from .. import measurement as M


def analyse(profile: str = "normal_001") -> dict:
    key = ("analyse", profile)
    if key not in S.DAY_CACHE:
        S.DAY_CACHE[key] = M._measure_day(profile, [])
    out = dict(S.DAY_CACHE[key])
    out["note"] = (
        "Thermal and voltage measurements cover all 24 hours. "
        "N-1 measurements are evaluated at the maximum-total-load hour. "
        "security5.vector is [T_B,V_B,T_N1,V_N1,F_N1], where lower is better in every "
        "dimension. Values are observations only; the simulator does not recommend an intervention."
    )
    return out
