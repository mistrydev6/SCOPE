"""Static configuration for the IEEE-118 planning tools."""

from pathlib import Path
import os

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
DATA = PROJECT_DIR / "data"
TOPOLOGY = "topology_ieee118.json"

SCENARIO_DATASET = "ieee118_scenarios"
SCENARIO_DIR = DATA / SCENARIO_DATASET
SCENARIO_RECORDS = SCENARIO_DIR / "records"
SCENARIO_MAPPING = SCENARIO_DIR / "mapping_report.json"

MEMORY = Path(
    os.environ.get(
        "MEMORY_OVERRIDE",
        PROJECT_DIR / "runs" / "memory",
    )
)

LIST_LIMIT = 40

V_MIN, V_MAX = 0.95, 1.05
V_MIN_N1 = 0.92
THERMAL_MAX = 100.0
THERMAL_MAX_N1 = 110.0

COST_LINE_FIXED, COST_LINE_PER_MVA = 2.0, 0.05
COST_TRAFO_FIXED, COST_TRAFO_PER_MVA = 2.5, 0.035
COST_NEW_FIXED, COST_NEW_PER_MVA = 20.0, 0.10
COST_BUS_FIXED, COST_BUS_PER_MVAR = 0.5, 0.02
