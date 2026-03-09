from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from typing import Dict, List

import numpy as np
import pulp

from mpc.core import GameConfig, OnlineMPC, State
from mpc.predictor import DeterministicWeatherPredictor
from mpc.simulator import run_online_simulation
from oracle import solve_offline_oracle


@dataclass
class ValidationReport:
    online_success: bool
    oracle_success: bool
    online_final_money: float
    oracle_final_money: float
    gap: float


@dataclass
class ToolComparisonReport:
    case_name: str
    tool_status: str
    tool_objective: float
    tool_init_water: int
    tool_init_food: int
    mpc_status: str
    mpc_objective: float
    objective_gap: float


def backtest_against_oracle(
    cfg: GameConfig,
    weather_sequence: List[int],
    init_water: int,
    init_food: int,
    horizon: int,
    n_scenarios: int,
) -> ValidationReport:
    online_solver = OnlineMPC(cfg, horizon=horizon, n_scenarios=n_scenarios)
    online_result = run_online_simulation(
        cfg=cfg,
        mpc=online_solver,
        weather_sequence=weather_sequence,
        init_water=init_water,
        init_food=init_food,
    )

    oracle_result = solve_offline_oracle(
        cfg=cfg,
        weather_sequence=weather_sequence,
        init_water=init_water,
        init_food=init_food,
    )

    gap = oracle_result.final_money - online_result.final_state.money
    return ValidationReport(
        online_success=online_result.success,
        oracle_success=oracle_result.success,
        online_final_money=online_result.final_state.money,
        oracle_final_money=oracle_result.final_money,
        gap=gap,
    )


def report_to_dict(report: ValidationReport) -> Dict[str, float | bool]:
    return {
        "online_success": report.online_success,
        "oracle_success": report.oracle_success,
        "online_final_money": report.online_final_money,
        "oracle_final_money": report.oracle_final_money,
        "gap": report.gap,
    }


def _load_tool_module(tool_name: str):
    base = Path(__file__).resolve().parents[2] / "tools"
    path = base / f"{tool_name}.py"
    loader = SourceFileLoader(tool_name, str(path))
    spec = spec_from_loader(tool_name, loader)
    if spec is None:
        raise RuntimeError(f"Unable to load {path}")
    module = module_from_spec(spec)
    loader.exec_module(module)
    return module


def _build_cfg_from_tool_module(module) -> GameConfig:
    mines = list(module.MINES) if hasattr(module, "MINES") else [module.MINE]
    villages = list(module.VILLAGES) if hasattr(module, "VILLAGES") else [module.VILLAGE]

    class ToolLevelConfig:
        NUM_NODES = module.NUM_NODES
        NUM_DAYS = module.NUM_DAYS
        INIT_MONEY = module.INIT_MONEY
        WEIGHT_LIMIT = module.WEIGHT_LIMIT
        MINE_INCOME = module.MINE_INCOME
        WATER_WEIGHT = module.WATER_WEIGHT
        WATER_PRICE_BASE = module.WATER_PRICE_BASE
        FOOD_WEIGHT = module.FOOD_WEIGHT
        FOOD_PRICE_BASE = module.FOOD_PRICE_BASE
        START = module.START
        END = module.END
        MINES = mines
        VILLAGES = villages
        EDGES = list(module.EDGES)
        WEATHER_TRANSITION = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    base_consumption = {
        0: tuple(module.BASE_CONS[1]),
        1: tuple(module.BASE_CONS[2]),
        2: tuple(module.BASE_CONS[3]),
    }
    return GameConfig.from_level(ToolLevelConfig, base_consumption=base_consumption)


def _solve_tool_reference(module, time_limit: int = 120) -> Dict[str, float | str | int]:
    prob, v, _ = module.build_model()
    solver = pulp.HiGHS(msg=False, timeLimit=time_limit)
    try:
        prob.solve(solver)
    except Exception:
        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))

    status = pulp.LpStatus[prob.status]
    objective = float(pulp.value(prob.objective))
    init_w = int(round(pulp.value(v["buy_w"][0])))
    init_f = int(round(pulp.value(v["buy_f"][0])))
    return {
        "status": status,
        "objective": objective,
        "init_water": init_w,
        "init_food": init_f,
    }


def compare_solver_with_tool_reference(
    tool_name: str, time_limit: int = 120
) -> ToolComparisonReport:
    module = _load_tool_module(tool_name)
    ref = _solve_tool_reference(module, time_limit=time_limit)

    cfg = _build_cfg_from_tool_module(module)
    weather_seq = [int(w) - 1 for w in module.WEATHER]

    predictor = DeterministicWeatherPredictor(weather_seq)
    offline_mpc = OnlineMPC(cfg, horizon=cfg.num_days, n_scenarios=1, predictor=predictor)

    init_water = int(ref["init_water"])
    init_food = int(ref["init_food"])
    init_money = cfg.init_money - cfg.water_price * init_water - cfg.food_price * init_food
    init_state = State(
        day=0,
        pos=cfg.start,
        water=init_water,
        food=init_food,
        money=init_money,
        weather=weather_seq[0] if weather_seq else 0,
    )

    action, info = offline_mpc.solve(init_state, deterministic_weather=weather_seq)
    mpc_status = str(info.get("status", "Unknown"))
    mpc_obj = (
        float(action.expected_terminal_value)
        if action.expected_terminal_value is not None
        else float("-inf")
    )
    gap = float(ref["objective"] - mpc_obj)

    return ToolComparisonReport(
        case_name=tool_name,
        tool_status=str(ref["status"]),
        tool_objective=float(ref["objective"]),
        tool_init_water=int(ref["init_water"]),
        tool_init_food=int(ref["init_food"]),
        mpc_status=mpc_status,
        mpc_objective=mpc_obj,
        objective_gap=gap,
    )
