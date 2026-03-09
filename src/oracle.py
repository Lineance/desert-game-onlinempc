from dataclasses import dataclass
from typing import List

from mpc.core import GameConfig, OnlineMPC
from mpc.predictor import DeterministicWeatherPredictor
from mpc.simulator import run_online_simulation


@dataclass
class OracleResult:
    success: bool
    final_money: float
    end_day: int


def solve_offline_oracle(
    cfg: GameConfig,
    weather_sequence: List[int],
    init_water: int,
    init_food: int,
) -> OracleResult:
    predictor = DeterministicWeatherPredictor(weather_sequence)
    mpc = OnlineMPC(cfg, horizon=cfg.num_days, n_scenarios=1, predictor=predictor)
    sim = run_online_simulation(
        cfg=cfg,
        mpc=mpc,
        weather_sequence=weather_sequence,
        init_water=init_water,
        init_food=init_food,
    )
    return OracleResult(
        success=sim.success,
        final_money=sim.final_state.money,
        end_day=sim.final_state.day,
    )
