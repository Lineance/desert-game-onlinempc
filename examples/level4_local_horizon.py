import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from env.config import Level4Config, Weather
from mpc.core import GameConfig, OnlineMPC
from mpc.simulator import run_online_simulation


def main() -> None:
    cfg = GameConfig.from_level(Level4Config)
    solver = OnlineMPC(cfg, horizon=8, n_scenarios=5)

    weather_seq = [Weather.SUNNY] * cfg.num_days
    result = run_online_simulation(
        cfg=cfg,
        mpc=solver,
        weather_sequence=weather_seq,
        init_water=260,
        init_food=220,
    )

    print("[Level4-Local] success:", result.success)
    print("[Level4-Local] final day:", result.final_state.day)
    print("[Level4-Local] final money:", result.final_state.money)


if __name__ == "__main__":
    main()
