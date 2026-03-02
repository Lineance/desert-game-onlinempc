import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from env.config import Level3Config, Weather
from mpc.core import GameConfig, OnlineMPC
from mpc.simulator import run_online_simulation


def main() -> None:
    cfg = GameConfig.from_level(Level3Config)
    solver = OnlineMPC(cfg, horizon=cfg.num_days, n_scenarios=1)

    weather_seq = [
        Weather.SUNNY if i % 2 == 0 else Weather.HOT for i in range(cfg.num_days)
    ]
    result = run_online_simulation(
        cfg=cfg,
        mpc=solver,
        weather_sequence=weather_seq,
        init_water=180,
        init_food=180,
    )

    print("[Level3-Global] success:", result.success)
    print("[Level3-Global] final day:", result.final_state.day)
    print("[Level3-Global] final money:", result.final_state.money)


if __name__ == "__main__":
    main()
