import argparse
import json
import os
import sys
from dataclasses import asdict

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from env.config import Level3Config, Weather
from mpc.core import GameConfig, OnlineMPC
from mpc.simulator import run_online_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Level3 online MPC")
    parser.add_argument("--horizon", type=int, default=4, help="MPC视野长度")
    parser.add_argument("--scenarios", type=int, default=3, help="天气情景数")
    parser.add_argument("--init-water", type=int, default=180)
    parser.add_argument("--init-food", type=int, default=180)
    args = parser.parse_args()

    cfg = GameConfig.from_level(Level3Config)
    solver = OnlineMPC(cfg, horizon=args.horizon, n_scenarios=args.scenarios)

    weather_seq = [Weather.SUNNY if i % 2 == 0 else Weather.HOT for i in range(cfg.num_days)]
    result = run_online_simulation(
        cfg=cfg,
        mpc=solver,
        weather_sequence=weather_seq,
        init_water=args.init_water,
        init_food=args.init_food,
    )

    serializable_trace = []
    for row in result.trace:
        row_copy = dict(row)
        action = row_copy.get("action")
        row_copy["action"] = asdict(action) if action is not None else None
        serializable_trace.append(row_copy)

    payload = {
        "success": result.success,
        "final_state": asdict(result.final_state),
        "trace": serializable_trace,
    }

    output_dir = os.path.join(ROOT, "outputs", "level34_results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "level3_result.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
