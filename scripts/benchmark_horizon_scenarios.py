# ruff: noqa: E402
import argparse
import os
import sys
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from env.config import Level3Config, Level4Config
from mpc.benchmark import run_grid_benchmark
from mpc.core import GameConfig


def _pick_cfg(level: int) -> tuple[str, GameConfig, int, int]:
    if level == 3:
        cfg = GameConfig.from_level(Level3Config)
        return "level3", cfg, 180, 180
    cfg = GameConfig.from_level(Level4Config)
    return "level4", cfg, 260, 220


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark horizon and n_scenarios")
    parser.add_argument("--level", type=int, choices=[3, 4], required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizons", type=int, nargs="+", required=True)
    parser.add_argument("--scenarios", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--init-water", type=int, default=None)
    parser.add_argument("--init-food", type=int, default=None)
    args = parser.parse_args()

    level_name, cfg, default_w, default_f = _pick_cfg(args.level)
    init_water = args.init_water if args.init_water is not None else default_w
    init_food = args.init_food if args.init_food is not None else default_f

    records = run_grid_benchmark(
        cfg=cfg,
        level_name=level_name,
        horizons=args.horizons,
        scenario_counts=args.scenarios,
        episodes=args.episodes,
        base_seed=args.seed,
        init_water=init_water,
        init_food=init_food,
        workers=max(1, args.workers),
        solver_threads=max(1, args.solver_threads),
    )

    out_dir = os.path.join(ROOT, "outputs", "benchmarks", "raw")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{level_name}_benchmark_{ts}.csv")

    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"Saved benchmark records: {out_path}")
    print(f"rows={len(records)}")
    print(f"workers={max(1, args.workers)}, solver_threads={max(1, args.solver_threads)}")


if __name__ == "__main__":
    main()
