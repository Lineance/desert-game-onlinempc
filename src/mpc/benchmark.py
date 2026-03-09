from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Dict, Iterable, List

import numpy as np

from env.config import Weather
from mpc.core import GameConfig, OnlineMPC, State
from mpc.predictor import MarkovWeatherPredictor
from mpc.simulator import apply_action


@dataclass
class EpisodeMetrics:
    level: str
    horizon: int
    n_scenarios: int
    episode: int
    success: bool
    final_money: float
    end_day: int
    avg_step_time_ms: float
    p95_step_time_ms: float
    infeasible_rate: float


@dataclass
class AggregateMetrics:
    level: str
    horizon: int
    n_scenarios: int
    episodes: int
    success_rate: float
    mean_final_money: float
    std_final_money: float
    mean_avg_step_time_ms: float
    p95_step_time_ms: float
    mean_infeasible_rate: float


@dataclass
class EpisodeTask:
    cfg: GameConfig
    level_name: str
    horizon: int
    n_scenarios: int
    episode: int
    weather_sequence: List[int]
    init_water: int
    init_food: int
    predictor_seed: int
    solver_threads: int


def sample_weather_sequence(
    transition: np.ndarray,
    num_days: int,
    rng: np.random.Generator,
    start_weather: int = Weather.SUNNY,
) -> List[int]:
    weather: List[int] = []
    state = start_weather
    for _ in range(num_days):
        weather.append(state)
        state = int(rng.choice(3, p=transition[state]))
    return weather


def run_online_episode(
    cfg: GameConfig,
    level_name: str,
    horizon: int,
    n_scenarios: int,
    episode: int,
    weather_sequence: List[int],
    init_water: int,
    init_food: int,
    predictor_seed: int,
    solver_threads: int = 1,
) -> EpisodeMetrics:
    init_money = cfg.init_money - cfg.water_price * init_water - cfg.food_price * init_food
    state = State(
        day=0,
        pos=cfg.start,
        water=init_water,
        food=init_food,
        money=init_money,
        weather=weather_sequence[0],
    )

    predictor = MarkovWeatherPredictor(cfg.weather_transition, rng_seed=predictor_seed)
    mpc = OnlineMPC(
        cfg,
        horizon=horizon,
        n_scenarios=n_scenarios,
        predictor=predictor,
        solver_threads=solver_threads,
    )

    step_times: List[float] = []
    infeasible_steps = 0

    for day in range(1, cfg.num_days + 1):
        weather_today = weather_sequence[day - 1]
        state = State(
            day=state.day,
            pos=state.pos,
            water=state.water,
            food=state.food,
            money=state.money,
            weather=weather_today,
        )

        t0 = perf_counter()
        action, info = mpc.solve(state)
        step_ms = (perf_counter() - t0) * 1000
        step_times.append(step_ms)

        if info.get("status") != "Optimal":
            infeasible_steps += 1

        state = apply_action(cfg, state, action, weather_today)

        if state.water < 0 or state.food < 0:
            break

        if state.pos == cfg.end:
            final_money = (
                state.money
                + 0.5 * cfg.water_price * state.water
                + 0.5 * cfg.food_price * state.food
            )
            state = State(
                day=state.day,
                pos=state.pos,
                water=state.water,
                food=state.food,
                money=final_money,
                weather=state.weather,
            )
            break

    success = state.pos == cfg.end and state.water >= 0 and state.food >= 0
    avg_step = float(np.mean(step_times)) if step_times else 0.0
    p95_step = float(np.percentile(step_times, 95)) if step_times else 0.0
    infeasible_rate = float(infeasible_steps / len(step_times)) if step_times else 0.0

    return EpisodeMetrics(
        level=level_name,
        horizon=horizon,
        n_scenarios=n_scenarios,
        episode=episode,
        success=success,
        final_money=float(state.money),
        end_day=state.day,
        avg_step_time_ms=avg_step,
        p95_step_time_ms=p95_step,
        infeasible_rate=infeasible_rate,
    )


def _run_task(task: EpisodeTask) -> Dict[str, object]:
    os.environ["OMP_NUM_THREADS"] = str(task.solver_threads)
    os.environ["MKL_NUM_THREADS"] = str(task.solver_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(task.solver_threads)

    metrics = run_online_episode(
        cfg=task.cfg,
        level_name=task.level_name,
        horizon=task.horizon,
        n_scenarios=task.n_scenarios,
        episode=task.episode,
        weather_sequence=task.weather_sequence,
        init_water=task.init_water,
        init_food=task.init_food,
        predictor_seed=task.predictor_seed,
        solver_threads=task.solver_threads,
    )
    return asdict(metrics)


def run_grid_benchmark(
    cfg: GameConfig,
    level_name: str,
    horizons: Iterable[int],
    scenario_counts: Iterable[int],
    episodes: int,
    base_seed: int,
    init_water: int,
    init_food: int,
    start_weather: int = Weather.SUNNY,
    workers: int = 1,
    solver_threads: int = 1,
) -> List[Dict[str, object]]:
    tasks: List[EpisodeTask] = []

    for horizon in horizons:
        for n_scenarios in scenario_counts:
            for episode in range(episodes):
                weather_rng = np.random.default_rng(
                    base_seed + episode * 97 + horizon * 13 + n_scenarios * 7
                )
                weather_seq = sample_weather_sequence(
                    transition=cfg.weather_transition,
                    num_days=cfg.num_days,
                    rng=weather_rng,
                    start_weather=start_weather,
                )
                tasks.append(
                    EpisodeTask(
                        cfg=cfg,
                        level_name=level_name,
                        horizon=horizon,
                        n_scenarios=n_scenarios,
                        episode=episode,
                        weather_sequence=weather_seq,
                        init_water=init_water,
                        init_food=init_food,
                        predictor_seed=base_seed + episode,
                        solver_threads=solver_threads,
                    )
                )

    records: List[Dict[str, object]] = []
    if workers <= 1:
        for task in tasks:
            records.append(_run_task(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for row in executor.map(_run_task, tasks):
                records.append(row)

    records.sort(
        key=lambda r: (
            str(r["level"]),
            int(r["horizon"]),
            int(r["n_scenarios"]),
            int(r["episode"]),
        )
    )
    return records


def aggregate_records(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, object]]] = {}
    for row in records:
        key = (row["level"], int(row["horizon"]), int(row["n_scenarios"]))
        grouped.setdefault(key, []).append(row)

    outputs: List[Dict[str, object]] = []
    for (level, horizon, n_scenarios), rows in grouped.items():
        success = np.array([1.0 if bool(r["success"]) else 0.0 for r in rows], dtype=float)
        final_money = np.array([float(r["final_money"]) for r in rows], dtype=float)
        avg_step = np.array([float(r["avg_step_time_ms"]) for r in rows], dtype=float)
        p95_step = np.array([float(r["p95_step_time_ms"]) for r in rows], dtype=float)
        infeasible = np.array([float(r["infeasible_rate"]) for r in rows], dtype=float)

        agg = AggregateMetrics(
            level=level,
            horizon=horizon,
            n_scenarios=n_scenarios,
            episodes=len(rows),
            success_rate=float(np.mean(success)),
            mean_final_money=float(np.mean(final_money)),
            std_final_money=float(np.std(final_money)),
            mean_avg_step_time_ms=float(np.mean(avg_step)),
            p95_step_time_ms=float(np.percentile(p95_step, 95)),
            mean_infeasible_rate=float(np.mean(infeasible)),
        )
        outputs.append(asdict(agg))

    outputs.sort(key=lambda x: (x["level"], x["horizon"], x["n_scenarios"]))
    return outputs
