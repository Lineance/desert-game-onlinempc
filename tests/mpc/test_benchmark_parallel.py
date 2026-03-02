import pytest

from mpc.benchmark import run_grid_benchmark


@pytest.mark.parametrize("workers", [1, 2])
def test_benchmark_parallel_runs(level3_cfg, workers):
    records = run_grid_benchmark(
        cfg=level3_cfg,
        level_name="level3",
        horizons=[2],
        scenario_counts=[1],
        episodes=2,
        base_seed=42,
        init_water=120,
        init_food=120,
        workers=workers,
        solver_threads=1,
    )
    assert len(records) == 2


def test_benchmark_parallel_consistency(level3_cfg):
    serial = run_grid_benchmark(
        cfg=level3_cfg,
        level_name="level3",
        horizons=[2],
        scenario_counts=[1],
        episodes=2,
        base_seed=42,
        init_water=120,
        init_food=120,
        workers=1,
        solver_threads=1,
    )

    parallel = run_grid_benchmark(
        cfg=level3_cfg,
        level_name="level3",
        horizons=[2],
        scenario_counts=[1],
        episodes=2,
        base_seed=42,
        init_water=120,
        init_food=120,
        workers=2,
        solver_threads=1,
    )

    key_fields = [
        "level",
        "horizon",
        "n_scenarios",
        "episode",
        "success",
        "final_money",
        "end_day",
    ]
    for s, p in zip(serial, parallel):
        for key in key_fields:
            if key == "final_money":
                assert p[key] == pytest.approx(s[key], abs=1e-6)
            else:
                assert p[key] == s[key]
