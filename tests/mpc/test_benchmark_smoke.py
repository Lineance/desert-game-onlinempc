from mpc.benchmark import run_grid_benchmark


def test_benchmark_smoke_level3(level3_cfg):
    records = run_grid_benchmark(
        cfg=level3_cfg,
        level_name="level3",
        horizons=[2],
        scenario_counts=[1],
        episodes=1,
        base_seed=42,
        init_water=120,
        init_food=120,
    )

    assert len(records) == 1
    row = records[0]
    for key in [
        "level",
        "horizon",
        "n_scenarios",
        "success",
        "final_money",
        "avg_step_time_ms",
        "p95_step_time_ms",
        "infeasible_rate",
    ]:
        assert key in row
