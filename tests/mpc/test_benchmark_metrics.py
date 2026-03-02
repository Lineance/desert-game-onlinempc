import pytest

from mpc.benchmark import aggregate_records


def test_aggregate_records_basic_math():
    rows = [
        {
            "level": "level3",
            "horizon": 2,
            "n_scenarios": 1,
            "episode": 0,
            "success": True,
            "final_money": 100.0,
            "end_day": 8,
            "avg_step_time_ms": 10.0,
            "p95_step_time_ms": 12.0,
            "infeasible_rate": 0.0,
        },
        {
            "level": "level3",
            "horizon": 2,
            "n_scenarios": 1,
            "episode": 1,
            "success": False,
            "final_money": 80.0,
            "end_day": 7,
            "avg_step_time_ms": 14.0,
            "p95_step_time_ms": 16.0,
            "infeasible_rate": 0.25,
        },
    ]

    out = aggregate_records(rows)
    assert len(out) == 1
    agg = out[0]

    assert agg["success_rate"] == pytest.approx(0.5)
    assert agg["mean_final_money"] == pytest.approx(90.0)
    assert agg["mean_avg_step_time_ms"] == pytest.approx(12.0)
    assert agg["mean_infeasible_rate"] == pytest.approx(0.125)
