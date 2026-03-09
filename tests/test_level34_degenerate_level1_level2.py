import pytest

from mpc.validator import compare_solver_with_tool_reference

EXPECTED_OBJECTIVES = {
    "level1-solved": 10470.0,
    "level2-solved": 12730.0,
}


def test_level34_degenerate_matches_level1_reference_objective() -> None:
    report = compare_solver_with_tool_reference("level1-solved", time_limit=180)

    assert report.tool_status in {"Optimal", "Not Solved", "Undefined"}
    assert report.tool_objective == pytest.approx(EXPECTED_OBJECTIVES["level1-solved"], abs=1e-3)
    assert report.mpc_status in {"Optimal", "NotSolved"}
    assert report.objective_gap >= -1e-3


def test_level34_degenerate_matches_level2_reference_objective() -> None:
    report = compare_solver_with_tool_reference("level2-solved", time_limit=180)

    assert report.tool_status in {"Optimal", "Not Solved", "Undefined"}
    assert report.tool_objective == pytest.approx(EXPECTED_OBJECTIVES["level2-solved"], abs=1e-3)
    assert report.mpc_status in {"Optimal", "NotSolved"}
    assert report.objective_gap >= -1e-3
