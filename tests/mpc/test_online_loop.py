from hypothesis import given, settings
from hypothesis import strategies as st

from env.config import Weather
from mpc.core import OnlineMPC
from mpc.simulator import run_online_simulation


@given(
    init_water=st.integers(min_value=80, max_value=220),
    init_food=st.integers(min_value=80, max_value=220),
    horizon=st.integers(min_value=2, max_value=6),
)
@settings(max_examples=20, deadline=None)
def test_online_loop_fuzz_level3(level3_cfg, init_water, init_food, horizon):
    solver = OnlineMPC(level3_cfg, horizon=horizon, n_scenarios=2)
    weather_seq = [Weather.SUNNY if i % 2 == 0 else Weather.HOT for i in range(level3_cfg.num_days)]

    result = run_online_simulation(
        cfg=level3_cfg,
        mpc=solver,
        weather_sequence=weather_seq,
        init_water=init_water,
        init_food=init_food,
    )

    assert result.final_state.day <= level3_cfg.num_days
    if result.final_state.water >= 0 and result.final_state.food >= 0:
        assert result.final_state.money >= 0


def test_online_loop_end_to_end(level4_cfg, weather_level4):
    solver = OnlineMPC(level4_cfg, horizon=8, n_scenarios=3)
    result = run_online_simulation(
        cfg=level4_cfg,
        mpc=solver,
        weather_sequence=weather_level4,
        init_water=200,
        init_food=200,
    )

    assert result.final_state.day <= level4_cfg.num_days
    assert len(result.trace) >= 1
