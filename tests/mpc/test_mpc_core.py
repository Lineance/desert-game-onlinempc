from env.config import Weather
from mpc.core import OnlineMPC, State


def test_mpc_returns_valid_action(level3_cfg):
    solver = OnlineMPC(level3_cfg, horizon=4, n_scenarios=2)
    state = State(
        day=0,
        pos=level3_cfg.start,
        water=120,
        food=120,
        money=5000,
        weather=Weather.SUNNY,
    )

    action, info = solver.solve(state, deterministic_weather=[Weather.SUNNY] * 4)

    assert info["status"] in ("Optimal", "infeasible")
    assert action.day == 1
    assert action.buy_water >= 0
    assert action.buy_food >= 0


def test_horizon_adjustable(level4_cfg):
    solver = OnlineMPC(level4_cfg, horizon=6, n_scenarios=2)
    solver.set_horizon(3)

    state = State(
        day=1,
        pos=level4_cfg.start,
        water=150,
        food=150,
        money=4000,
        weather=Weather.HOT,
    )

    _, info = solver.solve(state, deterministic_weather=[Weather.SUNNY] * 10)
    assert info["horizon"] == 3
