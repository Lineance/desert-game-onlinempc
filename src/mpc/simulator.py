from dataclasses import dataclass
from typing import Dict, List, Optional

from env.config import Weather
from mpc.core import Action, GameConfig, OnlineMPC, State


@dataclass
class SimulationResult:
    success: bool
    final_state: State
    trace: List[Dict[str, object]]


def apply_action(
    cfg: GameConfig, state: State, action: Action, weather_today: int
) -> State:
    next_pos = state.pos
    moved = False
    if (
        action.move_to is not None
        and action.move_to != state.pos
        and weather_today != Weather.SANDSTORM
    ):
        if action.move_to in cfg.neighbors[state.pos]:
            next_pos = action.move_to
            moved = True

    staying = not moved
    can_mine = staying and next_pos in cfg.mines
    mine = can_mine and action.mine

    cons_factor = 1 + (1 if moved else 0) + (2 if mine else 0)
    base_w, base_f = cfg.base_consumption[weather_today]
    water_cons = base_w * cons_factor
    food_cons = base_f * cons_factor

    if state.water < water_cons or state.food < food_cons:
        return State(
            day=state.day + 1,
            pos=next_pos,
            water=-1,
            food=-1,
            money=state.money,
            weather=weather_today,
        )

    buy_w = max(0, int(action.buy_water))
    buy_f = max(0, int(action.buy_food))

    can_buy = next_pos in cfg.villages
    if not can_buy:
        buy_w = 0
        buy_f = 0

    water_after = state.water - water_cons + buy_w
    food_after = state.food - food_cons + buy_f
    weight = cfg.water_weight * water_after + cfg.food_weight * food_after
    if weight > cfg.weight_limit:
        water_after = state.water - water_cons
        food_after = state.food - food_cons
        buy_w = 0
        buy_f = 0

    buy_cost = 2 * cfg.water_price * buy_w + 2 * cfg.food_price * buy_f
    if buy_cost > state.money:
        buy_w = 0
        buy_f = 0
        buy_cost = 0
        water_after = state.water - water_cons
        food_after = state.food - food_cons

    mine_gain = cfg.mine_income if mine else 0

    return State(
        day=state.day + 1,
        pos=next_pos,
        water=water_after,
        food=food_after,
        money=state.money - buy_cost + mine_gain,
        weather=weather_today,
    )


def run_online_simulation(
    cfg: GameConfig,
    mpc: OnlineMPC,
    weather_sequence: List[int],
    init_water: int,
    init_food: int,
    horizon_schedule: Optional[Dict[int, int]] = None,
) -> SimulationResult:
    init_cost = cfg.water_price * init_water + cfg.food_price * init_food
    init_money = cfg.init_money - init_cost
    state = State(
        day=0,
        pos=cfg.start,
        water=init_water,
        food=init_food,
        money=init_money,
        weather=weather_sequence[0],
    )

    trace: List[Dict[str, object]] = [
        {
            "day": 0,
            "pos": state.pos,
            "water": state.water,
            "food": state.food,
            "money": state.money,
            "weather": state.weather,
            "action": None,
        }
    ]

    for day in range(1, cfg.num_days + 1):
        if horizon_schedule and day in horizon_schedule:
            mpc.set_horizon(horizon_schedule[day])

        remaining_weather = weather_sequence[day - 1 :]
        action, _ = mpc.solve(state, deterministic_weather=remaining_weather)
        weather_today = weather_sequence[day - 1]

        next_state = apply_action(cfg, state, action, weather_today)
        trace.append(
            {
                "day": day,
                "pos": next_state.pos,
                "water": next_state.water,
                "food": next_state.food,
                "money": next_state.money,
                "weather": weather_today,
                "action": action,
            }
        )

        state = next_state
        if state.water < 0 or state.food < 0:
            return SimulationResult(False, state, trace)
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
            return SimulationResult(True, state, trace)

    success = state.pos == cfg.end
    return SimulationResult(success, state, trace)
