from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pulp

from env.config import BASE_CONSUMPTION
from env.config import Weather as ConfigWeather
from env.utils import get_adjacency_matrix, get_neighbors, shortest_next_hop
from mpc.predictor import MarkovWeatherPredictor


@dataclass
class State:
    day: int
    pos: int
    water: int
    food: int
    money: float
    weather: int


@dataclass
class Action:
    day: int
    move_to: Optional[int]
    mine: bool
    buy_water: int
    buy_food: int
    expected_terminal_value: Optional[float] = None


@dataclass
class GameConfig:
    num_nodes: int
    num_days: int
    init_money: int
    weight_limit: int
    mine_income: int
    water_weight: int
    water_price: int
    food_weight: int
    food_price: int
    start: int
    end: int
    mines: List[int]
    villages: List[int]
    adj: np.ndarray
    neighbors: Dict[int, List[int]]
    weather_transition: np.ndarray
    base_consumption: Dict[int, Tuple[int, int]]

    @classmethod
    def from_level(
        cls,
        level_cfg,
        base_consumption: Optional[Dict[int, Tuple[int, int]]] = None,
    ) -> "GameConfig":
        adj = get_adjacency_matrix(level_cfg.NUM_NODES, level_cfg.EDGES)
        return cls(
            num_nodes=level_cfg.NUM_NODES,
            num_days=level_cfg.NUM_DAYS,
            init_money=level_cfg.INIT_MONEY,
            weight_limit=level_cfg.WEIGHT_LIMIT,
            mine_income=level_cfg.MINE_INCOME,
            water_weight=level_cfg.WATER_WEIGHT,
            water_price=level_cfg.WATER_PRICE_BASE,
            food_weight=level_cfg.FOOD_WEIGHT,
            food_price=level_cfg.FOOD_PRICE_BASE,
            start=level_cfg.START,
            end=level_cfg.END,
            mines=list(level_cfg.MINES),
            villages=list(level_cfg.VILLAGES),
            adj=adj,
            neighbors={i: get_neighbors(adj, i) for i in range(level_cfg.NUM_NODES)},
            weather_transition=np.array(level_cfg.WEATHER_TRANSITION, dtype=float),
            base_consumption=base_consumption or BASE_CONSUMPTION,
        )


class OnlineMPC:
    def __init__(
        self, cfg: GameConfig, horizon: int = 5, n_scenarios: int = 3, predictor=None
    ):
        self.cfg = cfg
        self.horizon = horizon
        self.n_scenarios = n_scenarios
        self.predictor = predictor or MarkovWeatherPredictor(cfg.weather_transition)

    def set_horizon(self, horizon: int) -> None:
        if horizon <= 0:
            raise ValueError("horizon 必须 > 0")
        self.horizon = horizon

    def solve(
        self, s: State, deterministic_weather: Optional[List[int]] = None
    ) -> Tuple[Action, Dict[str, object]]:
        cfg = self.cfg

        remaining_days = cfg.num_days - s.day
        H = min(self.horizon, remaining_days)
        if H <= 0:
            return Action(s.day, cfg.end, False, 0, 0, None), {
                "status": "terminal",
                "horizon": 0,
            }

        if deterministic_weather is not None:
            scen = list(deterministic_weather[:H])
            if len(scen) < H:
                scen.extend([s.weather] * (H - len(scen)))
            scenarios = [list(scen) for _ in range(self.n_scenarios)]
        else:
            scenarios = self.predictor.sample_scenarios(s.weather, H, self.n_scenarios)

        prob = pulp.LpProblem("Desert_MPC", pulp.LpMaximize)

        S = range(self.n_scenarios)
        T_rel = range(1, H + 1)
        T_all = range(H + 1)
        Nodes = range(cfg.num_nodes)

        loc = pulp.LpVariable.dicts("loc", (S, T_all, Nodes), cat="Binary")
        stay = pulp.LpVariable.dicts("stay", (S, T_rel, Nodes), cat="Binary")
        move = pulp.LpVariable.dicts("move", (S, T_rel, Nodes, Nodes), cat="Binary")
        mine = pulp.LpVariable.dicts("mine", (S, T_rel), cat="Binary")

        water = pulp.LpVariable.dicts("w", (S, T_all), lowBound=0, cat="Integer")
        food = pulp.LpVariable.dicts("f", (S, T_all), lowBound=0, cat="Integer")
        money = pulp.LpVariable.dicts("m", (S, T_all), lowBound=0, cat="Continuous")

        buy_w = pulp.LpVariable.dicts("bw", T_rel, lowBound=0, cat="Integer")
        buy_f = pulp.LpVariable.dicts("bf", T_rel, lowBound=0, cat="Integer")

        eff_w = pulp.LpVariable.dicts("ew", (S, T_rel), lowBound=0, cat="Continuous")
        eff_f = pulp.LpVariable.dicts("ef", (S, T_rel), lowBound=0, cat="Continuous")
        reached = pulp.LpVariable.dicts("r", (S, T_all), cat="Binary")

        for s_idx in S:
            for i in Nodes:
                prob += loc[s_idx][0][i] == (1 if i == s.pos else 0)
            prob += water[s_idx][0] == s.water
            prob += food[s_idx][0] == s.food
            prob += money[s_idx][0] == s.money
            prob += reached[s_idx][0] == (1 if s.pos == cfg.end else 0)

        if H == remaining_days:
            for s_idx in S:
                prob += reached[s_idx][H] == 1

        prob += (
            pulp.lpSum(
                money[s_idx][H]
                + 0.5 * cfg.water_price * water[s_idx][H]
                + 0.5 * cfg.food_price * food[s_idx][H]
                for s_idx in S
            )
            / self.n_scenarios
        )

        big_m = 50000

        for s_idx in S:
            for t_rel in T_rel:
                weather_type = scenarios[s_idx][t_rel - 1]
                base_w, base_f = cfg.base_consumption[weather_type]

                prob += pulp.lpSum(loc[s_idx][t_rel][i] for i in Nodes) == 1

                for j in Nodes:
                    flow_in = pulp.lpSum(
                        move[s_idx][t_rel][i][j] for i in cfg.neighbors[j]
                    )
                    prob += loc[s_idx][t_rel][j] == stay[s_idx][t_rel][j] + flow_in

                for i in Nodes:
                    flow_out = pulp.lpSum(
                        move[s_idx][t_rel][i][j] for j in cfg.neighbors[i]
                    )
                    prob += loc[s_idx][t_rel - 1][i] == stay[s_idx][t_rel][i] + flow_out

                total_stay = pulp.lpSum(stay[s_idx][t_rel][i] for i in Nodes)
                total_move = pulp.lpSum(
                    move[s_idx][t_rel][i][j] for i in Nodes for j in cfg.neighbors[i]
                )
                prob += total_stay + total_move == 1

                if weather_type == ConfigWeather.SANDSTORM:
                    prob += total_move == 0

                if cfg.mines:
                    prob += mine[s_idx][t_rel] <= pulp.lpSum(
                        stay[s_idx][t_rel][m] for m in cfg.mines
                    )
                else:
                    prob += mine[s_idx][t_rel] == 0

                if cfg.villages:
                    in_village = pulp.lpSum(loc[s_idx][t_rel][v] for v in cfg.villages)
                    prob += buy_w[t_rel] <= big_m * in_village
                    prob += buy_f[t_rel] <= big_m * in_village
                else:
                    prob += buy_w[t_rel] == 0
                    prob += buy_f[t_rel] == 0

                purchase_cost = (
                    2 * cfg.water_price * buy_w[t_rel]
                    + 2 * cfg.food_price * buy_f[t_rel]
                )
                prob += purchase_cost <= money[s_idx][t_rel - 1]

                prob += (
                    cfg.water_weight * water[s_idx][t_rel - 1]
                    + cfg.food_weight * food[s_idx][t_rel - 1]
                    <= cfg.weight_limit
                )

                if cfg.villages:
                    prob += cfg.water_weight * (
                        water[s_idx][t_rel - 1] + buy_w[t_rel]
                    ) + cfg.food_weight * (
                        food[s_idx][t_rel - 1] + buy_f[t_rel]
                    ) <= cfg.weight_limit + big_m * (
                        1 - pulp.lpSum(loc[s_idx][t_rel][v] for v in cfg.villages)
                    )

                cons_factor = total_stay + 2 * total_move + 2 * mine[s_idx][t_rel]
                water_cons = base_w * cons_factor
                food_cons = base_f * cons_factor

                r_prev = reached[s_idx][t_rel - 1]

                prob += eff_w[s_idx][t_rel] <= water_cons
                prob += eff_w[s_idx][t_rel] <= big_m * (1 - r_prev)
                prob += eff_w[s_idx][t_rel] >= water_cons - big_m * r_prev

                prob += eff_f[s_idx][t_rel] <= food_cons
                prob += eff_f[s_idx][t_rel] <= big_m * (1 - r_prev)
                prob += eff_f[s_idx][t_rel] >= food_cons - big_m * r_prev

                prob += water[s_idx][t_rel - 1] >= eff_w[s_idx][t_rel]
                prob += food[s_idx][t_rel - 1] >= eff_f[s_idx][t_rel]

                prob += (
                    water[s_idx][t_rel]
                    == water[s_idx][t_rel - 1] - eff_w[s_idx][t_rel] + buy_w[t_rel]
                )
                prob += (
                    food[s_idx][t_rel]
                    == food[s_idx][t_rel - 1] - eff_f[s_idx][t_rel] + buy_f[t_rel]
                )

                prob += (
                    cfg.water_weight * water[s_idx][t_rel]
                    + cfg.food_weight * food[s_idx][t_rel]
                    <= cfg.weight_limit
                )

                mine_revenue = cfg.mine_income * mine[s_idx][t_rel]
                prob += (
                    money[s_idx][t_rel]
                    == money[s_idx][t_rel - 1] + mine_revenue - purchase_cost
                )

                prob += reached[s_idx][t_rel] >= loc[s_idx][t_rel][cfg.end]
                prob += reached[s_idx][t_rel] >= reached[s_idx][t_rel - 1]
                prob += (
                    reached[s_idx][t_rel]
                    <= loc[s_idx][t_rel][cfg.end] + reached[s_idx][t_rel - 1]
                )

                for i in Nodes:
                    if i != cfg.end:
                        prob += loc[s_idx][t_rel][i] <= 1 - reached[s_idx][t_rel - 1]

                prob += mine[s_idx][t_rel] <= 1 - r_prev
                prob += buy_w[t_rel] <= big_m * (1 - r_prev)
                prob += buy_f[t_rel] <= big_m * (1 - r_prev)

        solver = pulp.HiGHS(msg=False, timeLimit=30)
        try:
            prob.solve(solver)
        except Exception:
            prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=30))

        if prob.status != pulp.LpStatusOptimal:
            return self._emergency_action(s), {"status": "infeasible", "horizon": H}

        s0 = 0
        t = 1
        move_to = None
        for i in Nodes:
            for j in cfg.neighbors[i]:
                if pulp.value(move[s0][t][i][j]) > 0.5:
                    move_to = j
                    break
            if move_to is not None:
                break

        is_mining = pulp.value(mine[s0][t]) > 0.5
        action = Action(
            day=s.day + 1,
            move_to=move_to,
            mine=bool(is_mining and move_to is None),
            buy_water=int(round(pulp.value(buy_w[t]))),
            buy_food=int(round(pulp.value(buy_f[t]))),
            expected_terminal_value=float(pulp.value(prob.objective)),
        )
        return action, {"status": pulp.LpStatus[prob.status], "horizon": H}

    def _emergency_action(self, s: State) -> Action:
        target = shortest_next_hop(self.cfg.adj, s.pos, self.cfg.end)
        return Action(
            day=s.day + 1,
            move_to=target,
            mine=False,
            buy_water=0,
            buy_food=0,
            expected_terminal_value=None,
        )
