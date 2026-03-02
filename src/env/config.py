from typing import List, Tuple

import numpy as np


class Weather:
    SUNNY = 0
    HOT = 1
    SANDSTORM = 2

    NAMES = ["晴朗", "高温", "沙暴"]


BASE_CONSUMPTION = {
    Weather.SUNNY: (3, 4),
    Weather.HOT: (9, 9),
    Weather.SANDSTORM: (10, 10),
}


class Level3Config:
    NUM_NODES = 13
    NUM_DAYS = 10
    INIT_MONEY = 10000
    WEIGHT_LIMIT = 1200
    MINE_INCOME = 200

    WATER_WEIGHT = 3
    WATER_PRICE_BASE = 5
    FOOD_WEIGHT = 2
    FOOD_PRICE_BASE = 10

    START = 0
    END = 12
    MINES: List[int] = [8]
    VILLAGES: List[int] = []

    EDGES: List[Tuple[int, int]] = [
        (1, 5),
        (1, 2),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
        (3, 8),
        (3, 9),
        (4, 5),
        (4, 6),
        (4, 7),
        (5, 6),
        (6, 7),
        (6, 12),
        (6, 13),
        (7, 11),
        (7, 12),
        (8, 9),
        (9, 10),
        (9, 11),
        (10, 11),
        (10, 13),
        (11, 12),
        (11, 13),
        (12, 13),
    ]

    WEATHER_MODES = {
        "no_sandstorm": [0.5, 0.5, 0.0],
        "sunny_bias": [0.7, 0.3, 0.0],
        "hot_bias": [0.3, 0.7, 0.0],
    }

    WEATHER_TRANSITION = np.array(
        [
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )


class Level4Config:
    NUM_NODES = 25
    NUM_DAYS = 30
    INIT_MONEY = 10000
    WEIGHT_LIMIT = 1200
    MINE_INCOME = 1000

    WATER_WEIGHT = 3
    WATER_PRICE_BASE = 5
    FOOD_WEIGHT = 2
    FOOD_PRICE_BASE = 10

    START = 0
    END = 24
    MINES: List[int] = [17]
    VILLAGES: List[int] = [13]

    EDGES: List[Tuple[int, int]] = [
        (1, 2),
        (1, 6),
        (2, 3),
        (2, 7),
        (3, 4),
        (3, 8),
        (4, 5),
        (4, 9),
        (5, 10),
        (6, 7),
        (6, 11),
        (7, 8),
        (7, 12),
        (8, 9),
        (8, 13),
        (9, 10),
        (9, 14),
        (10, 15),
        (11, 12),
        (11, 16),
        (12, 13),
        (12, 17),
        (13, 14),
        (13, 18),
        (14, 15),
        (14, 19),
        (15, 20),
        (16, 17),
        (16, 21),
        (17, 18),
        (17, 22),
        (18, 19),
        (18, 23),
        (19, 20),
        (19, 24),
        (20, 25),
    ]

    WEATHER_MODES = {
        "balanced": [0.45, 0.45, 0.1],
    }

    WEATHER_TRANSITION = np.array(
        [
            [0.25, 0.5, 0.25],
            [0.4, 0.39, 0.21],
            [0.33, 0.5, 0.17],
        ]
    )
