from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from env.config import Weather


@dataclass
class MarkovWeatherPredictor:
    transition: np.ndarray
    rng_seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.rng_seed)
        if self.transition.shape != (3, 3):
            raise ValueError("transition 必须是 3x3 矩阵")

    def sample_scenarios(
        self, current_weather: int, horizon: int, n_scenarios: int
    ) -> List[List[int]]:
        scenarios: List[List[int]] = []
        for _ in range(n_scenarios):
            state = current_weather
            seq = []
            for _ in range(horizon):
                state = int(self._rng.choice(3, p=self.transition[state]))
                seq.append(state)
            scenarios.append(seq)
        return scenarios


@dataclass
class DeterministicWeatherPredictor:
    weather_sequence: List[int]

    def sample_scenarios(
        self, current_weather: int, horizon: int, n_scenarios: int
    ) -> List[List[int]]:
        del current_weather
        seq = self.weather_sequence[:horizon]
        if len(seq) < horizon:
            seq = seq + [Weather.SUNNY] * (horizon - len(seq))
        return [list(seq) for _ in range(n_scenarios)]
