import numpy as np

from env.config import Weather
from mpc.predictor import MarkovWeatherPredictor


def test_markov_predictor_shape_and_value_range(level3_cfg):
    predictor = MarkovWeatherPredictor(level3_cfg.weather_transition, rng_seed=42)
    scenarios = predictor.sample_scenarios(
        current_weather=Weather.SUNNY, horizon=5, n_scenarios=4
    )

    assert len(scenarios) == 4
    for seq in scenarios:
        assert len(seq) == 5
        assert all(w in (0, 1, 2) for w in seq)


def test_markov_predictor_reproducible(level4_cfg):
    p1 = MarkovWeatherPredictor(level4_cfg.weather_transition, rng_seed=7)
    p2 = MarkovWeatherPredictor(level4_cfg.weather_transition, rng_seed=7)

    assert p1.sample_scenarios(
        Weather.HOT, horizon=6, n_scenarios=3
    ) == p2.sample_scenarios(Weather.HOT, horizon=6, n_scenarios=3)


def test_transition_matrix_must_be_3x3():
    bad = np.array([[1.0]])
    try:
        MarkovWeatherPredictor(bad)
    except ValueError:
        assert True
    else:
        assert False
