import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from env.config import Level3Config, Level4Config, Weather
from mpc.core import GameConfig


@pytest.fixture(scope="session")
def level3_cfg() -> GameConfig:
    return GameConfig.from_level(Level3Config)


@pytest.fixture(scope="session")
def level4_cfg() -> GameConfig:
    return GameConfig.from_level(Level4Config)


@pytest.fixture(scope="session")
def weather_level3():
    return [
        Weather.SUNNY,
        Weather.HOT,
        Weather.SUNNY,
        Weather.HOT,
        Weather.SUNNY,
        Weather.HOT,
        Weather.SUNNY,
        Weather.HOT,
        Weather.SUNNY,
        Weather.HOT,
    ]


@pytest.fixture(scope="session")
def weather_level4():
    return [Weather.SUNNY] * 30
