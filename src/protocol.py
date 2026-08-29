# Module purpose: Define the frozen development protocol shared by real-market experiments.

"""Shared defaults for the real-market development and final protocols."""

from dataclasses import dataclass


PAPER_MARKETS = ("DE_LU", "DK_1", "DK_2", "SE_3")
ENTSOE_MARKETS = ("DK_1", "DK_2", "SE_3", "DE_LU")
MARKET_NAME_MAP = {"DE_LU": "DE-LU", "DK_1": "DK1", "DK_2": "DK2", "SE_3": "SE3"}


# Store defaults shared by real-market experiments without overriding synthetic or theory settings.
@dataclass(frozen=True)
class DevelopmentProtocol:
    alpha: float = 0.10
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    random_state: int = 42
    rolling_window: int = 168


DEVELOPMENT_PROTOCOL = DevelopmentProtocol()


# Derive the stable market-specific seed used throughout the paper experiments.
def market_random_state(market: str, base_seed: int) -> int:
    return int(base_seed + sum(ord(character) for character in market))
