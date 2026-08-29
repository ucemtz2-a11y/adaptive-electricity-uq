# Keep the market names, seeds, and main experiment settings in one place.

"""Shared defaults for the real-market development and final protocols."""

from dataclasses import dataclass


PAPER_MARKETS = ("DE_LU", "DK_1", "DK_2", "SE_3")
ENTSOE_MARKETS = ("DK_1", "DK_2", "SE_3", "DE_LU")
MARKET_NAME_MAP = {"DE_LU": "DE-LU", "DK_1": "DK1", "DK_2": "DK2", "SE_3": "SE3"}


# These defaults are shared by the real-market runs only; the simulations use their own settings.
@dataclass(frozen=True)
class DevelopmentProtocol:
    alpha: float = 0.10
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    random_state: int = 42
    rolling_window: int = 168


DEVELOPMENT_PROTOCOL = DevelopmentProtocol()


# Turn each market name into the same seed offset every time the experiment is run.
def market_random_state(market: str, base_seed: int) -> int:
    return int(base_seed + sum(ord(character) for character in market))
