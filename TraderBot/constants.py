import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


@dataclass
class _Paths:
    ROOT = Path(__file__).resolve().parent
    # Find the root of the module if the parent is not it, this assumes that there
    # is exactly one setup.py in the entire module.
    while not ROOT.joinpath("setup.py").exists() and ROOT != ROOT.parent:
        ROOT = ROOT.parent

    DATA = ROOT.joinpath("data")
    LOGS = ROOT.joinpath("logs")


logging.getLogger(__name__)

_symbol_pretty_mapping: dict[str, str] = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDCAD": "USD/CAD",
    "USDCHF": "USD/CHF",
    "USDJPY": "USD/JPY",
}


class _Symbol(StrEnum):
    EURUSD = "EURUSD"
    GBPUSD = "GBPUSD"
    USDCAD = "USDCAD"
    USDCHF = "USDCHF"
    USDJPY = "USDJPY"

    def pretty_format(self) -> str:
        """Returns the pretty format of the symbol."""
        return _symbol_pretty_mapping[self.value]


class _CandleTimeframe(StrEnum):
    M1 = "1m"
    M5 = "5m"
    H1 = "1h"
    D1 = "1d"

    def to_pandas_timeframe(self) -> str:
        match self:
            case _CandleTimeframe.M1:
                return "1min"
            case _CandleTimeframe.M5:
                return "5min"
            case _CandleTimeframe.H1:
                return "1h"
            case _CandleTimeframe.D1:
                return "1D"
            case _:
                raise NotImplementedError(
                    f"{self=} not yet supported for 'to_pandas_timeframe'."
                )

    @staticmethod
    def from_pandas_timeframe(pandas_timeframe: str) -> "_CandleTimeframe":
        match pandas_timeframe:
            case "1min":
                return _CandleTimeframe.M1
            case "5min":
                return _CandleTimeframe.M5
            case "1h":
                return _CandleTimeframe.H1
            case "1D":
                return _CandleTimeframe.D1
            case _:
                raise ValueError(f"{pandas_timeframe=} is not supported!")
