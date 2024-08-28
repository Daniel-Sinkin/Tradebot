import datetime as dt
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

logging.getLogger(__name__)


@dataclass
class Paths_:
    DATA = Path("data")


class Symbol(StrEnum):
    EURUSD = "EURUSD"
    GBPUSD = "GBPUSD"
    USDCAD = "USDCAD"
    USDCHF = "USDCHF"
    USDJPY = "USDJPY"


class CandleTimeframe(StrEnum):
    M1 = "1m"
    M5 = "5m"
    H1 = "1h"
    D1 = "1d"

    def to_pandas_timeframe(self) -> str:
        match self:
            case CandleTimeframe.M1:
                return "1min"
            case CandleTimeframe.M5:
                return "5min"
            case CandleTimeframe.H1:
                return "1h"
            case CandleTimeframe.D1:
                return "1D"
            case _:
                raise NotImplementedError(
                    f"{self=} not yet supported for 'to_pandas_timeframe'."
                )

    @staticmethod
    def from_pandas_timeframe(pandas_timeframe: str) -> "CandleTimeframe":
        match pandas_timeframe:
            case "1min":
                return CandleTimeframe.M1
            case "5min":
                return CandleTimeframe.M5
            case "1h":
                return CandleTimeframe.H1
            case "1D":
                return CandleTimeframe.D1
            case _:
                raise ValueError(f"{pandas_timeframe=} is not supported!")
