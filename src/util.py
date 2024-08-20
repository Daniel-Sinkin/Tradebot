from typing import Optional

import pandas as pd


def build_candle(
    prices: pd.Series,
    timeframe: str,
    volumes: Optional[float | pd.Series] = None,
    include_tick_volumes: bool = False,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Constructs a DataFrame of OHLC (Open, High, Low, Close) candles from price data.

    ### Parameters:
    * prices : pd.Series
        * A pandas Series of price data, indexed by datetime.
    * timeframe : str
        * A string representing the resampling timeframe (e.g., '1T' for 1 minute, '1H' for 1 hour).
    * volumes : Optional[float | pd.Series], optional
        * Either a single float representing the volume for each period or a pandas Series of volumes, indexed by datetime.
        * If None, volume data will not be included.
    * include_tick_volumes : bool, optional
        * If True, includes a column for tick volumes (i.e., the count of trades per resampled period). Default is False.
    * dropna : bool, optional
        * If True, drops any NaN values from the resulting DataFrame. Default is True.

    ### Returns:
    * pd.DataFrame
        * A DataFrame with columns for Open, High, Low, Close prices, and optionally Volume and TickVolume.

    ### Raises:
    * TypeError
        * If the provided volumes argument is neither a float nor a pd.Series.
    """
    resampled = prices.resample(timeframe)
    candles = resampled.ohlc()

    if include_tick_volumes:
        candles["TickVolume"] = resampled.count()

    if volumes is not None:
        if isinstance(volumes, float):
            candles["Volume"] = volumes
        elif isinstance(volumes, pd.Series):
            assert len(volumes) == len(
                candles
            ), "Volume series must match the length of the candles DataFrame."
            candles["Volume"] = volumes
        else:
            raise TypeError(f"Invalid type for volumes: {type(volumes)=}.")

    # Rename columns using a dictionary-based approach
    candles.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"},
        inplace=True,
    )

    if dropna:
        candles.dropna(inplace=True)

    return candles


def loc_sorted(self):
    pass
