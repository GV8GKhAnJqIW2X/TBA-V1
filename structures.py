import pandas as pd

class Source():
    def __init__(
        self, 
        open,
        high,
        low,
        close,
    ):
        self.open = pd.Series(open)
        self.high = pd.Series(high)
        self.low = pd.Series(low)
        self.close = pd.Series(close)
    
    # impl
    def iloc(
        self, 
        start, 
        end=None,
        step=None,
    ):
        return Source(
            self.open.iloc[start:end:step],
            self.high.iloc[start:end:step],
            self.low.iloc[start:end:step],
            self.close.iloc[start:end:step],
        )
    
    # impl
    def g_ohlc4(self):
        return (self.open + self.high + self.low + self.close) / 4

class Label():
    def __init__(
        self, 
        short: int, 
        long: int, 
        neutral: int,
    ):
        self.long = long
        self.short = short
        self.neutral = neutral

class Feature_Arrays():
    def __init__(self, *arrays: tuple[pd.Series]): # добавление новых элементов путем начальной инициализации массива с фиксированной длинной 
        (
            self.f1, 
            self.f2, 
            self.f3, 
            self.f4, 
            self.f5,
        ) = arrays

class Feature_Series():
    def __init__(self, *series: tuple[float]):
        (
            self.f1, 
            self.f2, 
            self.f3, 
            self.f4, 
            self.f5,
        ) = series

class ML_Model():
    def __init__(
        self, 
        y_train_array: pd.Series[int],
        distances: list,
        predictions: list,
        signals: list,
        is_EMA_up_trends: list,
        is_EMA_down_trends: list,
        is_SMA_up_trends: list,
        is_SMA_down_trends: list,
        last_distance: float,
        prediction: int | float,
        signal: int | float,
        bars_held: int,
    ):
        self.y_train_array = y_train_array
        self.distances = distances
        self.predictions = predictions
        self.signals = signals
        self.is_EMA_up_trends = is_EMA_up_trends
        self.is_EMA_down_trends = is_EMA_down_trends
        self.is_SMA_up_trends = is_SMA_up_trends
        self.is_SMA_down_trends = is_SMA_down_trends
        self.last_distance = last_distance
        self.prediction = prediction
        self.signal = signal
        self.bars_held = bars_held

class Filter():
    def __init__(self, *filters: tuple[bool]):
        (
            self.f1,
            self.f2, 
            self.f3, 
        ) = filters

class Indicators():
    def __init__(
        self, 
        ADX: float,
        CCI: float,
        WT: float,
        TSI: float,
        RSI: float,
    ):
        self.ADX = ADX
        self.CCI = CCI
        self.WT = WT
        self.TSI = TSI
        self.RSI = RSI

class Bars_Count():
    def __init__(
        self, 
        bars_used: int,
        max_period: int,
        max_bars_back: int,
    ):
        self.bars_used = bars_used
        self.max_period = max_period
        self.max_bars_back = max_bars_back