from project_exctentions.g_utils import *
from onion.l1.l2.g_structures import *
from onion.l2.l1_features.g_y_ml import *

import pandas as pd

def g_x_y_arrays_partial_filling_A_comparison_A_klines_train_held(
    open,
    high,
    low,
    close,
    y_train_array_fill_target,
    x_train_arrays_fill_target,
    klines_train,
    klines_train_held,
    max_window_features,
    features_used,
    tp_train=settings["SIGNAL_GENERATION"]["ML"]["tp_train"],
    USE_shift_filling=False,
    USE_ready_made_series=False,
    ready_made_series=None,
    check_params=True,
):
    """
    подразумевается что y_train_array заполнен равномерно

    src (open, high, low, close) len == klines_train + max_window_features * 2 + 1
    
    """
    # check params
    if check_params:
        if not isinstance(open, pd.Series):
            open = pd.Series(open)
        if not isinstance(high, pd.Series):
            high = pd.Series(high)
        if not isinstance(low, pd.Series):
            low = pd.Series(low)
        if not isinstance(close, pd.Series):
            close = pd.Series(close)
        
        if not isinstance(klines_train, int):
            raise ValueError("Klines train is not int")
        if not isinstance(klines_train_held, int):
            raise ValueError("Klines train held is not int")
        if not isinstance(max_window_features, int):
            raise ValueError("Max window features is not int")
        
        ## additional check
        if len(open) < klines_train + max_window_features * 5 + 1:
            raise ValueError("There are not enough values to generate x_train_arrays")
    
    # init
    number_need_to_filled = g_number_need_to_filled(y_train_array_fill_target)
    
    if USE_shift_filling:
        if number_need_to_filled == 0:
            y_train_array_fill_target = np.roll(y_train_array_fill_target, -1)
            x_train_arrays_fill_target = {
                key: np.roll(x, -1) 
                for key, x in x_train_arrays_fill_target.items()
            }
            number_need_to_filled = 1
    
    # main
    for i in range(klines_train):
        if i < klines_train - number_need_to_filled:
            continue

        i_src = i + max_window_features * 5 + 1
        y_train_array_fill_target[i] = g_y_train_signal_A_comparison_A_klines_train_held(
            last_price=close[i_src],
            last_price_MI_klines_train_held=close[i_src - klines_train_held],
            tp_train=tp_train,
        )
        x_features_series = ready_made_series
        if not USE_ready_made_series:
            x_features_series = g_x_features_series(
                *g_iloc((open, high, low, close), slice(i, i_src)),
                features_used=features_used,
            )
        for key in features_used:
            x_train_arrays_fill_target[key][i] = x_features_series[key]

        # print
        if number_need_to_filled > 1:
            print(f"\r{i + 1}/{klines_train} init train arrays", end="")
            if i + 1 == klines_train:
                print()
                print("Train arrays initialized! The backtest has started...")
                print()

    return x_train_arrays_fill_target, y_train_array_fill_target