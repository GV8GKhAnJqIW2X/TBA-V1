from onion.l2.l4_after_signal.g_modules_backtest import *

def g_choice_AS_add_modules(
    additional_modules_used,
    price_pos,
    price_last,
    side_pos,
    signal,
    qty,
    leverage,

):
    return {
        "avg": g_avg(
            price_pos,
            price_last,
            leverage,
            side_pos,
            signal,
            qty,
            avg_power=additional_modules_used["avg"],
        ),
    }