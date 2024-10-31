from onion.l1.s_logging import logger
from onion.l1.g_settings import settings_ml, settings_bt

import numpy as np
import pandas as pd

# @logger.catch
async def g_data_backtest(
    data,
    klines_train_used_num=settings_ml["klines_train_used_num"],
    tp=settings_bt["tp"],
    sl=settings_bt["sl"],
    leverage=settings_bt["leverage"],
    avg_power=settings_bt["avg_power"],
    balance=settings_bt["balance"],
    balance_used=settings_bt["balance_used"],
):
    def g_zeroing_out(
        qty=4,
        add=(),
        in_=0
    ):
        return (*np.full(qty, in_), *add)

    def g_avg_module():
        # the modular system allows me to
        # use implicit modification of variable values
        nonlocal\
            price_open_avg,\
            price_last,\
            price_last,\
            side_pos,\
            side_predict,\
            qty,\
            data

        qty_new =(
            (abs(price_open_avg / price_last - 1)
            * leverage)
            * qty
            * avg_power
        ) \
            * side_pos * side_predict \
            + qty
        price_open_avg = (
            (price_open_avg * qty) +
            (price_last * (qty_new - qty))
        ) / qty_new
        qty = qty_new
        data.loc[i, "BT/ label"] = 2

    def g_sl_module():
        nonlocal\
            in_position,\
            balance_test,\
            qty,\
            data\

        balance_test -= qty
        (
            in_position,
            qty,
            data.loc[i, "BT/ label"]
        ) = g_zeroing_out(qty=3)

    def g_tp_module():
        nonlocal\
            in_position,\
            balance_test,\
            qty,\
            data\

        balance_test += qty * tp * leverage
        (
            in_position,
            qty,
            data.loc[i, "BT/ label"]
        ) = g_zeroing_out(qty=3)

    def g_close_module():
        nonlocal\
            in_position,\
            balance_test,\
            qty,\
            data\

        balance_test += qty * tp * leverage
        (
            in_position,
            qty,
            data.loc[i, "BT/ label"]
        ) = g_zeroing_out(qty=3)

    def g_open_module():
        nonlocal\
            in_position,\
            balance_test,\
            qty,\
            data,\
            price_open_avg,\
            side_pos

        in_position = 1
        data.loc[i, "BT/ label"] = in_position
        side_pos = side_predict
        price_open_avg = price_last
        qty = balance_test * balance_used

    data[["BT/ label", "BT/ balance"]] = np.nan

    (
        in_position,
        side_pos,
        price_open_avg,
        qty,
        balance_test,
    ) = g_zeroing_out(add=(balance,))

    for i, el in enumerate(
        data[["test", "close"]].iloc[klines_train_used_num:].values,
        start=klines_train_used_num
    ):
        side_predict, price_last = el
        data.loc[i, "BT/ balance"] = balance_test

        if in_position:
            pnl_percent = price_open_avg / price_last - 1
            {
                # sl module
                (
                    qty >= (balance_test * sl) and
                    pnl_percent * leverage <= (-1)
                ): g_sl_module,

                # tp module
                all((abs(pnl_percent) >= tp, (
                    (pnl_percent > 0 > side_pos) or
                    (side_pos > 0 > pnl_percent)
                ))): g_tp_module,

                # avg module
                (
                    side_predict != 0 and
                    not np.isnan(side_predict) and
                    qty < balance_test * sl
                ): g_avg_module,

                # close module
                qty <= 0: g_close_module,
            }.get(True, lambda: 0)()
        elif (
            side_predict and
            not np.isnan(side_predict)
        ):
            g_open_module()
    
    return data