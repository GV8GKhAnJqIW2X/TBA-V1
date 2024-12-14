from onion.l1.l1.g_settings_ import *
from onion.l2.l4_after_signal.g_modules_backtest import *

def g_backtest_AS_balance_A_single_A_iter(
    signal,
    price_last,
    return_values,
    leverage=settings["BACKTEST"]["leverage"],
    tp=settings["BACKTEST"]["tp_pos"],
    sl=settings["BACKTEST"]["sl_balance"],
    open=None,
    avg_power=settings["BACKTEST"]["avg"]["avg_multilple_my_side"],
    check_args=True,
):
    # check args
    if check_args:

        
        if return_values is None:
            return_values = (
                False,
                settings["BACKTEST"]["balance_usd"],
                0,
                0,
                0,
            )
    
    # init
    (
        in_position,
        balance,
        signal_pos,
        price_pos,
        qty,
    ) = return_values
    
    # main
    if in_position:
        price_pos, qty = g_avg()
        pnl_percent = (price_last / price_pos - 1) * signal_pos * leverage
        pnl_unrealized = pnl_percent * qty
        
        # sl/tp
        if (
            pnl_unrealized <= -(sl["qty_from_balance_threshold_percent"] * balance)
            or pnl_percent >= tp["pnl_threshold"]
        ):
            balance += pnl_unrealized
            return (
                False,
                balance,
                0,
                0,
                0,
            )

        # avg
        if signal:
            qty_old = qty
            (
                price_pos,
                qty,
            ) = g_avg(
                price_pos,
                price_last,
                signal_pos,
                signal,
                qty,
                leverage,
                avg_power=avg_power,
            )
            if qty_old > qty:
                pnl_percent = (price_last / price_pos - 1) * signal_pos * leverage
                pnl_unrealized_old = pnl_unrealized
                pnl_unrealized = pnl_percent * qty
                balance = balance + pnl_unrealized - pnl_unrealized_old
            return (
                in_position,
                balance,
                signal_pos,
                price_pos,
                qty,
            )

    elif signal:
        return (
            True,
            balance,
            signal,
            price_last,
            balance * open["qty_from_balance_percent"],
        )
        
    return (
        balance, 
        (
            in_position,
            balance,
            signal_pos,
            price_pos,
            qty,
        ),
    )

def g_backtest_AS_balance_A_comparison_A_single_A_iter(
    return_values,
    signal,
    signal2,
    price_last,
    leverage=settings["BACKTEST"]["leverage"],
    qty_balance_used_open=settings["BACKTEST"]["qty_balance_used_open"],
    tp_pos=settings["BACKTEST"]["tp_pos"],
    sl_balance = settings["BACKTEST"]["sl_balance"],
    avg_multilple_my_side = settings["BACKTEST"]["avg"]["avg_multilple_my_side"],
    avg_multilple_no_my_side = settings["BACKTEST"]["avg"]["avg_multilple_no_my_side"],
    tax_exchange=settings["BACKTEST"]["tax_exchange"],
    check_args=True,
):
    # check args
    if check_args:
        if return_values is None:
            return_values = (
                False,
                0,
                0,
                0,
                settings["BACKTEST"]["balance_usd"]
            )
    
    # init
    (
        in_position,
        price_pos,
        signal_pos,
        qty,
        balance,
    ) = return_values
    closed_return_values = lambda b: (
        False,
        0,
        0,
        0,
        b,
    )

    # main
    if in_position:
        pnl_unrealized_AS_prcnt = (price_last / price_pos - 1) * signal_pos * leverage
        pnl_unrealized_AS_qty = pnl_unrealized_AS_prcnt * qty
        # if 16000 < i < 17000:
        #     print(
        #         i, 
        #         round(pnl_unrealized_AS_prcnt, 2), 
        #         round(pnl_unrealized_AS_qty, 2), 
        #         sep=" | ",
        #     )

        # sl/tp
        if (
            pnl_unrealized_AS_qty <= balance * sl_balance
            or pnl_unrealized_AS_prcnt >= tp_pos
        ):
            return closed_return_values(balance + pnl_unrealized_AS_qty - qty * tax_exchange * leverage)
        elif signal2:
            return g_avg(
                pnl_unrealized_AS_qty,
                pnl_unrealized_AS_prcnt,
                price_last,
                signal2,
                avg_multilple_no_my_side,
                avg_multilple_my_side,
                in_position,
                price_pos,
                signal_pos,
                qty,
                balance,    
                tax_exchange * leverage,
            )
    elif signal:
        return (
            True,
            price_last,
            signal,
            balance * qty_balance_used_open,
            balance,
        )
    
    return return_values
    
