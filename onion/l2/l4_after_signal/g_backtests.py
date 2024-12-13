from onion.l1.l1.g_settings_ import *
from onion.l2.l4_after_signal.g_modules_backtest import *

def g_backtest_AS_balance_A_single_A_iter(
    signal,
    price_last,
    return_values,
    leverage=settings["BACKTEST"]["leverage"],
    tp=settings["BACKTEST"]["module_can_used"]["tp"],
    sl=settings["BACKTEST"]["module_can_used"]["sl"],
    open=settings["BACKTEST"]["module_can_used"]["open"],
    avg_power=settings["BACKTEST"]["module_can_used"]["avg"]["avg_power"],
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
    qty_balance_used_open=0.01,
    tp_pos=0.17,
    sl_balance = -0.2,
    avg_multilple_my_side = 2,
    avg_multilple_no_my_side = 0.5,
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

        # sl/tp
        if (
            pnl_unrealized_AS_qty <= balance * sl_balance
            or pnl_unrealized_AS_prcnt >= tp_pos
        ):
            return closed_return_values(balance + pnl_unrealized_AS_qty)
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
    
