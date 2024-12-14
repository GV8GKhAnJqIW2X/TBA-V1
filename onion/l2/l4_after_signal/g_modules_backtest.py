from onion.l1.l1.g_settings_ import *

def g_sl(
    balance,
    qty,
    in_position,
    leverage,
    price_pos,
    pnl_percent,
    qty_from_balance_threshold_percent=None,
    check_args=True,
):
    # check args
    if check_args:
        if not isinstance(balance, (float, int)):
            raise ValueError("balance must be a float or int")
        if not isinstance(qty, (int, float)):
            raise ValueError("qty must be an int or float")
        if not isinstance(in_position, bool):
            raise ValueError("in_position must be a boolean")
        if not isinstance(leverage, (float, int)):
            raise ValueError("leverage must be an int or float")
    
    # main
    return (
        price_pos,
        0,
        balance - qty * pnl_percent,
        False,
        0,
    )
def g_tp(
    balance,
    qty,
    in_position,
    leverage,
    price_pos,
    pnl_percent,
    check_args=True,
):
    # check args
    if check_args:
        if not isinstance(balance, (float, int)):
            raise ValueError("balance must be a float or int")
        if not isinstance(qty, (int, float)):
            raise ValueError("qty must be an int or float")
        if not isinstance(in_position, bool):
            raise ValueError("in_position must be a boolean")
        if not isinstance(leverage, (float, int)):
            raise ValueError("leverage must be an int or float")
    
    # main
    return (
        price_pos,
        0,
        balance + qty * pnl_percent,
        False,
        0,
    )

def g_open(
    price_last,
    balance,
    balance_used_min_percent,
    signal,
    check_args=True,
):
    # check args
    if check_args:
        if not isinstance(price_last, (float, int)):
            raise ValueError("price_last must be a float or int")
        if not isinstance(balance, (float, int)):
            raise ValueError("balance must be a float or int")
        if not isinstance(balance_used_min_percent, (float, int)):
            raise ValueError("balance_used_min_percent must be a float or int")
    
    # main
    return (
        True,
        signal,
        price_last,
        balance * balance_used_min_percent,
    )

def g_close(
    balance,
    qty,
    pnl_percent,
):
    return (
        False,
        0,
        0,
        0,
    )

def g_avg(
    pnl_unrealized_AS_qty,
    pnl_unrealized_AS_prcnt,
    price_last,
    signal,
    avg_multiplier_no_my_side,
    avg_multiplier_my_side,
    in_position,
    price_pos,
    signal_pos,
    qty,
    balance,
    tax_exchange,
    check_args=True,
):
    # check args
    if check_args:
        pass
    
    # main
    qty_new_order = (
        abs(
            pnl_unrealized_AS_qty 
            * (avg_multiplier_no_my_side if pnl_unrealized_AS_qty < 0 and signal != signal_pos else avg_multiplier_my_side)
        )
        * (-1 if signal != signal_pos else 1)

    )
    qty_new = qty_new_order + qty

    if qty_new_order >= 0:
        price_pos =  (
            price_pos * qty
            + price_last * qty_new_order
        ) / qty_new

    if qty_new <= 0:
        return (
            False,
            0,
            0,
            0,
            balance - qty - qty * tax_exchange,
        )

    return (
        in_position,
        price_pos,
        signal_pos,
        qty_new,
        balance - (qty_new_order * pnl_unrealized_AS_prcnt if qty_new_order < 0 else 0)\
        + (qty_new_order * tax_exchange if qty_new_order < 0 else 0),
    )
   