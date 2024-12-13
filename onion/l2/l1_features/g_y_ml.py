from onion.l1.l1.g_settings_ import *

def g_y_train_signal_A_comparison_A_klines_train_held(
    last_price,
    last_price_MI_klines_train_held,
    tp_train=settings["SIGNAL_GENERATION"]["ML"]["tp_train"],
    check_params=True,
):
    """
    по дефолту эта функция сравнивает цены так:
    price[klines_train_held back] < price = -1 (short)
    price[klines_train_held back] > price = 1 (long)
    price[klines_train_held back] == price = 0 (non)

    """
    # check params
    if check_params:
        if not isinstance(last_price, float):
            raise ValueError("Last price is not float")
        if not isinstance(last_price_MI_klines_train_held, float):
            raise ValueError("Last price MI klines train held is not float")
    
    # init
    last_price_MI_klines_train_held_MU_tp_train = last_price_MI_klines_train_held * tp_train
    
    # main
    if last_price_MI_klines_train_held + last_price_MI_klines_train_held_MU_tp_train < last_price:
        signal = -1
    elif last_price_MI_klines_train_held - last_price_MI_klines_train_held_MU_tp_train > last_price:
        signal = 1
    else:
        signal = 0
    
    return signal