from onion.l1.s_logging import logger
from onion.l2.g_indicators import g_data_x
from onion.l2.g_utils import g_rolling_apply
from onion.l2.g_model import g_y_train, g_knn_predict

import pandas as pd
from pprint import pprint

# @logger.catch
async def g_data_test(
    klines,
    ml_sett,
    indcs_list,
):
    data = pd.DataFrame({
        "close": klines[:, 4],
        "high": klines[:, 2],
        "low": klines[:, 3],
    })
    data = await g_data_x(
        data=data, 
        l1_indcs_train_sett=ml_sett["l1_indcs_train_sett"], 
        l2_indcs_train_sett=ml_sett["l2_indcs_train_sett"],
        indcs_list=indcs_list,
    )
    data["train"] = await g_y_train(data=data)
    data["test"] = await g_rolling_apply(
        arr=data[indcs_list + ["train"]], 
        window=ml_sett["klines_train_used_num"], 
        func=lambda v: g_knn_predict(
            x_train=v[indcs_list].iloc[:-1].values,
            x_test=v[indcs_list].iloc[-1].values.reshape(1, 6),
            y_train=v["train"].iloc[:-1].values,
        )
    )
    return data
