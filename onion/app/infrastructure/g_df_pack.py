from onion.domain.indicators import g_indicators_data
from onion.domain.model_ import (
    g_y_train,
    g_knn_predict,
    g_clean_x
)
from onion.app.domain.func_session_global import g_klines
from onion.app.infrastructure.data_processing import g_rolling_apply, g_klines_splitting
    
async def g_df_pack(data, settings_ml):
    if data is not None:
        data = g_indicators_data(
            data=data,
            in_need_l1={
                "RSI": dict(period=14,),
                "ADX": dict(period=14,),
                "CCI": dict(period=21,),
                "WT": dict(period=14,),
                "TSI": dict(period=14,),
            },
            in_need_l2={"LD": dict(bars_back=100,)},
        )
        data["train_label"] = g_y_train(
            data,
            feauture_main={"name": "INDCS/ RSI", "sell": 70, "buy": 30},
            features_add={
                "INDCS/ ADX": dict(
                    sell=20, 
                    buy=40, 
                    comparison_invert=True,
                ),
                "INDCS/ TSI": dict(
                    sell=0.8, 
                    buy=-0.8, 
                    comparison_invert=False,
                )
            }
        )
        indcs_colums = [column for column in data.columns if "INDCS" in column]
        data[indcs_colums] = g_clean_x(data[indcs_colums])
        data["predicted_label"] = g_rolling_apply(
            arr=data[indcs_colums + ["train_label"]],
            window=settings_ml["klines_train_used"],
            func=lambda v: g_knn_predict(
                x_train=v[indcs_colums].iloc[:-1].values,
                x_test=v[indcs_colums].iloc[-1].values.reshape(1, 6),
                y_train=v["train_label"].iloc[:-1].values,
            )
        )
        return data
    return None