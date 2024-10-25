from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

def g_y_train(
    data,
    feauture_main={"name": "RSI", "sell": 70, "buy": 30},
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
):
    # feauture_main["name"] = "INDCS/ " + feauture_main["name"]
    # for key in frozenset(features_add.keys()):
    #     features_add["INDCS/ " + key] = features_add.pop(key)
    
    invert_func = lambda v, comparison_invert: np.invert(v) if comparison_invert else v
    main_sell = np.all((
        data[feauture_main["name"]] > feauture_main["sell"],
        np.all([
            invert_func(
                data[feature] > features_add[feature]["sell"], 
                comparison_invert=features_add[feature]["comparison_invert"]
            ) 
            for feature in features_add
        ], axis=0)
    ), axis=0)
    main_buy = np.all((
        data[feauture_main["name"]] < feauture_main["buy"],
        np.all([
            invert_func(
                data[feature] < features_add[feature]["buy"], 
                comparison_invert=features_add[feature]["comparison_invert"]
            ) 
            for feature in features_add
        ], axis=0)
    ), axis=0)

    # def g_validate_add_features(features_list):
    #     predict_arr = np.full(len(data), np.nan)
        
        
        # invert_func = lambda v, bool_: np.invert(v) if bool_ else v
        # return [
        #     np.logical_and(side, np.all(cond, axis=0))
        #     for side, cond in zip(
        #         (main_sell, main_buy),
        #         zip(*[[*cond] for cond in [
        #             invert_func(
        #                 (
        #                     (data[feature] > thresholds[0]),
        #                     ((data[feature] < thresholds[1]) if thresholds[1] != None else (data[feature] > thresholds[0]))
        #                 ),
        #                 thresholds[2]
        #             )
        #             for feature, thresholds in features_add.items()
        #         ]])
        #     )
        # ]

    # if features_add:
    #     main_sell, main_buy = g_validate_add_features()
    return np.where(main_sell, -1, np.where(main_buy, 1, 0))

def g_clean_x(x):
    return pd.DataFrame(
        SimpleImputer(strategy="mean")\
            .fit_transform(x.replace({-np.inf: np.nan, np.inf: np.nan})),
        columns=x.columns
    )

def g_knn_predict(
    x_train,
    x_test,
    y_train,
    n_neighbors=3
):
    return KNeighborsClassifier(n_neighbors=n_neighbors)\
        .fit(x_train, y_train)\
        .predict(x_test)
