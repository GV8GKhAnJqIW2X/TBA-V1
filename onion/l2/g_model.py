from onion.l1.s_logging import logger
from onion.l1.g_settings import settings_ml

from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# @logger.catch
async def g_y_train(
    data,
    feature_main=settings_ml["test_sett"]["feature_main"],
    feature_add=settings_ml["test_sett"]["feature_add"]
):
    invert_func = lambda v, comparison_invert: np.invert(v) if comparison_invert else v
    main_sell = np.all((
        data[feature_main["name"]] > feature_main["sell"],
        np.all([
            invert_func(
                data[feature] > feature_add[feature]["sell"], 
                comparison_invert=feature_add[feature]["comparison_invert"]
            ) 
            for feature in feature_add
        ], axis=0)
    ), axis=0)
    main_buy = np.all((
        data[feature_main["name"]] < feature_main["buy"],
        np.all([
            invert_func(
                data[feature] < feature_add[feature]["buy"], 
                comparison_invert=feature_add[feature]["comparison_invert"]
            ) 
            for feature in feature_add
        ], axis=0)
    ), axis=0)
    return np.where(main_sell, -1, np.where(main_buy, 1, 0))

def g_knn_predict(
    x_train,
    x_test,
    y_train,
    n_neighbors=settings_ml["n_neighbors"]
):
    return KNeighborsClassifier(n_neighbors=n_neighbors)\
        .fit(x_train, y_train)\
        .predict(x_test)