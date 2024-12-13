from onion.l1.l1.g_distances import *

def g_lorentzian_prediction(
    y_train_array,
    x_train_arrays, 
    x_features_series, 
    klines_train,
    klines_train_held,
    neighbors_count,
):
    predictions = []
    distances = []
    last_distance = -1.0
    for i in range(klines_train):
        distance = g_lorentzian_distance_A_series_array(i, x_features_series, x_train_arrays,)
        if distance >= last_distance and i % klines_train_held == 0:
            last_distance = distance
            distances.append(distance)
            predictions.append(round(y_train_array[i]))
            if len(predictions) > neighbors_count:
                # получаем значение ближе к концу массива (75%)
                # Использование 75% может быть связано с тем, 
                # что последние значения в массиве расстояний могут лучше 
                # отражать текущую тенденцию или состояние рынка.
                last_distance = distances[round(neighbors_count * 3 / 4)]
                distances.pop(0)
                predictions.pop(0)
    
    return sum(predictions)