import numpy as np

def g_lorentzian_distance(
    i, 
    feature_series, 
    feature_arrs
) -> float:
    """
    PARAMS:
    i - int The index of the current bar.
    feature_series - NDArray<float> The feature series.
    feature_arrs - dict[str, NDArray] The feature series to calculate distances for.

    RETURNS:
    float The Lorentzian distance for the current bar.
    
    """
    return np.sum([
        np.log(1 + abs(feature_series[key] - array[i]))
        for key, array in feature_arrs.items()
    ])