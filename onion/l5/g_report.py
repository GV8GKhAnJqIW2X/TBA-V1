import pandas as pd

def g_report(data):
    return pd.Series([el["BT/ balance"].iloc[-1] for el in data]).describe()