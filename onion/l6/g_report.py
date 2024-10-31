import pandas as pd

def g_report_balance(files_list, load_func):
    balance = {}
    for symbol in files_list:
        data = load_func(symbol)
        balance[symbol] = data["BT/ balance"].dropna()
    return pd.DataFrame(balance).iloc[-1].describe()