from json import load

with open("settings_bt.json", "r", encoding="utf-8") as f:
    settings_ = load(f)
    settings_ml = settings_["ml_sttngs"]
    settings_bt = settings_["bt_sttngs"]

if __name__ == "__main__":
    from pprint import pprint

    pprint(settings_bt)
