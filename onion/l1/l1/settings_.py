import json

from project_exctentions.g_utils import g_not_iter_from_iter

with open("settings__.json", "r", encoding="utf-8") as f:
    settings = json.load(f)
    settings_default = {
        "settings_target": "user",
        "BACKTEST": {
            "USE_dynamic_exits": False,
            "klines_test":  500,
            "sl": -1,
            "tp": 0.02,
            "leverage": 10,
            "avg_module": {
                "avg_power": 1
            },
            "balance_usd": 300,
            "balance_used_min_percent": 0.01,
            "balance_used_max_percent": 0.2
        },
        "SIGNAL_GENERATION": {
            "ML": {
                "neighbors_count": 8,
                "klines_train": 2000,
                "klines_train_held": 4,
                "features_used": {
                    "RSI_1": {"window": 14},
                    "RSI_2": {"window": 9},
                    "ADX_1": {"window": 20},
                    "CCI_1": {"window": 20},
                    "WT_1": {"window": 15}
                }
            },
            "filters_used": {
                "signals_held": {
                    "held_threshold": 4,
                    "zeros_skip": True,
                    "zeros_skip_held_threshold": 1
                },
                "volatility": {
                    "min_len": 1,
                    "max_len": 10
                },
                "regime": {"threshold": 1},
                "ADX": {"threshold": 25},
                "EMA": {"window": 200},
                "SMA": {"window": 200}
            }
        }   
    }
    if settings["settings_target"].lower().strip() != "user":
        settings = settings_default
    
    settings["SIGNAL_GENERATION"]["max_window_features"] = max(g_not_iter_from_iter(settings["SIGNAL_GENERATION"]["ML"]["features_used"]))
    settings["SIGNAL_GENERATION"]["max_window_filters"] = max(g_not_iter_from_iter(settings["SIGNAL_GENERATION"]["filters_used"]))
    settings["SIGNAL_GENERATION"]["features_used_count"] = len(settings["SIGNAL_GENERATION"]["ML"]["features_used"])
    settings["SIGNAL_GENERATION"]["filters_used_count"] = len(settings["SIGNAL_GENERATION"]["filters_used"])