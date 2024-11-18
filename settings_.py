import json

with open("settings.json", "r", encoding="utf-8") as f:
    settings = json.load(f)
    settings_filter = settings.get("filter".upper())
    settings_ml = settings.get("ml".upper())