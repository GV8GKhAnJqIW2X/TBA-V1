from onion.l1.s_logging import logger

import numpy as np

# @logger.catch
async def g_data_trace(
    y, 
    name_trace="8===D", 
    color_rgb=(200, 200, 200),
):
    y = y.dropna()
    return {
        "x": np.arange(len(y)),
        "y": y,
        "name": name_trace,
        "line": {"color": f"rgb{color_rgb}"}
    }

# @logger.catch
async def g_data_mark(
    y,
    labels,
    classes,
):
    return [
        (lambda v=np.where(labels == class_): {
            "x": np.arange(len(labels))[v],
            "y": y.loc[v],
            "name": classes[class_]["name"],
            "line": {"color": f"rgb{classes[class_]["color_rgb"]}"},
        })()
        for class_ in classes
    ]