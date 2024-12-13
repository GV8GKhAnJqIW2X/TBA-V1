import numpy as np

def g_normalize(
    src, 
    min_value=0, 
    max_value=1,
    check_args=True,
):
    """Нормализация значения src в диапазоне [min_value, max_value]."""

    # # check args
    # if check_args:
    #     if not isinstance(src, np.ndarray):
    #         src = np.array(src)

    # Инициализация исторических минимумов и максимумов
    historic_min = float('inf')
    historic_max = float('-inf')

    # Обновление исторического минимума и максимума
    if src is not None:  # Проверка на None
        historic_min = np.min(src)
        historic_max = np.max(src)

    # Нормализация
    range_diff = historic_max - historic_min

    # Избегаем деления на ноль
    if range_diff <= 1e-10:
        range_diff = 1e-10

    normalized_value = min_value + (max_value - min_value) * (src - historic_min) / range_diff

    return normalized_value