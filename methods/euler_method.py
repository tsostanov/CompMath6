import numpy as np
from prettytable import PrettyTable


def euler_method(func, y0, t0, t1, h, e):
    t = t0
    y = np.array(y0)
    results = [(t, y)]

    table = PrettyTable()
    table.title = "Проверка по правилу Рунге"
    table.field_names = ["t", "y", "R (Runge)"]
    table.float_format = ".5"

    while t < t1:
        y_h = y + h * func(t, y)[0]
        y_half_h = y + (h / 2) * func(t, y)[0]
        print(t, 'при шаге h:', h ,y_h,'при шаге h/2:', h / 2, y_half_h)

        R = np.max(np.abs(y_half_h - y_h)) / (2 ** 1 - 1)

        table.add_row([f"{t:.5f}", f"{y_h[0]:.5f}", f"{R:.5f}"])

        if R <= e:
            y = y_h
            t = t + h
            results.append((t, y))
        else:
            h /= 2
            print("Шаг был уменьшен до", h)

    print(table)
    return results
