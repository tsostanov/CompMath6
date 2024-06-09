import numpy as np
from prettytable import PrettyTable


def runge_kutta_method(func, y0, t0, t1, h, e):
    t = t0
    y = np.array(y0)
    results = [(t, y[0])]

    table = PrettyTable()
    table.title = "Решение методом Рунге-Кутты"
    table.field_names = ["t", "y", "R"]
    table.float_format = ".5"

    while t < t1:
        k1 = h * func(t, y)[0]
        k2 = h * func(t + 0.5 * h, y + 0.5 * k1)[0]
        k3 = h * func(t + 0.5 * h, y + 0.5 * k2)[0]
        k4 = h * func(t + h, y + k3)[0]
        y_h = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        h /= 2
        k1 = h * func(t, y)[0]
        k2 = h * func(t + 0.5 * h, y + 0.5 * k1)[0]
        k3 = h * func(t + 0.5 * h, y + 0.5 * k2)[0]
        k4 = h * func(t + h, y + k3)[0]
        y_half_h = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        h *= 2

        R = np.max(np.abs(y_half_h - y_h)) / (2 ** 4 - 1)

        table.add_row([f"{t:.5f}", f"{y_h[0]:.5f}", f"{R:.5f}"])

        if R <= e:
            y = y_h
            t = t + h
            results.append((t, y[0]))
        else:
            h /= 2
            print("Шаг был уменьшен до", h)

    print(table)
    return results
