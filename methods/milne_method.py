import numpy as np
from prettytable import PrettyTable
from .runge_kutta_method import runge_kutta_method


def milne_method(func, y0, t0, t1, h, e, exact_solution, max_step_reductions=1000):
    results = runge_kutta_method(func, y0, t0, t1, h, e)

    if len(results) < 4:
        print("Ошибка: Не удалось получить первые три точки методом Рунге-Кутты.")
        return []

    t = t0
    x_needed = [t0, t0 + h, t0 + 2 * h, t0 + 3 * h]
    y_list = []
    counter = 0
    while counter != 4:
        for x, y in results:
            if abs(x - x_needed[counter]) < 0.0000001:
                y_list.append(y)
                counter += 1
                break

    while len(y_list) < ((t1 - t0) / h):
        y_list.append(y_list[0])


    table = PrettyTable()
    table.title = "Метод Милна"
    table.field_names = ["t", "y", "R"]
    table.float_format = ".5"

    i = 3
    step_reductions = 0

    while i < (len(y_list) - 1):
        y_pred = y_list[i - 3] + 4 * h * (
                2 * func(t + (i - 2) * h, y_list[i - 2]) - func(t + (i - 1) * h, y_list[i - 1]) +
                2 * func(t + i * h, y_list[i])) / 3

        y_corr = y_list[i - 1] + h * (
                func(t + (i - 1) * h, y_list[i - 1]) + 4 * func(t + i * h, y_list[i]) +
                func(t + (i + 1) * h, y_pred)) / 3
        while np.abs(y_corr - y_pred) > e:
            y_pred = y_corr
            y_corr = y_list[i - 1] + h * (
                    func(t + (i - 1) * h, y_list[i - 1]) + 4 * func(t + i * h, y_list[i]) +
                    func(t + (i + 1) * h, y_pred)) / 3

        y_list[i + 1] = y_corr

        R = np.abs(exact_solution(t + (i + 1) * h) - y_corr)
        if R > e:
            h /= 2
            step_reductions += 1
            print("Шаг был уменьшен до", h)
            if step_reductions > max_step_reductions:
                print("Достигнуто максимальное количество уменьшений шага.")
                break
        else:
            table.add_row([f"{t + (i + 1) * h:.5f}", f"{y_corr:.5f}", f"{R:.5f}"])
            i += 1


    print(table)

    x = np.arange(t0, t1 + h, h)
    results = ([(xi, yi) for xi, yi in zip(x, y_list)])
    return results
