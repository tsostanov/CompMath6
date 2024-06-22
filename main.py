import numpy as np
import matplotlib.pyplot as plt
from equations import equations, exact_solutions, find_initial_C
from methods.euler_method import euler_method
from methods.runge_kutta_method import runge_kutta_method
from methods.milne_method import milne_method
from prettytable import PrettyTable


def get_valid_input(prompt, type_func, condition_func=None, error_message="Неверный ввод! Попробуйте снова."):
    while True:
        try:
            user_input = type_func(input(prompt))
            if condition_func and not condition_func(user_input):
                raise ValueError
            return user_input
        except ValueError:
            print(error_message)


def main():
    print("Выберите систему уравнений:")
    for i, eq in enumerate(equations, 1):
        print(f"{i}. {eq['name']}")

    equation_choice = get_valid_input(
        "Введите номер системы: ",
        int,
        lambda x: 1 <= x <= len(equations),
        "Неверный номер системы! Попробуйте снова."
    ) - 1
    selected_system = equations[equation_choice]['func']

    print("\nВыберите метод решения:")
    print("1. Метод Эйлера")
    print("2. Метод Рунге-Кутты 4-го порядка")
    print("3. Метод Милна")
    method_choice = get_valid_input(
        "Введите номер метода: ",
        int,
        lambda x: 1 <= x <= 3,
        "Неверный номер метода! Попробуйте снова."
    )

    x0 = get_valid_input("Начальное значение x: ", float)
    y0 = get_valid_input(
        "Начальное значение y: ",
        lambda x: list(map(float, x.split())),
        lambda x: len(x) > 0,
        "Неверные начальные условия! Попробуйте снова."
    )
    exact_solution = find_initial_C(x0, y0, equation_choice)

    h = get_valid_input("Шаг h: ", float, lambda x: x > 0, "Шаг должен быть положительным! Попробуйте снова.")
    n = get_valid_input("Количество шагов n: ", int, lambda x: x > 0, "Количество шагов должно быть положительным! Попробуйте снова.")
    e = get_valid_input("Введите требуемую точность e: ", float, lambda x: x > 0, "Точность должна быть положительной! Попробуйте снова.")

    t0 = x0
    t1 = t0 + n * h

    if method_choice == 1:
        results = euler_method(selected_system, y0, t0, t1, h, e)
    elif method_choice == 2:
        results = runge_kutta_method(selected_system, y0, t0, t1, h, e)
    elif method_choice == 3:
        results = milne_method(selected_system, y0, t0, t1, h, e, exact_solution)

    table = PrettyTable()
    table.title = "Решение"
    table.field_names = ["i", "xi", "yi", "f(xi, yi)", "Точное решение"]
    table.float_format = ".5"

    for i, (t, y) in enumerate(results):
        exact_y = exact_solution(t) if equation_choice < len(exact_solutions) else "N/A"
        f_value = selected_system(t, y)
        table.add_row([i, t, y.flatten()[0], f_value.flatten()[0], exact_y])

    print(table)

    ts, ys = zip(*results)
    ys = np.array([y.flatten() for y in ys])

    for i in range(ys.shape[1]):
        plt.plot(ts, ys[:, i], label=f'Численное решение y{i + 1}')

    if equation_choice < len(exact_solutions):
        exact_ys = [exact_solution(t) for t in ts]
        plt.plot(ts, exact_ys, label='Точное решение', linestyle='--')

    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
