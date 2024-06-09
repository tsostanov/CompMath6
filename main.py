import numpy as np
import matplotlib.pyplot as plt
from equations import equations, exact_solutions
from methods.euler_method import euler_method
from methods.runge_kutta_method import runge_kutta_method
from methods.milne_method import milne_method
from prettytable import PrettyTable


def main():
    print("Выберите систему уравнений:")
    for i, eq in enumerate(equations, 1):
        print(f"{i}. {eq['name']}")

    equation_choice = int(input("Введите номер системы: ")) - 1
    selected_system = equations[equation_choice]['func']
    exact_solution = exact_solutions[equation_choice]

    print("\nВыберите метод решения:")
    print("1. Метод Эйлера")
    print("2. Метод Рунге-Кутты 4-го порядка")
    print("3. Метод Милна")
    method_choice = int(input("Введите номер метода: "))

    print("\nВведите начальное значение x, начальные условия y0, шаг h, количество шагов n и точность e")
    x0 = float(input("Начальное значение x: "))
    y0 = list(map(float, input("Начальные условия y0 (через пробел): ").split()))
    h = float(input("Шаг h: "))
    n = int(input("Количество шагов n: "))
    e = float(input("Введите требуемую точность e: "))

    t0 = x0
    t1 = t0 + n * h

    if method_choice == 1:
        results = euler_method(selected_system, y0, t0, t1, h, e)
    elif method_choice == 2:
        results = runge_kutta_method(selected_system, y0, t0, t1, h, e)
    elif method_choice == 3:
        results = milne_method(selected_system, y0, t0, t1, h, e, exact_solution)
    else:
        print("Неверный выбор метода!")
        return

    table = PrettyTable()
    table.title = "Решение"
    table.field_names = ["i", "xi", "yi", "f(xi, yi)", "Точное решение"]
    table.float_format = ".5"

    for i, (t, y) in enumerate(results):
        exact_y = exact_solutions[equation_choice](t) if equation_choice < len(exact_solutions) else "N/A"
        f_value = selected_system(t, y)
        table.add_row([i, t, y.flatten()[0], f_value.flatten()[0], exact_y])

    print(table)

    ts, ys = zip(*results)
    ys = np.array([y.flatten() for y in ys])

    for i in range(ys.shape[1]):
        plt.plot(ts, ys[:, i], label=f'Численное решение y{i + 1}')

    if equation_choice < len(exact_solutions):
        exact_ys = [exact_solutions[equation_choice](t) for t in ts]
        plt.plot(ts, exact_ys, label='Точное решение', linestyle='--')

    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
