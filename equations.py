import numpy as np


def first_order_equation(x, y):
    dydx = y + (1 + x) * y ** 2
    return dydx


def second_order_equation(x, y):
    dydx = np.sin(x) - y
    return dydx


def third_order_equation(x, y):
    dydx = -y + x ** 2
    return dydx


equations = [
    {'name': "y' = y + (1 + x) * y^2", 'func': first_order_equation},
    {'name': "y' = sin(x) - y", 'func': second_order_equation},
    {'name': "y' = -y + x^2", 'func': third_order_equation}
]


def exact_solution_1(x):
    return -np.exp(x) / (x * np.exp(x))


def exact_solution_2(x):
    return np.sin(x) / 2 - np.cos(x) / 2


def exact_solution_3(x):
    return x ** 2 - 2 * x + 2


exact_solutions = [exact_solution_1, exact_solution_2, exact_solution_3]
