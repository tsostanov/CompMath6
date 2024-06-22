import numpy as np
import sys
import sympy as sp


def first_order_equation(x, y):
    dydx = y + (1 + x) * y ** 2
    return dydx


def second_order_equation(x, y):
    dydx = np.sin(x) - y
    return dydx


def third_order_equation(x, y):
    dydx = -y + x ** 2
    return dydx


def find_initial_C(x0, y0, equation_choice):
    x, C = sp.symbols('x C')
    equation = exact_solutions[equation_choice]
    solution_eq = equation(x0, C) - y0[0]
    solved_C = sp.solve(solution_eq, C)
    if not solved_C:
        print("В начальной точке решение неопределенно")
        sys.exit()
    print(solved_C)
    return lambda x: equation(x, solved_C[0])


equations = [
    {'name': "y' = y + (1 + x) * y^2", 'func': first_order_equation},
    {'name': "y' = sin(x) - y", 'func': second_order_equation},
    {'name': "y' = -y + x^2", 'func': third_order_equation}
]

exact_solutions = [
    lambda x, C: -sp.exp(x) / (C + x * sp.exp(x)),
    lambda x, C: C * sp.exp(-x) + sp.sin(x) / 2 - sp.cos(x) / 2,
    lambda x, C: C * sp.exp(-x) + x ** 2 - 2 * x + 2
]

