import math

import numpy as np
import sympy

from solver.classic_gaussian_elimination import ClassicGaussianElimination


def introduce_discretisation(linear_system, N, boundaries):
    x, h = sympy.symbols("x, h")
    step_size = (boundaries[1] - boundaries[0]) / (N - 1)

    discretizised_system = []
    points = []

    for i, equation in enumerate(linear_system):
        point = step_size * i
        points.append(point)

        equation = equation.subs([(h, step_size), (x, point)])
        discretizised_system.append(equation)

    return discretizised_system, points


def construct_matrix(linear_system):
    matrix = []
    result_vector = []
    for equation in linear_system:
        row = []

        for i in range(1, len(linear_system) + 1):
            row.append(equation.rhs.coeff(sympy.Symbol("y_{" + str(i) + "}")))

        result_vector.append(equation.lhs)
        matrix.append(row)

    A = np.array(matrix)
    b = np.array(result_vector)

    return b, A


def main():
    n = 10
    boundary_low = 0
    boundary_high = math.pi / 2

    x, y, y_diff_2 = sympy.symbols("x y_{i} y''")

    # Define finite difference operators
    y_back, y_forward, h = sympy.symbols("y_{i-1} y_{i+1} h")
    second_order_approx = (y_back - 2 * y + y_forward) / (h ** 2)

    forward_first_order_approx = (y_forward - y) / h
    backward_first_order_approx = (y - y_back) / h

    diff_expr = y_diff_2 + y
    diff_expr = diff_expr.subs(y_diff_2, second_order_approx)
    diff_expr = sympy.simplify(diff_expr)
    diff_expr = sympy.collect(diff_expr, y)

    diff_eq = sympy.Eq(3 * sympy.sin(x), diff_expr)
    diff_eq = sympy.Eq(diff_eq.lhs * h ** 2, diff_eq.rhs * h ** 2)

    linear_system = []

    for i in range(2, n):
        y_back_indexed, y_forward_indexed, y_indexed = sympy.symbols(
            "y_{" + str(i - 1) + "} y_{" + str(i + 1) + "} y_{" + str(i) + "}")

        linear_system.append(diff_eq.subs([(y_back, y_back_indexed), (y_forward, y_forward_indexed), (y, y_indexed)]))

    y_diff_1 = sympy.symbols("y'_{i}")
    boundary_expr = y + y_diff_1
    boundary_expr_low = boundary_expr.subs(y_diff_1, forward_first_order_approx)
    boundary_expr_low = sympy.simplify(boundary_expr_low)
    boundary_expr_low = sympy.collect(boundary_expr_low, y)

    boundary_eq_low = sympy.Eq(0 * h, (boundary_expr_low * h).subs(
        [(y, sympy.Symbol("y_{1}")), (y_forward, sympy.Symbol("y_{2}"))]))
    linear_system.insert(0, boundary_eq_low)

    boundary_expr_high = boundary_expr.subs(y_diff_1, backward_first_order_approx)
    boundary_expr_high = sympy.simplify(boundary_expr_high)
    boundary_expr_high = sympy.collect(boundary_expr_high, y)

    boundary_eq_high = sympy.Eq(0 * h, (boundary_expr_high * h).subs(
        [(y, sympy.Symbol("y_{" + str(n) + "}")), (y_back, sympy.Symbol("y_{" + str(n - 1) + "}"))]))
    linear_system.append(boundary_eq_high)

    linear_system, points = introduce_discretisation(linear_system, n, (boundary_low, boundary_high))
    b, A = construct_matrix(linear_system)

    y_gauss = ClassicGaussianElimination.solve(A, b)

    x = sympy.Symbol("x")
    actual_function = (3 / 8) * (((math.pi + 2) * sympy.cos(x)) - ((math.pi - 2) * sympy.sin(x))) - (
                (3 / 2) * x * sympy.cos(x))

    mean_squared_error = 0

    for i in range(len(points)):
        mean_squared_error += (actual_function.subs(x, points[i]) - y_gauss[i]) ** 2

    mean_squared_error /= len(points)

    print(mean_squared_error)

    # A = np.array([[6.25, -1, 0.5],
    #               [-1, 5, 2.12],
    #               [0.5, 2.12, 3.6]])
    #
    # b = np.array([7.5, -8.68, -0.24])
    #
    # x = ClassicGaussianElimination.solve(A, b)
    #
    # print("GAUSSIAN ELIMINATION: ")
    # print("x = " + str(x) + "\n")
    #
    # x = Jacobi.solve(A, b, epsilon=10 ** -3)
    #
    # print("JACOBI METHOD: ")
    # print("x = " + str(x) + "\n")


if __name__ == '__main__':
    main()
