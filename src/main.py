import math

import numpy as np
import sympy

from solver.classic_gaussian_elimination import ClassicGaussianElimination
from solver.gaussian_elimination_pivot_selection import GaussianEliminationPivotSelection
from solver.jacobi import Jacobi
from solver.sor import SOR


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
    x, y, y_diff_1, y_diff_2 = sympy.symbols("x y y' y''")
    y_back, y_forward, h = sympy.symbols("y_{i-1} y_{i+1} h")

    orig_diff_eq = sympy.Eq(3 * sympy.sin(x), y_diff_2 + y)

    print("BOUNDARY VALUE PROBLEM SOLVER")
    print("-----------------------------")

    print("Solve: ")
    sympy.pprint(orig_diff_eq)

    boundary_cond_1 = sympy.Eq(y_diff_1 + y, 0)
    boundary_cond_2 = sympy.Eq(y_diff_1 + y, 0)

    print("\nFor x = 0: ", end="")
    sympy.pprint(boundary_cond_1)
    print("For x = pi/2: ", end="")
    sympy.pprint(boundary_cond_2)

    print("-----------------------------")

    n = 100
    boundary_low = 0
    boundary_high = math.pi / 2

    # Define finite difference operators
    second_order_approx = (y_back - 2 * y + y_forward) / (h ** 2)

    forward_first_order_approx = (y_forward - y) / h
    backward_first_order_approx = (y - y_back) / h

    diff_expr = orig_diff_eq.rhs
    diff_expr = diff_expr.subs(y_diff_2, second_order_approx)
    diff_expr = sympy.simplify(diff_expr)
    diff_expr = sympy.collect(diff_expr, y)

    diff_eq = sympy.Eq(orig_diff_eq.lhs, diff_expr)
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

    print("Introduce discretization for n = " + str(n))
    linear_system, points = introduce_discretisation(linear_system, n, (boundary_low, boundary_high))

    print("Construct linear equation system")
    b, A = construct_matrix(linear_system)
    print(A)
    print(b)

    solvers = [ClassicGaussianElimination(), GaussianEliminationPivotSelection(), Jacobi(), SOR()]
    results = []

    function = (3 / 8) * (((math.pi + 2) * sympy.cos(x)) - ((math.pi - 2) * sympy.sin(x))) - (
            (3 / 2) * x * sympy.cos(x))
    exact_solution = [function.subs(x, points[i]) for i in range(len(points))]

    print("\nExact solution")
    print("Solution: (" + ",".join([str(round(x, 4)) for x in exact_solution]) + ")")

    for solver in solvers:
        y_approx = solver.solve(A, b)

        if isinstance(y_approx, str):
            print("\n" + solver.__class__.__name__ + ": " + y_approx)
            continue

        mean_abs_error = 0
        mean_squared_error = 0

        for i in range(len(points)):
            mean_squared_error += (exact_solution[i] - y_approx[i]) ** 2
            mean_abs_error += abs(exact_solution[i] - y_approx[i])

        mean_squared_error /= len(points)
        mean_abs_error /= len(points)

        results.append({"name": solver.__class__.__name__,
                        "solution": y_approx,
                        "mse": mean_squared_error,
                        "mae": mean_abs_error})

    for result in results:
        print("\n" + result["name"] + "\n"
                                      "Solution: (" + ",".join([str(round(x, 4)) for x in result["solution"]]) + ")\n"
                                                                                                                 "Mean abs. error: " + str(
            round(result["mae"], 4)) + "\n"
                                       "Mean square error: " + str(round(result["mse"], 4)))

    plot = sympy.plotting.plot(function,
                               xlim=(0, math.pi / 2),
                               ylim=(-5, 5),
                               label="Exact", legend=True,
                               line_color="k",
                               show=False)

    colors = ["r", "g", "b", "y"]

    for ind, result in enumerate(results):
        data = [(points[i], result["solution"][i]) for i in range(len(points))]

        func = sympy.polys.polyfuncs.interpolate(data, x)
        plot.append(sympy.plotting.plot(func, label=result["name"],
                                        line_color=colors[ind], show=False)[0])
    plot.show()


if __name__ == '__main__':
    main()
