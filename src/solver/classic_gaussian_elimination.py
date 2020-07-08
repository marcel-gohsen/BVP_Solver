import numpy as np

from solver.solver import Solver


class ClassicGaussianElimination(Solver):
    def __init__(self):
        pass

    def solve(self, A, b):
        """
        Solve linear equation system A * x = b
        :param A: Matrix A as numpy array
        :param b: Vector b as numpy array
        """

        A = A.astype(float)
        b = b.astype(float)
        dim_A = A.shape

        rows = dim_A[0]
        columns = dim_A[1]
        current_row = 0

        # Elimination
        for j in range(columns):
            for i in range(current_row + 1, rows):
                mu = A[i][j] / A[current_row][j]

                for k in range(columns):
                    A[i][k] = A[i][k] - (mu * A[current_row][k])

                b[i] = b[i] - mu * b[current_row]

            current_row += 1

        solution = np.zeros((rows,))
        current_column = columns - 1

        # Solve and re-substitute
        for i in reversed(range(rows)):
            for j in reversed(range(current_column, columns)):
                if j == current_column:
                    solution[i] = (b[i] / A[i][j])
                else:
                    b[i] -= A[i][j] * solution[j]

            current_column -= 1

        return solution
