import numpy as np

from solver.solver import Solver


class GaussianEliminationPivotSelection(Solver):
    def solve(self, A, b):
        A = A.astype(float)
        b = b.astype(float)
        dim_A = A.shape

        rows = dim_A[0]
        columns = dim_A[1]

        # Elimination
        for i in range(rows - 1):
            i_max, j_max = self._max_pivot(A, start_row=i)

            # swap rows with max pivots
            A[[i, i_max]] = A[[i_max, i]]
            b[[i, i_max]] = b[[i_max, i]]

            for k in range(i + 1, rows):
                mu = A[k][j_max] / A[i][j_max]

                for j in range(columns):
                    A[k][j] = A[k][j] - (mu * A[i][j])

                b[k] = b[k] - (mu * b[i])

        solution = np.array([None] * rows, dtype=np.float)

        for i in reversed(range(rows)):
            res = b[i]
            div = None

            param_ind = None
            for j in range(columns):
                if not np.isclose(A[i][j], 0, rtol=10**-10):
                    if np.isnan(solution[j]):
                        div = A[i][j]
                        param_ind = j
                    else:
                        res -= solution[j] * A[i][j]

            solution[param_ind] = res / div

        return solution

    @staticmethod
    def _max_pivot(A, start_row=0):
        dim_A = A.shape

        rows = dim_A[0]
        columns = dim_A[1]

        max = 0
        ind = None

        for i in range(start_row, rows):
            for j in range(columns):
                if abs(A[i][j]) > max:
                    max = abs(A[i][j])

                    ind = (i, j)

        return ind
