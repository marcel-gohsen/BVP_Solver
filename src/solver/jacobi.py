import numpy as np

from solver.solver import Solver


class Jacobi(Solver):
    def __init__(self, norm_ord=np.inf, epsilon=10**-3):
        self.norm_ord = norm_ord
        self.epsilon = epsilon

    def solve(self, A, b):
        A = A.astype(float)
        b = b.astype(float)

        rows = A.shape[0]
        columns = A.shape[1]

        B = np.zeros(A.shape)
        c = np.zeros(b.shape)

        for i in range(rows):
            coeff_ij = 0.0

            for j in range(columns):
                if i != j:
                    B[i][j] = - A[i][j]
                else:
                    coeff_ij = A[i][j]

            for j in range(columns):
                B[i][j] /= coeff_ij

            c[i] = b[i] / coeff_ij

        if np.linalg.norm(B, self.norm_ord) >= 1:
            return "Convergence condition is not satisfied!"

        solution = np.random.rand(rows)
        error = np.inf

        while error > self.epsilon:
            next_solution = B.dot(solution) + c
            error = np.linalg.norm(next_solution - solution, self.norm_ord)

            solution = next_solution

        return solution
