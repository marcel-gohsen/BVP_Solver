import numpy as np

from solver.solver import Solver


class SOR(Solver):
    def __init__(self, epsilon=10 ** -3, w=1.11):
        self.epsilon = epsilon
        self.w = w

    def solve(self, A, b):
        A = A.astype(float)
        b = b.astype(float)

        rows = A.shape[0]
        columns = A.shape[1]
        solution = np.random.rand(rows)
        last_error = np.linalg.norm(np.matmul(A, solution) - b)

        while True:
            for i in range(rows):
                sigma = 0.0

                for j in range(columns):
                    if i != j:
                        sigma += A[i][j] * solution[j]

                solution[i] = ((1 - self.w) * solution[i]) + ((self.w / A[i][i]) * (b[i] - sigma))

            error = np.linalg.norm(np.matmul(A, solution) - b)

            if error <= self.epsilon or last_error < error:
                break

            last_error = error

        return solution
