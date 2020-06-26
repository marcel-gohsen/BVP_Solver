import numpy as np


class Jacobi:
    @staticmethod
    def solve(A, b, norm_ord=np.inf, epsilon=10 ** -3):
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

        if np.linalg.norm(B, norm_ord) >= 1:
            print("Convergence condition is not satisfied!")
            return None

        solution = np.random.rand(rows)
        error = np.inf

        while error > epsilon:
            next_solution = B.dot(solution) + c
            error = np.linalg.norm(next_solution - solution, norm_ord)

            solution = next_solution

        return solution
