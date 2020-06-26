import numpy as np

from solver.classic_gaussian_elimination import ClassicGaussianElimination
from solver.jacobi import Jacobi


def main():
    A = np.array([[6.25, -1, 0.5],
                  [-1, 5, 2.12],
                  [0.5, 2.12, 3.6]])

    b = np.array([7.5, -8.68, -0.24])

    x = ClassicGaussianElimination.solve(A, b)

    print("x = " + str(x) + "\n")

    A = np.array([[6.25, -1, 0.5],
                  [-1, 5, 2.12],
                  [0.5, 2.12, 3.6]])

    b = np.array([7.5, -8.68, -0.24])

    x = Jacobi.solve(A, b, epsilon=10 ** -8)

    print("x = " + str(x) + "\n")


if __name__ == '__main__':
    main()
