"""
Program med ikke-lineær regresjonsklasser

All kode er egenskrevet
"""

import numpy as np


class least_squares:
    def __init__(self, data, times, N):
        self.data = data
        self.t = times
        self.N = N  # number of points to divvy interval into

    def f(self, x, v, P, t0):
        # analytic expression, (8) in part2 article
        return v * np.cos(2 * np.pi * (x - t0) / P)

    def residual(self, m, Bs, sigma=1):

        v = Bs[:, :, :, 0]
        P = Bs[:, :, :, 1]
        t0 = Bs[:, :, :, 2]
        return (self.data[m] - self.f(self.t[m], v, P, t0)) ** 2 / sigma ** 2

    def find_best(self, v_d, v_u, P_d, P_u, t_d, t_u):
        v = np.linspace(v_d, v_u, self.N)
        P = np.linspace(P_d, P_u, self.N)
        t = np.linspace(t_d, t_u, self.N)

        permutations = np.transpose(np.asarray(np.meshgrid(v, P, t)), (1, 2, 3, 0))
        # created 3d-meshgrid and inverts it such that shape is (n,n,n,3),
        # meaing elements of cube, (n,n,n), are all possible variations of (v,P,t)

        R = np.zeros(([self.N] * 3))
        # 3d residual array

        for m in range(len(self.data)):
            R += self.residual(m, permutations)
            # adds residual of one datapoint through all permutations to array,
            # would probably be faster to do all datapoints through one perumation at a time

        idx = np.unravel_index(np.argmin(R, axis=None), R.shape)
        # finds index of minimum permuation, best combination in permutations
        return permutations[idx]


class non_linear_reg:
    # Gauss-Newton algorithm
    def __init__(self, data, time):
        self.data = data
        self.t = time

    def f(self, x, v, P, t0):
        # analytic expression, (8)
        return v * np.cos(2 * np.pi * (x - t0) / P)

    def residual(self, v, P, t0):
        res = self.data - self.f(self.t, v, P, t0)
        return res

    def Jacobi(self, x, v, P, t0):
        # Jacobian of f, equation (10) in part2
        dr_dv = -np.cos(2 * np.pi * (x - t0) / P)
        dr_dP = 2 * np.pi * v * (t0 - x) * np.sin(2 * np.pi * (x - t0) / P) / P ** 2
        dr_dt0 = -2 * np.pi * v * np.sin(2 * np.pi * (x - t0) / P) / P

        return np.transpose([dr_dv, dr_dP, dr_dt0])
        # Jacobi-matrix, 3 columns, len(x) rows

    def solve(self, v0, P0, t00, N=100):

        Bs = np.array([v0, P0, t00])
        for i in range(N):
            Ji = self.Jacobi(self.t, *Bs)
            ri = self.residual(*Bs)
            # finds Jacobi and resudial matrix to find next
            worst = ri.max()

            if abs(ri.max() - worst) < 1e-15:
                print(f"{i} iterations used to reach desired tolerance level")
                break

            try:
                Bs = Bs - np.matmul(
                    np.linalg.inv(np.matmul(np.transpose(Ji), Ji)),
                    np.matmul(np.transpose(Ji), ri),
                )
            except:  # if initial guesses are too bad, could get singular matrix
                print("Bad values found")
                break

        return Bs
