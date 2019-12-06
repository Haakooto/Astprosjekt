import numpy as np

class least_squares:
	def __init__(self, data, times, N):
		# self.data = np.transpose(np.array([[[data] * N] * N] * N), (3, 0, 1, 2))
		# self.t = np.transpose(np.array([[[time] * N] * N] * N), (3, 0, 1, 2))
		self.data = data
		self.t = times
		self.N = N

	def f(self, x, v, P, t0):
		return v * np.cos(2 * np.pi * (x - t0) / P)

	def residual(self, m, Bs, sigma = 1):

		v = Bs[:, :, :, 0]
		P = Bs[:, :, :, 1]
		t0 = Bs[:, :, :, 2]
		return (self.data[m] - self.f(self.t[m], v, P, t0)) ** 2 / sigma ** 2

	def find_best(self, v_d, v_u, P_d, P_u, t_d, t_u):
		v = np.linspace(v_d, v_u, self.N)
		P = np.linspace(P_d, P_u, self.N)
		t = np.linspace(t_d, t_u, self.N)


		permutations = np.transpose(np.asarray(np.meshgrid(v, P, t)), (1, 2, 3, 0))

		R = np.zeros(([self.N] * 3))

		for m in range(len(self.data)):
			R += self.residual(m, permutations)
		# print(permutations.shape)
		# residuals = self.residual(permutations)
		# best = np.argmin(residuals)
		# print(np.argmin(R, axis=None))
		idx = np.unravel_index(np.argmin(R, axis = None), R.shape)
		# print(idx)
		return permutations[idx]

		# return permutations[best]

class non_linear_reg:
	def __init__(self, data, time):
		self.data = data
		self.t = time

	def f(self, x, v, P, t0):
		return v * np.cos(2 * np.pi * (x - t0) / P)

	def residual(self, x, y, v, P, t0):
		res = y - self.f(x, v, P, t0)
		return res

	def Jacobi(self, x, v, P, t0):
		dr_dv = -np.cos(2 * np.pi * (x - t0) / P)
		dr_dP = 2 * np.pi * v * (t0 - x) * np.sin(2 * np.pi * (x - t0) / P) / P ** 2
		dr_dt0 = -2 * np.pi * v * np.sin(2 * np.pi * (x - t0) / P) / P

		return np.transpose([dr_dv, dr_dP, dr_dt0])

	def solve(self, v0, P0, t00, N=100):

		Bs = np.array([v0, P0, t00])
		worst = 100
		for i in range(N):
			Ji = self.Jacobi(self.t, *Bs)
			ri = self.residual(self.t, self.data, *Bs)

			if abs(ri.max() - worst) < 1e-15:
				print(f"{i} iterations used to read desired tolerance level")
				break

			try:
				Bs = Bs - np.matmul(
					np.linalg.inv(np.matmul(np.transpose(Ji), Ji)),
					np.matmul(np.transpose(Ji), ri),
				)
			except:
				break

			worst = ri.max()

		return Bs
