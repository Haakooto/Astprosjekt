import numpy as np
import matplotlib.pyplot as plt

def f(mu, sig, x):
	exp = -0.5 * (x - mu)**2 * sig**(-2)
	return (np.sqrt(2 * np.pi) * sig)**(-1) * np.exp(exp)
			
def P(a, b, mu, sig):
	dx = 0.001
	n = int((b - a) / dx)
	x = np.linspace(a, b, n) + mu
	y = f(mu, sig, x)
	return x, y, np.trapz(y, dx = dx)

def FWHM(X, Y, sig):
	if X.shape != Y.shape:
		return False

	half_max = max(Y) / 2
	a = np.argwhere(abs(Y - half_max) < 1e-4)

	try:
		d = X[a[-1]] - X[a[0]]
	except:
		print(a)

	s = 2*np.sqrt(2*np.log(2))*sig

	return a,s,d


x, y, A = P(-3,3,0,1)
print(A)
a, s, d = FWHM(x, y, 1)
print(s,d)
# plt.plot(x,y)
# plt.grid()
# plt.show()
