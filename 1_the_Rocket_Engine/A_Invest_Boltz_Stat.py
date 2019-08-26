import numpy as np
import matplotlib.pyplot as plt

mass = 3.347E-27
T = 3000
k = 1.38E-23

def f(mu, sig, x):
	exp = -0.5 * (x - mu)**2 * sig**(-2)
	return (np.sqrt(2 * np.pi) * sig)**(-1) * np.exp(exp)

def P(a, b, mu, sig):
	dx = 0.001
	n = int((b - a) / dx)
	x = np.linspace(a, b, n) + mu
	y = f(mu, sig, x)
	return x, y, np.trapz(y, dx = dx)

def MaxBoz(v, m = 2.0159 * 1.66e-27, T = 3000):
	kB = 1.380649e-23
	e = - 0.5 * m * v ** 2 * (kB * T) ** (-1)
	p1 = m ** (3/2) * (2 * np.pi * kB * T) ** (-3/2)
	p3 = 4 * np.pi * v ** 2
	return p1 * np.exp(e) * p3

def PMaxBoz(a, b):
	dx = 0.001
	n = int((b - a) / dx)
	x = np.linspace(a, b, n)
	y = MaxBoz(x)
	return x, y

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

width = 2.5E4
x, y, A = P(-width,width,0,np.sqrt(k*T/mass))
#print(A)
#a, s, d = FWHM(x, y, 1)
#print(s,d)
plt.plot(x,y)
plt.grid()
plt.show()

x, y = PMaxBoz(-10,10)
plt.plot(x,y)
plt.grid()
plt.show()
