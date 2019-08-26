import numpy as np
import matplotlib.pyplot as plt

N = 1E5
m = 3.347E-27
T = 3000
k = 1.38E-23
sig_b = np.sqrt(k*T/m)


def f1(mu, sig, x):
	exp = -0.5 * (x - mu)**2 * sig**(-2)
	return (np.sqrt(2 * np.pi) * sig)**(-1) * np.exp(exp)

def f2(m,T,x):
	return (m/(2*np.pi*k*T))**(3/2)*np.exp(-m*x**2/(2*k*T))*4*np.pi*x**2

def P(a, b, mu, sig,f):
	n = 1E4+1
	dx = (b-a)/n
	x = np.linspace(a, b, n) + mu
	y = f(mu, sig, x)
	return x, y, np.trapz(y, dx = dx)

sig = 1
def test_P():
	s1 = P(-sig,sig,0,sig,f1)[-1]
	s2 = P(-2*sig,2*sig,0,sig,f1)[-1]
	s3 = P(-3*sig,3*sig,0,sig,f1)[-1]
	assert abs(0.68-s1)<0.005
	assert abs(0.95-s2)<0.005
	assert abs(0.997-s3)<0.0005
test_P()

def MaxBoz(v, m = 2.0159 * 1.66e-27, T = 3000):
	kB = 1.380649e-23
	e = - 0.5 * m * v ** 2 * (kB * T) ** (-1)
	p1 = m ** (3/2) * (2 * np.pi * kB * T) ** (-3/2)
	p3 = 4 * np.pi * v ** 2
	return p1 * np.exp(e) * p3

def PMaxBoz(a, b):
	n = 1E4+1
	dx = (b-a)/n
	x = np.linspace(a, b, n)
	y = MaxBoz(x)
	return x, y

"""
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
"""

width = 2.5E4
x, y, A = P(-width,width,0,sig_b,f1)
plt.plot(x,y)
plt.grid()
plt.show()

B = P(5E3,3E4,0,sig_b,f1)[-1]
print(B) #0.07755
print(B*N) #7755.38

xv,yv,sv = P(0,3E4,m,T,f2)
plt.plot(xv,yv)
plt.grid()
plt.show()
#a, s, d = FWHM(x, y, 1)
#print(s,d)


x, y = PMaxBoz(-10,10)
plt.plot(x,y)
plt.grid()
plt.show()
