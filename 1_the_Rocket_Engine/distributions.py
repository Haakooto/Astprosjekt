import numpy as np
import matplotlib.pyplot as plt
import ast2000tools.constants as const
import sys

n = 1e4
m = const.m_H2
T = 3000
k = const.k_B
sig_b = np.sqrt(k*T/m)


def gauss(x, sig, mu):
	return (np.sqrt(2 * np.pi) * sig)**(-1) * np.exp(-0.5 * (x - mu)**2 * sig**(-2))

def max_boz(V, T, m):
	return (m/(2*np.pi*k*T))**(3/2)*np.exp(-m*V**2/(2*k*T))*4*np.pi*V**2

def max_boz_x(v, T, m):
	return np.sqrt(m/(2*np.pi*k*T))*np.exp(-0.5*m*v**2/(k*T))

def P_g(a, b, mu, sig):
	dx = (b-a)/n
	x = np.linspace(a, b, n)
	y = gauss(x, sig, mu)
	return x, y, np.trapz(y, dx = dx)

def P_mb(a, b, T=T, m=m):
	assert b <= a, "a must be smaller than b and not the same"

	if a == -b:
		f = max_boz_x
	elif a == 0:
		f = max_boz

	dx = (b-a)/n
	x = np.linspace(a, b, n)
	y = f(x, T, m)
	return x, y, np.trapz(y, dx = dx)


def test_P_g():
	sig = 1
	mu = 0
	tol = 0.005

	s1 = P_g(-sig, sig, mu, sig)
	s2 = P_g(-2*sig, 2*sig, mu, sig)
	s3 = P_g(-3*sig, 3*sig, mu, sig)

	assert abs(0.68-s1[2]) < tol
	assert abs(0.95-s2[2]) < tol
	assert abs(0.997-s3[2]) < tol

def test_P_mb():
	pass

def main():
	test_P_g()

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

#x, y, A = P_mb(-15000,15000)
#print(A)
#plt.plot(x,y)
#plt.show()

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
"""
if __name__ == "__main__":
	main()
