#!/usr/bin/env python

from numpy import exp, isclose, inf
from scipy import constants
from scipy.integrate import fixed_quad, quad

# AB
integral = fixed_quad(lambda z: z**3/(1-z)**5/(exp(z/(1-z))-1), 0, 1, n=40)[0]
stefan = 2*constants.pi*constants.k**4/(constants.c**2*constants.h**3) * integral
print(f'Stefan-Boltzmann constant = {stefan:.6e} W/m^2/K^4')
if isclose(stefan, constants.Stefan_Boltzmann):
	print('The value is consistent with the scipy.constants value.')

# C
integral2 = quad(lambda x: x**3/(exp(x)-1), 0, inf)[0]
if isclose(integral, integral2):
	print('The two integrals are consistent.')
