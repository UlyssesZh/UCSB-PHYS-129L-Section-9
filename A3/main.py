#!/usr/bin/env python

from os.path import isfile

from numpy import pi, cos, sin, linspace, exp
from numpy.random import seed as srand, normal, rand
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# A
def H(theta):
	return theta**4 - 8*theta**2 - 2*cos(4*pi*theta)
def dH(theta):
	return 4*theta**3 - 16*theta + 8*pi*sin(4*pi*theta)

def basic_plot():
	fig, ax = plt.subplots()
	ax.set_xlabel(r'$\theta$')
	ax.set_ylabel(r'$H(\theta)$')
	ax.set_xlim(-3, 3)
	theta_span = linspace(-3, 3, 301)
	ax.plot(theta_span, H(theta_span))
	return fig, ax

def gradient_descent(theta0, rates, filename):
	if isfile(filename):
		return
	fig, ax = basic_plot()
	line, = ax.plot([], [], c='r', marker='o')
	time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
	theta = theta0
	def init_func():
		line.set_data([theta], [H(theta)])
		return line,
	def update(rate):
		nonlocal theta
		theta -= rate * dH(theta)
		line.set_data([theta], [H(theta)])
		time_text.set_text(f'{rate=:.4f}')
		return line, time_text
	ani = FuncAnimation(fig, update, frames=rates, init_func=init_func, blit=True)
	ani.save(filename, writer='ffmpeg', fps=10)

gradient_descent(-1, 1/linspace(50,200,31), 'gradient_descent_-1.mkv')
gradient_descent(0.5, 1/linspace(50,200,31), 'gradient_descent_0.5.mkv')
gradient_descent(3, 1/linspace(50,200,31), 'gradient_descent_3.mkv')

# B
def metropolis_hastings(theta0, beta, filename, steps=30, seed=1108):
	if isfile(filename):
		return
	srand(seed)
	fig, ax = basic_plot()
	line, = ax.plot([], [], c='r', marker='o')
	time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
	theta = theta0
	h = H(theta)
	t = 0
	def init_func():
		line.set_data([theta], [h])
		return line,
	def update(i):
		nonlocal theta, h, t
		while True:
			new_theta = theta + normal()
			new_h = H(new_theta)
			r = exp(-beta*(new_h - h))
			t += 1
			if r > 1 or rand() < r:
				theta, h = new_theta, new_h
				break
		line.set_data([theta], [h])
		time_text.set_text(f'{t=}')
		return line, time_text
	ani = FuncAnimation(fig, update, frames=steps, init_func=init_func, blit=True)
	ani.save(filename, writer='ffmpeg', fps=10)

metropolis_hastings(-1, 1, 'metropolis_hastings_-1_1.mkv')
metropolis_hastings(0.5, 1, 'metropolis_hastings_0.5_1.mkv')
metropolis_hastings(3, 1, 'metropolis_hastings_3_1.mkv')
metropolis_hastings(-1, 3, 'metropolis_hastings_-1_3.mkv')
metropolis_hastings(0.5, 3, 'metropolis_hastings_0.5_3.mkv')
metropolis_hastings(3, 3, 'metropolis_hastings_3_3.mkv')

# C
def annealing(theta0, beta_list, filename, seed=1108):
	if isfile(filename):
		return
	srand(seed)
	fig, ax = basic_plot()
	line, = ax.plot([], [], c='r', marker='o')
	time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
	theta = theta0
	h = H(theta)
	def init_func():
		line.set_data([theta], [h])
		return line,
	def update(beta):
		nonlocal theta, h
		new_theta = theta + normal()
		new_h = H(new_theta)
		r = exp(-beta*(new_h - h))
		if r > 1 or rand() < r:
			theta, h = new_theta, new_h
		line.set_data([theta], [h])
		time_text.set_text(f'{beta=:.4f}')
		return line, time_text
	ani = FuncAnimation(fig, update, frames=beta_list, init_func=init_func, blit=True)
	ani.save(filename, writer='ffmpeg', fps=10)

annealing(-1, linspace(1,2,30), 'annealing_-1.mkv')
annealing(0.5, linspace(1,2,30), 'annealing_0.5.mkv')
annealing(3, linspace(1,2,30), 'annealing_3.mkv')
