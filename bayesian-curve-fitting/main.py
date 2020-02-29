import numpy as np
import matplotlib.pyplot as plt
from inspect import signature


def get_num_arguments(function):
	"""
	Return the number of arguments to a given function
	"""
	sig = signature(function)
	return len(sig.parameters)


def quadratic(x, a, b, c):
	"""
	Standard quadratic function
	"""
	return a*np.square(x) + b*x + c


def exponential_decay(x, tau):
	"""
	Exponential decay function in one parameter
	"""
	return np.exp(np.divide(-x, tau))


def generate_random_data(foo, args, errors, start, end, npoints):
	"""
	Generate data set of gaussian distributed data points
	Points are uniformally sampled along range of independent values
	"""

	# Upgrade argument input to a list if only one arg provided
	if not isinstance(args, (list, tuple)):
		args = [args]

	# Generate x values
	x_values = np.linspace(start, end, npoints)

	# Produce data set
	y_values = foo(x_values, *args)

	# Add gaussian noise
	y_values += np.random.normal(0, errors, npoints)

	return x_values, y_values


def plot_error_bars(axes, xvals, yvals, yerr, color='r', ecolor='k',
					elinewidth=0.7, capsize=3, capthick=0.5, s=15):
	"""
	It is complicated to get matplotlib to nicely plot errorbars for you
	Allow me to nicely wrap this all up in one function call
	"""

	axes.errorbar(xvals, yvals, yerr=yerr, linewidth=0, ecolor=ecolor, elinewidth=elinewidth,
				  capsize=capsize, capthick=capthick)
	axes.scatter(xvals, yvals, s=s, color=color)


def visualize_fit_data(x_values, y_values, y_errors, foo, args_fit, args_true):
	"""
	Plot data set and associated best fit
	"""
	# Upgrade argument input to a list if only one arg provided
	if not isinstance(args_fit, (list, tuple)):
		args_fit = [args_fit]
	if not isinstance(args_true, (list, tuple)):
		args_true = [args_true]

	# Plot data set
	fig = plt.figure(figsize=(16, 9))
	ax = fig.add_subplot((111))
	plot_error_bars(ax, x_values, y_values, y_errors)

	# Include a fit
	x_fit = np.linspace(x_values.min(), x_values.max(), num=200)
	y_fit = foo(x_fit, *args_fit)
	ax.plot(x_fit, y_fit, label='Best fit')

	# Include true relation
	x_true = x_fit
	y_true = foo(x_fit, *args_true)
	ax.plot(x_true, y_true, label='True relation')

	# Final touches
	plt.legend(framealpha=1.0, fontsize=18)
	plt.show()


if __name__ == '__main__':

	# Generate random data set to quadratic spread
	start = 0
	end = 5
	npoints = 10

	args_true = [3, -2, 5]
	args_fit = [3.2, -2, 4.8]
	y_error = 1

	x_values, y_values = generate_random_data(quadratic, args_true, y_error, start, end, npoints)
	visualize_fit_data(x_values, y_values, y_error, quadratic, args_fit, args_true)

	# Generate random data set to exponential spread
	start = 0
	end = 5
	npoints = 10

	args_true = 6
	args_fit = 5.6
	y_error = 1

	x_values, y_values = generate_random_data(exponential_decay, args_true, y_error, start, end, npoints)
	visualize_fit_data(x_values, y_values, y_error, exponential_decay, args_fit, args_true)
