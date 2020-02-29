import numpy as np
import matplotlib.pyplot as plt
from inspect import signature


def get_num_arguments(function):
	"""
	Return the number of arguments to a given function
	EXCLUDES argument for independent variable
	"""
	sig = signature(function)
	return len(sig.parameters) - 1


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


def bayesian_curve_fit(x_values, y_values, y_errors, foo, prior_bounds, npoints):
	"""
	Perform bayesian inference to determine best fit parameters,
	assuming that measurement errors are gaussian distributed
	"""
	# Upgrade error input to a list if only one error provided
	if not isinstance(y_errors, (list, tuple)):
		y_errors = [y_errors] * len(x_values)
	
	# Compute uniform priors
	prior_points = []
	priors = []
	
	for i, (start, end) in enumerate(prior_bounds):

		prior_points.append(npoints + i)
		priors.append(np.linspace(start, end, prior_points[i]))
	
	# Create mesh grids for each parameter
	arg_grids = np.meshgrid(*priors)

	# Compute the posterior
	exp_argument = 0

	for i, (x_val, y_val, y_err) in enumerate(zip(x_values, y_values, y_errors)):

		# Compute the fit's prediction
		y_fit = foo(x_val, *arg_grids)
		exp_argument -= np.square(y_val - y_fit) / 2 / np.square(y_err)

	posterior = np.exp(exp_argument)
	
	# Determine generalized volume element
	element = 1
	for i in range(get_num_arguments(foo)):
		element *= priors[i][1] - priors[i][0]

	# Normalize!
	posterior /= np.sum(posterior) * element

	# Marginalize for each parameter

if __name__ == '__main__':

	# Generate random data set to quadratic spread
	start = 0
	end = 5
	npoints = 10

	args_true = [3, -2, 5]
	y_error = 1

	x_values, y_values = generate_random_data(quadratic, args_true, y_error, start, end, npoints)
	
	# Come up with bounds on our estimates
	prior_bounds = [[-10, 10], [-10, 10], [-10, 10]]
	npoints = 21
	
	# Bayesian curve fit the data
	args_fit = bayesian_curve_fit(x_values, y_values, y_error, quadratic, prior_bounds, npoints)
	exit()
	
	# Visualize the results
	visualize_fit_data(x_values, y_values, y_error, quadratic, args_fit, args_true)

