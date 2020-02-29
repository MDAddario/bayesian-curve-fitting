from main import *

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
