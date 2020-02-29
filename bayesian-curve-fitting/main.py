import numpy as np
import matplotlib.pyplot as plt
from inspect import signature


def get_num_arguments(function):
	"""
	Return the number of arguments to a given function
	"""
	sig = signature(function)
	return len(sig.parameters)


def nonsense(arg_1, arg_2):
	pass


if __name__ == '__main__':

	num_args = get_num_arguments(nonsense)
	print(num_args)
