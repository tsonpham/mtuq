
import numpy as np

def linear_transform(i, a, b):
    """ Linear map suggested by N. Hansen for appropriate parameter scaling/variable encoding in CMA-ES

    Linear map from [0;10] to [a,b]

    source:
    (https://cma-es.github.io/cmaes_sourcecode_page.html)
    """
    transformed = a + (b-a) * i / 10
    return transformed

def inverse_linear_transform(transformed, a, b):
    """ Inverse linear mapping to reproject the variable in the [0; 10] range, from its original transformation bounds.

    """
    i = (10*(transformed-a))/(b-a)
    return i

def array_in_bounds(array, a, b):
    """
    Check if all elements of an array are in bounds.
    :param array: The array to check.
    :param a: The lower bound.
    :param b: The upper bound.
    :return: True if all elements of the array are in bounds, False otherwise.
    """
    for i in range(len(array)):
        if not in_bounds(array[i], a, b):
            return False
    return True
