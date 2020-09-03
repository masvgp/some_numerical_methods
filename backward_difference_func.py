import numpy as np


def back_diff(f, x, h):
  '''
  Implementation of the backward difference formula with step size h.
  f : function
    Vectorized function of a single variable
  x : vector
      a vector of values over which to compute the differences of f at x
  h : number
      Step size in difference formula
  '''

  return (f(x) - f(x - h)) / h


# # Matrix version
# A = np.eye(5)
# for i in range(4):
#   A[i + 1, i] = -1

# x = square(np.arange(5))

# dydx = np.dot(A, x)
# print(dydx)

# A_inv = np.linalg.inv(A)

# x = np.dot(A_inv, dydx)

# print(x)
