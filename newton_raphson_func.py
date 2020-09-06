# Newton-Raphson Root Finder.
import numpy as np
from finite_differences import finite_difference

# Create empty list to store x_k.
guessed = []


def newton_raphson(f, start, h, tol, dydx='center_diff'):
  """
  Implementation of the Newton-Raphson algorithm for zero finding.
  f : function
      Function for which the root is to be found.
  x : number or vector
      Either a number for which the finite difference is to be computed, or a vector of values over which to compute the differences of f.
  h : number
      Step size in difference formula
  dydx : a derivative or finite difference method.
         Choices include 'center_diff', 'back_diff', and 'forward_diff'.
         Defaults to the 'center_diff' method.
         The user may also define their derivative explicitly as a function of x.
  tol : number
        Threshold past which the algorithm terminates.
  start : number
          Starting guess.
  """

  # Set derivative method.
  if dydx == 'center_diff':
    dydx = finite_difference(f, start, h).center_diff()
  elif dydx == 'back_diff':
    dydx = finite_difference(f, start, h).back_diff()
  elif dydx == 'forward_diff':
    dydx = finite_difference(f, start, h).forward_diff()
  else:
    dydx = dydx(start)

  # Check to see if initial guess is already within tolerance.
  if abs(f(start)) < tol or (start in guessed):
    return str(start) + " is a zero of f."
  # Compute next x value and recursively call newton_raphson.
  else:
    guessed.append(start)
    x_next = start - f(start) / dydx
    return newton_raphson(f=f,
                          start=x_next,
                          h=h,
                          tol=tol,
                          dydx='center_diff')


# Test code
# def f(x):
#   return x**2


# start = -4
# h = 0.01
# tol = 0.0001
# print(newton_raphson(f, start, h, tol, dydx=lambda x: 2 * x))
