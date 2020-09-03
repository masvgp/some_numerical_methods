# Finite difference class
# import numpy as np


class finite_difference(object):
    """
    Class for finite differences.
    f : function
        Function for which a finite difference is to be computed.
    x : number or vector
        Either a number for which the finite difference is to be computed, or a vector of values over which to compute the differences of f.
    h : number
        Step size in difference formula
    """

    def __init__(self, f, x, h):
        #super(finite_difference, self).__init__()
        self.f = f
        self.x = x
        self.h = h

    def center_diff(self):
        '''
        Implementation of the center difference formula with step size h.
        Formula : (f(x + 1/2 * h) - f(x - 1/2 * h))/h
        f : function
            Function for which a center difference is to be computed.
        x : number or vector
            Either a number for which the backward difference is to be computed, or a vector of values over which to compute the differences of f.
        h : number
            Step size in difference formula
        '''
        # Copy variables to make function more readable.
        f, x, h = self.f, self.x, self.h
        return (f(x + (0.5) * h) - f(x - (0.5) * h)) / h

    def back_diff(self):
        '''
         Implementation of the backward difference formula with step size h.
         Formula : (f(x) - f(x - h))/h
         f : function
             Function for which a backward difference is to be computed.
         x : number or vector
             Either a number for which the backward difference is to be computed, or a vector of values over which to compute the differences of f.
         h : number
             Step size in difference formula
         '''
        f, x, h = self.f, self.x, self.h
        return (f(x) - f(x - h)) / h

    def forward_diff(self):
        '''
         Implementation of the center difference formula with step size h.
         Formula : (f(x + h) - f(x))/h
         f : function
             Function for which a center difference is to be computed.
         x : number or vector
             Either a number at which the backward difference is to be computed, or a vector of values over which to compute the differences of f.
         h : number
             Step size in difference formula
         '''
        f, x, h = self.f, self.x, self.h
        return (f(x + h) - f(x)) / h

# Test the class
# print(finite_difference(f=lambda x: x ** 2, x=4, h=1).back_diff())
