from sympt import RDSymbol

import dill as pickle

import numpy as np

# factorial
from math import factorial
import re

# Load the Taylor series expansion coefficients
with open('taylor_series.pkl', 'rb') as f:
    taylor_series = pickle.load(f)


b = lambda x, ops: RDSymbol(f'b_{x}^{{({ops})}}', order=1, real=True)

def compute_b(order, alpha, x, y):
    """
    Compute the b coefficients for a given order and variables.

    Parameters:
    ----------
    order : int
        The order of the expansion.
    alpha : str
        The variable representing the expansion parameter.
    x : sympy.Symbol
        The x variable.
    y : sympy.Symbol
        The y variable.
    lx : sympy.Symbol
        The characteristic length scale in the x direction.
    """

    res = 0
    for n in range(order + 1):
        for m in range(order + 1):
            if n + m > order:
                continue
            upper_label = f'{x}^{n}{y}^{m}' if n != 0 and m != 0 else f'{x}^{n}' if n != 0 else f'{y}^{m}' if m != 0 else '0'
            res += b(alpha, upper_label) * x**n * y**m
    return res  


def compute_bk_values(D, L, Dx, Dy, Dz, B0):
    """
    Compute the b_k values for a given set of parameters.

    Parameters:
    ----------
    D : float
        The diameter of the system.
    L : float
        The length of the system.
    Dx : float
        The x-component of the displacement.
    Dy : float
        The y-component of the displacement.
    Dz : float
        The z-component of the displacement.
    B0 : float
        The external magnetic field strength.

    Returns:
    -------
    dict
        A dictionary containing the computed b_k values.
    """

    bx_taylor_series = taylor_series['Bx']
    by_taylor_series = taylor_series['By']
    bz_taylor_series = taylor_series['Bz']

    b_values = {
        'x' : bx_taylor_series,
        'y' : by_taylor_series,
        'z' : bz_taylor_series
    }

    symbols_bk_subs = {}

    for axis, b_series in b_values.items():
        for k, bf in b_series.items():
            for n, bf_nm in enumerate(bf):
                m = k - n
                upper_label = f'x^{n}y^{m}' if n != 0 and m != 0 else f'x^{n}' if n != 0 else f'y^{m}' if m != 0 else '0'
                symbols_bk_subs[b(axis, upper_label)] = bf_nm(D/2, L, Dx, Dy, Dz, B0)

    return symbols_bk_subs


def derivatives_2d_center_numpy(f, h, n_order=4, axis='x'):
    """
    Compute all derivatives of f up to n_order at the center using np.gradient.
    Returns a dictionary with keys like 'x^n y^m'.
    
    Parameters:
        f : 2D numpy array
        hx, hy : grid spacings
        n_order : maximum total derivative order (default 4)
        axis : which axis (x, y, or z) the function represents (for labeling)
    """
    result = {}
    i0 = f.shape[0] // 2
    j0 = f.shape[1] // 2

    # Cache gradients
    gradients = {}  # key: (n,m), value: derivative array

    # Zeroth derivative
    gradients[(0,0)] = f
    result[b(axis, '0')] = f[i0,j0]

    # Loop over total order
    for k in range(1, n_order+1):
        for n in range(k+1):
            m = k - n
            # Construct label
            if n != 0 and m != 0:
                label = f'x^{n}y^{m}'
                g = gradients[(n - 1, m)]
                gx = np.gradient(g, h, axis=1)
                gradients[(n,m)] = gx
            elif n != 0:
                label = f'x^{n}'
                g = gradients[(n-1,m)]
                gx = np.gradient(g, h, axis=1)
                gradients[(n,m)] = gx

            elif m != 0:
                label = f'y^{m}'
                g = gradients[(n,m-1)]
                gy = np.gradient(g, h, axis=0)
                gradients[(n,m)] = gy


            coeff = 1 / (factorial(n) * factorial(m))
            deriv = gradients[(n,m)]
            
            result[b(axis, label)] = coeff * deriv[i0,j0]

    return result

import numdifftools as nd
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator


def derivatives_2d_center_numdifftools(f_interp, dx, n_order=4, axis='f'):
    """
    Compute all derivatives of a 2D interpolated function f_interp up to n_order at the center
    using numdifftools. Returns a dictionary of Taylor coefficients.

    Parameters:
        f_interp : callable
            2D interpolating function, f_interp(y, x) or f_interp([x, y])
        dx : float
            Grid spacing (used as step size for finite differences)
        n_order : int
            Maximum total derivative order (default 4)
        axis : str
            Label for the function (used in dict keys)
    """
    result = {}

    x0 = 0
    y0 = 0

    # Wrapper for numdifftools
    def func(xy):
        return float(f_interp([xy[1], xy[0]]))  # RegularGridInterpolator expects (y, x)

    # Zeroth derivative
    result[b(axis, '0')] = func([x0, y0])

    # Loop over total derivative order
    for k in range(1, n_order+1):
        for n in range(k+1):
            m = k - n
            label = f"x^{n}y^{m}" if n != 0 and m != 0 else f"x^{n}" if n != 0 else f"y^{m}"

            if n > 0 and m > 0:
                # Mixed derivative: compute x-derivative at each y
                def mixed_derivative(y):
                    deriv_x = nd.Derivative(lambda x: func([x, y]), n=n, step=dx, method='central', order=4)
                    return deriv_x(x0)
                deriv = nd.Derivative(mixed_derivative, n=m, step=dx, method='central', order=4)
                val = deriv(y0)
            elif n > 0:
                # Only x derivative
                deriv = nd.Derivative(lambda x: func([x, y0]), n=n, step=dx, method='central', order=4)
                val = deriv(x0)
            else:
                # Only y derivative
                deriv = nd.Derivative(lambda y: func([x0, y]), n=m, step=dx, method='central', order=4)
                val = deriv(y0)

            # Taylor coefficient
            coeff = val / (factorial(n) * factorial(m))
            result[b(axis, label)] = coeff

    return result






def numerical_taylor_series(B, n_order, dx, method='fd', **kwargs):

    """
    Compute the numerical Taylor series expansion of a function B up to a given order.

    Parameters:
    ----------
    B : list of numpy.ndarray
        The function values at different points. [Bx, By, Bz]
    n_order : int
        The order of the Taylor series expansion.
    dx : float
        The step size for the finite difference approximation.
    Returns:
    -------
    dict
        A dictionary containing the Taylor series coefficients for Bx, By, and Bz.
    """

    symbols_bk = {}

    for axis, b_axis in zip(['x', 'y', 'z'], B):
        if method == 'numpy':
            symbols_bk.update(derivatives_2d_center_numpy(b_axis, dx, axis=axis, n_order=n_order))
        elif method == 'ndtools':
            X = kwargs.get('X', None)
            Y = kwargs.get('Y', None)
            x_vals = X[0, :]  # X along columns
            y_vals = Y[:, 0]  # Y along rows
            f_interp = RegularGridInterpolator((y_vals, x_vals), b_axis, method='linear')
            symbols_bk.update(derivatives_2d_center_numdifftools(f_interp, dx, axis=axis, n_order=n_order))
        else:
            raise ValueError("Method must be 'fd' or 'numpy'")

    return symbols_bk
        
