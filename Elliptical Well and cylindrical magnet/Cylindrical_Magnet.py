import numpy as np
import sympy as sp
from scipy.integrate import quad
from sympt import RDSymbol, get_order

import sys
import os
sys.path.append("/home/leander/GoogleDriveSync/PhD/Python/")
from Packages.utils import T, Ik, multivariable_taylor_series, multivariable_taylor_series_parallel


def I(a, b, N):
    '''
    Define the function I(a, b, N) = \sum_{k=1}^{N} (-1)^k \binom{-1/2}{k} (b^{2k})/(a^{2k + 1}) I_k(k + 1)
    This is the N-th order expansion of the integral 
        I(a, b) = \int_{0}^{\pi} cos(theta) / sqrt(a^2 - b^2 cos(theta)) d\theta

    Parameters:
    ----------------
    a : sympy expression
        The parameter a in the integral 
            a = sqrt(xi^2 + R^2 + r^2)
    b : sympy expression
        The parameter b in the integral 
            b = sqrt(2rR)
    N : int
        The order of the expansion

    Returns:
    ----------------
    res : sympy expression
        The N-th order expansion of the integral I(a, b)
    '''

    res = 0
    for k in range(1, N+1):
        if (k + 1) % 2 != 0:
            continue
        res += sp.binomial(-sp.Rational(1,2), k) * (-1) ** k * (b**(2*k))/(a**(2*k + 1)) * Ik(k + 1)
    return res

def J(a, R, r, N):
    '''
    Define the function J(a, R, r, N) = \sum_{k=0}^{N} (2rR)^k (R I_k(k) - r I_k(k + 1)) \sum_{(n,m) \in T(k, 2)} \binom{-1/2}{n} (-1)^n / ((r^2 + R^2)^(m + 1) a^(2n + 1))
    This is the N-th order expansion of the integral
        J(a, R, r) = \int_{0}^{\pi} (R - r cos(theta)) /((r^2 + R^2 - 2rR\cos(\theta)) sqrt(a^2 - b^2 cos(theta)) d\theta

    Parameters:
    ----------------
    a : sympy expression
        The parameter a in the integral
    R : sympy expression
        The parameter R in the integral
    r : sympy expression
        The parameter r in the integral
    N : int
        The order of the expansion

    Returns:
    ----------------
    res : sympy expression
        The N-th order expansion of the integral J(a, R, r)
    '''

    res = 0
    for k in range(N+1):
        res_k = 0
        coeff = (2 * r * R)**k * (R * Ik(k) - r * Ik(k + 1))

        for (n, m) in T(k, 2):
            res_k += sp.binomial(-sp.Rational(1,2), n) * (-1)**n / ( (r**2 + R**2) ** (m + 1) * (a**(2*n + 1)))

        res += coeff * res_k

    return res


class CylindricalMagnet:
    '''
    Class to define the magnetic field of a cylindrical magnet

    Parameters:
    ----------------
    R : float
        The radius of the cylindrical magnet
    L : float
        The length of the cylindrical magnet
    position : sympy Matrix
        The position of the cylindrical magnet center in the global coordinate system
    '''

    def __init__(self, R, L, position : sp.Matrix):
        self.position = position
        self.geometry = {'R': R, 'L': L}
        self.B0 = sp.symbols('B0', real = True, positive = True)


    def B(self, values, r_charge):
        '''
        Calculate the magnetic field at a given position

        Parameters:
        ----------------
        values : dict
            The values of the parameters of the cylindrical magnet
        r_charge : numpy array
            The position of the charge in the global coordinate system

        Returns:
        ----------------
        B : numpy array
            The magnetic field at the position of the charge
        '''

        self.geometry['R'] = values['R']
        self.geometry['L'] = values['L']
        self.B0 = values['B0']

        self.position = values['position']


        rho = r_charge - self.position    # vector from magnet to charge

        x, y, z = rho
        r = np.sqrt(x**2 + y**2)

        phi = np.arctan2(y, x)

        xi_p = z + 1/2 * self.geometry['L']
        xi_m = z - 1/2 * self.geometry['L']

        fr_p = lambda theta: np.cos(theta) / np.sqrt(xi_p**2 + self.geometry['R']**2 + r**2 - 2 * self.geometry['R'] * r * np.cos(theta))
        fr_m = lambda theta: np.cos(theta) / np.sqrt(xi_m**2 + self.geometry['R']**2 + r**2 - 2 * self.geometry['R'] * r * np.cos(theta))

        fz_p = lambda theta: xi_p * (self.geometry['R'] - r * np.cos(theta)) / ((r**2 + self.geometry['R']**2 - 2*self.geometry['R']*r*np.cos(theta)) * np.sqrt(xi_p**2 + self.geometry['R']**2 + r**2 - 2 * self.geometry['R'] * r * np.cos(theta)))
        fz_m = lambda theta: xi_m * (self.geometry['R'] - r * np.cos(theta)) / ((r**2 + self.geometry['R']**2 - 2*self.geometry['R']*r*np.cos(theta)) * np.sqrt(xi_m**2 + self.geometry['R']**2 + r**2 - 2 * self.geometry['R'] * r * np.cos(theta)))

        Br = - 2 * self.B0 * self.geometry['R'] * (quad(fr_p, 0, sp.pi)[0] - quad(fr_m, 0, sp.pi)[0])
        Bz = 2 * self.B0 * self.geometry['R'] * (quad(fz_p, 0, sp.pi)[0] - quad(fz_m, 0, sp.pi)[0])

        self.r = r

        return Br * np.cos(phi) , Br * np.sin(phi), Bz
    
    def B_integral_expansion(self, r_charge: sp.Matrix, order=3):
        '''
        Calculate the magnetic field at a given position using the expansion of the integrals I and J

        Parameters:
        ----------------
        r_charge : sympy Matrix
            The position of the charge in the global coordinate system
        order : int
            The order of the expansion

        Returns:
        ----------------
        Bx : sympy expression
            The x-component of the magnetic field at the position of the charge
        By : sympy expression
            The y-component of the magnetic field at the position of the charge
        Bz : sympy expression
            The z-component of the magnetic field at the position of the charge
        '''

        rho = r_charge - self.position    # vector from magnet to charge

        x, y, z = rho
        r = sp.sqrt(x**2 + y**2)
        r_symbol = sp.symbols('r', real = True, positive = True)

        xi_p = z + sp.Rational(1, 2) * self.geometry['L']
        xi_m = z - sp.Rational(1, 2) * self.geometry['L']

        a_plus = sp.sqrt(xi_p**2 + self.geometry['R']**2 + r**2)
        a_minus = sp.sqrt(xi_m**2 + self.geometry['R']**2 + r**2)
        b = sp.sqrt(2 * r_symbol * self.geometry['R'])

        I_plus = sp.Add(*[term.collect(r_symbol) / r_symbol for term in I(a_plus, b, order).as_ordered_terms()]).subs(r_symbol, r)
        I_minus = sp.Add(*[term.collect(r_symbol) / r_symbol for term in I(a_minus, b, order).as_ordered_terms()]).subs(r_symbol, r)

        J_plus = xi_p * J(a_plus, self.geometry['R'], r_symbol, order).subs(r_symbol, r)
        J_minus = xi_m * J(a_minus, self.geometry['R'], r_symbol, order).subs(r_symbol, r)

        Bx = - 2 * self.B0 * self.geometry['R'] * (I_plus - I_minus) * x
        By = - 2 * self.B0 * self.geometry['R'] * (I_plus - I_minus) * y
        Bz = 2 * self.B0 * self.geometry['R'] * (J_plus - J_minus) 

        self.Bx = Bx
        self.By = By
        self.Bz = Bz

        return Bx, By, Bz

    def B_integral_expansion_linear(self, r, r0, taylor_order=3, parallel=False):
        
        # Check if there is the self.Bx
        if not hasattr(self, 'Bx') or not hasattr(self, 'By') or not hasattr(self, 'Bz'):
            raise ValueError("The B_integral_expansion function must be called before the B_integral_expansion_linear function")
        
        self.Bx_linear = BLinear(self.Bx, taylor_order, r, r0, 'B_x', parallel=parallel)
        self.By_linear = BLinear(self.By, taylor_order, r, r0, 'B_y', parallel=parallel)
        self.Bz_linear = BLinear(self.Bz, taylor_order, r, r0, 'B_z', parallel=parallel)

        return self.Bx_linear, self.By_linear, self.Bz_linear

class BLinear:
    def __init__(self, B, order, r, r0, name, symbol_orders=1, parallel=False):
        self.B = B
        self.order = order
        self.B_linear_dict = multivariable_taylor_series_parallel(B, r, r0, order, True) if parallel else multivariable_taylor_series(B, r, r0, order, True)
        self.B_linear = {}
        self.B_linear_values = {}
        self.symbol_orders = symbol_orders

        for key, terms in self.B_linear_dict.items():
            for value, op in terms:
                if value == 0:
                    continue
                b = RDSymbol(f'{{{name}}}' + f'^{{({str(op)})}}', real=True, order=self.symbol_orders)
                self.B_linear_values[b] = value
                self.B_linear[key] = self.B_linear.get(key, 0) + b * op

    def get(self, order):
        return self.B_linear.get(order, 0)
