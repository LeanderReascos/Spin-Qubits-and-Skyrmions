import numpy as np
import matplotlib.pyplot as plt
import json
import sympy as sp
from sympy.physics.quantum import Operator
from sympy.physics.paulialgebra import evaluate_pauli_product, Pauli

import sys
sys.path.append('../Packages/PySW/')

from Modules.sympy.classes import *
from Modules.sympy.utils import *
from Modules.sympy.truncated.solver import *


# Constants for the simulation (Natural Units)

h_bar = 1.0545718 * 1e-34
e = 1.60217662 * 1e-19
electron_charge = -e
m_0= 9.10938356 * 1e-31
mu_0 = 4 * np.pi * 1e-7

gamma = 0.19 # Effective mass of the electron in Silicon
m = gamma * m_0
g = 2
mu_b = e * h_bar / (2 * m_0)


class CylindricalMagnet:
    def __init__(self, M, R, L, Delta_z):
        self.B0 = mu_0 * M / (4*np.pi)

        self.M = M
        self.R = R
        self.L = L
        self.Delta_z = Delta_z


    def __str__(self):
        return f"Cylindrical Magnet with M = {self.M}, R = {self.R}, L = {self.L}, Delta_z = {self.Delta_z}"
    
    def __repr__(self):
        return f"Cylindrical Magnet with M = {self.M}, R = {self.R}, L = {self.L}, Delta_z = {self.Delta_z}"
    
    def update_parameters(self, M, R, L, Delta_z):
        self.B0 = mu_0 * M /(4*np.pi)
        self.update_geometry(R, L, Delta_z)

    def update_geometry(self, R, L, Delta_z):
        self.R = R
        self.L = L
        self.Delta_z = Delta_z

    def b_0(self, R=None, L=None, z=None):
        R = self.R if R is None else R
        L = self.L if L is None else L
        z = self.Delta_z if z is None else z
        return 2 * np.pi * self.B0 * ( (L - 2*z)/np.sqrt((L - 2*z)**2 + 4*R**2)  +  (L + 2*z)/np.sqrt((L + 2*z)**2 + 4*R**2) )

    def b_1(self, R=None, L=None, z=None):
        R = self.R if R is None else R
        L = self.L if L is None else L
        z = self.Delta_z if z is None else z
        return np.abs( 8 * np.pi * self.B0 * R**2 * ( 1/((L + 2*z)**2 + 4*R**2)**(3/2) - 1/((L - 2*z)**2 + 4*R**2)**(3/2)  ) ) 

    def b_2(self, R=None, L=None, z=None):
        R = self.R if R is None else R
        L = self.L if L is None else L
        z = self.Delta_z if z is None else z
        return np.abs(24 * np.pi * self.B0 * R**2 * (  (L - 2*z)/((L - 2*z)**2 + 4*R**2)**(5/2) + (L + 2*z)/((L + 2*z)**2 + 4*R**2)**(5/2) ))
    

NM = 1e-9

class Hamiltonian2D_Elliptical_Quantum_Well:
    def __init__(self, lx, ly, Bz, CyMagnet:CylindricalMagnet, minimal_coupling = True, strenght_minimal_coupling = 1):
        self.lx = lx
        self.ly = ly

        self.omega_x = h_bar / (m * (lx * NM)**2)
        self.omega_y = h_bar / (m * (ly * NM)**2)

        self.Bz = Bz
        self.CyMagnet = CyMagnet
        self.minimal_coupling = minimal_coupling
        self.strenght_minimal_coupling = strenght_minimal_coupling

        self.compute_parameters()

    def compute_parameters(self, strenght_minimal_coupling = None):
        if strenght_minimal_coupling is not None:
            self.strenght_minimal_coupling = strenght_minimal_coupling

        self.B = self.Bz + self.CyMagnet.b_0() if self.minimal_coupling else 0
        self.B = self.B * self.strenght_minimal_coupling
        self.Omega_z = g * mu_b / h_bar * (self.Bz + self.CyMagnet.b_0())
        self.omega_z = g * mu_b / h_bar * self.CyMagnet.b_2() * 1/NM**2
        self.omega = g * mu_b / h_bar * self.CyMagnet.b_1() * 1/NM

        self.omega_c = electron_charge * self.B / m

        self.Omega_x_2 = self.omega_x**2 + (self.omega_c / 2)**2
        self.Omega_y_2 = self.omega_y**2 + (self.omega_c / 2)**2
        self.Omega_3_2 = np.sqrt((self.Omega_x_2 - self.Omega_y_2)**2 + 8*(self.omega_c / 2)**2 * (self.Omega_x_2 + self.Omega_y_2))

        self.Omega_x = np.sqrt(self.Omega_x_2)
        self.Omega_y = np.sqrt(self.Omega_y_2)
        self.Omega_3 = np.sqrt(self.Omega_3_2)

        self.beta = - m * np.sqrt((self.Omega_x_2 + self.Omega_y_2)/2)
        self.alpha = 1/self.beta
        self.theta = np.arctan( 2*(self.omega_c / 2) * np.sqrt(2*self.Omega_x_2 + 2*self.Omega_y_2) / (self.Omega_x_2 - self.Omega_y_2) ) / 2 if self.Omega_x_2 != self.Omega_y_2 else -np.pi/4
        self.theta = self.theta if self.minimal_coupling and self.B != 0 else 0

        self.omega_1 = 1/np.sqrt(2) * np.sqrt(self.omega_x**2 + self.omega_y**2 + self.omega_c**2 + np.sqrt((self.omega_x**2 - self.omega_y**2)**2 + 2 * self.omega_c**2 * (self.omega_x**2 + self.omega_y**2) + self.omega_c**4))
        self.omega_2 = 1/np.sqrt(2) * np.sqrt(self.omega_x**2 + self.omega_y**2 + self.omega_c**2 - np.sqrt((self.omega_x**2 - self.omega_y**2)**2 + 2 * self.omega_c**2 * (self.omega_x**2 + self.omega_y**2) + self.omega_c**4))

        self.c1_2 = ( self.Omega_x_2 + 3 * self.Omega_y_2 + self.Omega_3_2 ) / (2 * (self.Omega_x_2 + self.Omega_y_2) )
        self.c2_2 = ( 3 * self.Omega_x_2 + self.Omega_y_2 - self.Omega_3_2 ) / (2 * (self.Omega_x_2 + self.Omega_y_2) )
        self.Omega_1_2 = 1/4 * (3*self.Omega_x_2 + self.Omega_y_2 + self.Omega_3_2)
        self.Omega_2_2 = 1/4 * (self.Omega_x_2 + 3*self.Omega_y_2 - self.Omega_3_2)

        self.c1 = np.sqrt(self.c1_2)
        self.c2 = np.sqrt(self.c2_2)
        self.Omega_1 = np.sqrt(self.Omega_1_2)
        self.Omega_2 = np.sqrt(self.Omega_2_2)

        self.u1 = np.sqrt(h_bar * self.c1 / (m * self.Omega_1))
        self.u2 = np.sqrt(h_bar * self.c2 / (m * self.Omega_2))
        self.v1 = np.sqrt(h_bar * m * self.Omega_1/self.c1)
        self.v2 = np.sqrt(h_bar * m * self.Omega_2/self.c2)

        self.omega_d_2 = (self.Omega_x_2 - self.Omega_y_2 + self.Omega_3_2)/(2 * self.Omega_3_2) if self.minimal_coupling and self.B != 0 else 1
        self.d_2 = (-self.Omega_x_2 + self.Omega_y_2 + self.Omega_3_2)/(self.Omega_3_2 * m**2 * (self.Omega_x_2 + self.Omega_y_2)) if self.minimal_coupling and self.B != 0 else 0


        self.delta_plus_1 = self.omega_z / 2 * (self.u1**2 * self.omega_d_2 + self.v1**2 * self.d_2)
        self.delta_plus_2 = self.omega_z / 2 * (self.u2**2 * self.omega_d_2 + self.v2**2 * self.d_2)

        self.delta_minus_1 = self.omega_z / 2 * (self.u1**2 * self.omega_d_2 - self.v1**2 * self.d_2) if self.lx != self.ly else 0
        self.delta_minus_2 = self.omega_z / 2 * (self.u2**2 * self.omega_d_2 - self.v2**2 * self.d_2) if self.lx != self.ly else 0

        self.tp = self.omega_z * self.omega_c / (2*self.Omega_3_2 * m) * (self.v1 * self.u2 + self.u1 * self.v2) if self.minimal_coupling and self.B != 0 else 0
        self.tm = self.omega_z * self.omega_c / (2*self.Omega_3_2 * m) * (self.v1 * self.u2 - self.u1 * self.v2) if self.minimal_coupling and self.B != 0 and self.lx != self.ly else 0

        self.f1 = self.omega / (2 * np.sqrt(2)) * self.u1 * np.cos(self.theta)
        self.f2 = self.omega / (2 * np.sqrt(2)) * self.u2 * np.cos(self.theta)

        self.g1 = self.omega / (2 * np.sqrt(2)) * self.v1 * np.sin(self.theta) * self.alpha
        self.g2 = self.omega / (2 * np.sqrt(2)) * self.v2 * np.sin(self.theta) * self.alpha


        # ---- Co-Moving Frame ----
        def f(x, y):
            numerator = 2*self.Omega_z * (x**2 - self.Omega_z**2) + self.Omega_z * self.omega_c**2 - self.omega_c * (y**2 + self.Omega_z**2)
            denominator = 2* (self.omega_1**2 - self.Omega_z**2) * (self.omega_2**2 - self.Omega_z**2)
            return numerator / denominator
    
        self.fxy = f(self.omega_x, self.omega_y)
        self.fyx = f(self.omega_y, self.omega_x)


        self.mu1 = self.u1 / np.sqrt(2) * self.beta * np.sin(self.theta)
        self.mu2 = self.u2 / np.sqrt(2) * self.beta * np.sin(self.theta)
        self.eta1 = self.v1 / np.sqrt(2) *  np.cos(self.theta)
        self.eta2 = self.v2 / np.sqrt(2) *  np.cos(self.theta)

        if hasattr(self, 'numeric_hamiltonian'):
            del self.numeric_hamiltonian
        
        


    def get_symbolic_hamiltonian(self, dim_bosonic=3):
        dim_spin = 2

        self.dim_bosonic = dim_bosonic
        self.total_dim = dim_spin * dim_bosonic**2

        Spin = RDBasis("\\sigma", 'spin', dim = dim_spin)
        self.Spin = Spin
        s0, sx, sy, sz = Spin._basis

        a_1 = RDBoson("a_1", subspace ="boson1", dim_projection=dim_bosonic)
        ad_1 = RDBoson("{a_1^\\dagger}", subspace ="boson1", is_annihilation=False, dim_projection=dim_bosonic)
        a_2 = RDBoson("a_2", subspace ="boson2", dim_projection=dim_bosonic)
        ad_2 = RDBoson("{a_2^\\dagger}", subspace ="boson2", is_annihilation=False, dim_projection=dim_bosonic)

        self.commutation_relations = {
            a_1 * ad_1 : ad_1 * a_1 + 1,
            a_2 * ad_2 : ad_2 * a_2 + 1,
            a_2 * ad_1 : ad_1 * a_2,
            ad_2 * a_1 : a_1 * ad_2,
            ad_2 * ad_1 : ad_1 * ad_2,
            a_2 * a_1 : a_1 * a_2
        }

        hbar = RDsymbol("hbar", order = 0)
        omega_1 = RDsymbol("omega_1", order = 0)
        omega_2 = RDsymbol("omega_2", order = 0)
        Omega_z = RDsymbol("\\Omega_{z}", order = 0)

        H0 = hbar * omega_1 * ad_1 * a_1 + hbar * omega_2 * ad_2 * a_2 - sp.Rational(1,2) * hbar * Omega_z * sz

        f1 = RDsymbol("f_1", order = 1)
        f2 = RDsymbol("f_2", order = 1)
        g1 = RDsymbol("g_1", order = 1)
        g2 = RDsymbol("g_2", order = 1)

        V1 = hbar * f1 * (ad_1 + a_1) * sx - I * hbar * g2 * (ad_2 - a_2) * sx + hbar * f2 * (ad_2 + a_2) * sy - I * hbar * g1 * (ad_1 - a_1) * sy

        delta_1 = RDsymbol("\\delta_1", order = 2)
        delta_2 = RDsymbol("\\delta_2", order = 2)

        delta_m_1 = RDsymbol("\\delta_{-1}", order = 2)
        delta_m_2 = RDsymbol("\\delta_{-2}", order = 2)

        H2 = sp.Rational(1, 2) * hbar * delta_1 *sz + sp.Rational(1, 2) * hbar * delta_2 *sz + hbar * delta_1 * ad_1 * a_1 * sz + hbar * delta_2 * ad_2 * a_2 * sz

        tp = RDsymbol("t_{+}", order = 2)
        tm = RDsymbol("t_{-}", order = 2)

        V2 = sp.Rational(1,2) * hbar * delta_m_1 * (ad_1**2 + a_1**2) * sz + sp.Rational(1,2) * hbar * delta_m_2 * (ad_2**2 + a_2**2) * sz + I * hbar * tp * (ad_1*ad_2 - a_1*a_2) * sz + I * hbar * tm * (ad_1*a_2 - a_1*ad_2) * sz

        self.subs_aux_dict = {
            hbar : 1,
            omega_1 : self.omega_1,
            omega_2 : self.omega_2,
            Omega_z : self.Omega_z,
            f1 : self.f1,
            f2 : self.f2,
            g1 : self.g1,
            g2 : self.g2,
            delta_1 : self.delta_plus_1,
            delta_2 : self.delta_plus_2,
            delta_m_1 : self.delta_minus_1,
            delta_m_2 : self.delta_minus_2,
            tp : self.tp,
            tm : self.tm
        }

        self.subs_dict = self.subs_aux_dict.copy()
        self.subs_dict.update({hbar : h_bar})

        self.symbols_dict = {
            'hbar' : hbar,
            'omega_1' : omega_1,
            'omega_2' : omega_2,
            'Omega_z' : Omega_z,
            'f1' : f1,
            'f2' : f2,
            'g1' : g1,
            'g2' : g2,
            'delta_1' : delta_1,
            'delta_2' : delta_2,
            'delta_m_1' : delta_m_1,
            'delta_m_2' : delta_m_2,
            'tp' : tp,
            'tm' : tm
        }

        self.operators_dict = {
            'sx' : sx,
            'sy' : sy,
            'sz' : sz,
            'ad_1' : ad_1,
            'a_1' : a_1,
            'ad_2' : ad_2,
            'a_2' : a_2
        }

        self.subspaces = [[Spin.subspace, dim_spin], [a_1.subspace, dim_bosonic], [a_2.subspace, dim_bosonic]]

        self.symbolic_hamiltonian = H0 + V1 + H2 + V2

        self.t = sp.Symbol('t', real=True, positive=True)

    def get_numeric_hamiltonian(self):
        if not hasattr(self, 'symbolic_hamiltonian'):
            self.get_symbolic_hamiltonian()
        self.numeric_hamiltonian = get_matrix(self.symbolic_hamiltonian.subs(self.subs_dict), self.subspaces)
        return self.numeric_hamiltonian
    
    def get_numeric_energies(self):
        if not hasattr(self, 'numeric_hamiltonian'):
            self.get_numeric_hamiltonian()
        energies = np.linalg.eigvalsh(np.array(self.numeric_hamiltonian, dtype=np.complex128))
        return np.sort(energies)
    
    def perturbation_expansion(self, order=2, full_diagonal=True):
        if not hasattr(self, 'symbolic_hamiltonian'):
            self.get_symbolic_hamiltonian()

        self.sol = solver(self.symbolic_hamiltonian, self.subspaces, order=order, full_diagonal=full_diagonal, subs_numerical=self.subs_dict)

        spin_0_state = get_state([[0, self.Spin.dim], [0, self.dim_bosonic], [0, self.dim_bosonic]])
        spin_1_state = get_state([[1, self.Spin.dim], [0, self.dim_bosonic], [0, self.dim_bosonic]])

        s0H_sols0 = expval(self.sol[0], spin_0_state)
        s1H_sols1 = expval(self.sol[0], spin_1_state)
        s0H_sols1 = expval(self.sol[0], spin_0_state, spin_1_state)
        s1H_sols0 = expval(self.sol[0], spin_1_state, spin_0_state)

        self.zeroth_energy = (s0H_sols0 + s1H_sols1) / 2
        self.qubit_frequency_perturbative = np.abs((s0H_sols0 - s1H_sols1).subs(self.subs_dict) / h_bar)

        self.H_qubit = sp.Matrix([[s0H_sols0, s0H_sols1], [s1H_sols0, s1H_sols1]]) # <00|H|00> where |00> is the ground state of the bosonic system

    def add_driving_expansion(self, lx_lE, omega_drive, order=2, co_moving_frame=False):

        if not hasattr(self, 'sol'):
            self.perturbation_expansion()
        
        self.lx_lE = lx_lE
        self.lE = self.lx / lx_lE

        bfreqs = np.abs([self.f1, self.f2, self.g1, self.g2])
        bfreqs = bfreqs[bfreqs != 0]
        ib = np.argmin(bfreqs)
        dfreqs = np.abs([self.delta_plus_1, self.delta_plus_2, self.delta_minus_1, self.delta_minus_2, self.tp, self.tm])
        dfreqs = dfreqs[dfreqs != 0]
        iz = np.argmin(dfreqs)

        lb = np.sqrt(h_bar / (m * bfreqs[ib])) * 1e9
        lz = np.sqrt(h_bar / (m * dfreqs[iz])) * 1e9


        E = h_bar **2 / (2 * m * e * (self.lE * NM)**3)

        self.E = E
        self.omega_drive = omega_drive

        self.subs_dict_driving = {}

        ad1 = self.operators_dict['ad_1']
        a1 = self.operators_dict['a_1']
        ad2 = self.operators_dict['ad_2']
        a2 = self.operators_dict['a_2']

        if not co_moving_frame:
            f1 = self.symbols_dict['f1']
            g2 = self.symbols_dict['g2']

            E0 = RDsymbol("E_0", order = 0)
            

            self.subs_dict_driving_amplitudes = {
                E0 : 2 * electron_charge * self.E / self.omega
            }

            H_drive = get_matrix(E0 * (f1 * (ad1 + a1) - sp.I * g2 * (ad2 - a2)), self.subspaces)
            H_drive_co_moving = sp.zeros(self.total_dim)

        else:
            dt = self.omega_c * electron_charge * E / (2 * m * self.Omega_x_2)
            Dt = electron_charge * E / (m * self.Omega_x_2) * omega_drive

            dt_ = RDsymbol("dt", order = 0)
            Dt_ = RDsymbol("Dt", order = 0)
            x0 = RDsymbol("x_0", order = 0)

            self.subs_dict_driving_amplitudes = {
                dt_ : dt,
                Dt_ : -Dt,
                x0 : h_bar/2 * self.omega * electron_charge * E / (m * self.Omega_x_2)
            }
            
            mu1 = RDsymbol("\\mu_1", order = 0)
            mu2 = RDsymbol("\\mu_2", order = 0)
            eta1 = RDsymbol("\\eta_1", order = 0)
            eta2 = RDsymbol("\\eta_2", order = 0)


            self.subs_aux_dict.update({
                mu1 : self.mu1,
                mu2 : self.mu2,
                eta1 : self.eta1,
                eta2 : self.eta2
            })

            self.subs_dict.update({
                mu1 : self.mu1,
                mu2 : self.mu2,
                eta1 : self.eta1,
                eta2 : self.eta2
            })

            H_drive = get_matrix(-dt_ * (mu1 * (ad1 + a1) + sp.I * eta2 * (ad2 - a2)) - Dt_ * (mu2 * (ad2 + a2) + sp.I * eta1 * (ad1 - a1)), self.subspaces)
            H_drive_co_moving = get_matrix(x0 * self.operators_dict['sx'], self.subspaces)

        self.H_drive_rotated = sp.Add(*rotate_with_S(H_drive, self.sol[1], order=order).values()) + H_drive_co_moving

        spin_0_state = get_state([[0, self.Spin.dim], [0, self.dim_bosonic], [0, self.dim_bosonic]])
        spin_1_state = get_state([[1, self.Spin.dim], [0, self.dim_bosonic], [0, self.dim_bosonic]])

        s0H_sols0 = expval(self.H_drive_rotated, spin_0_state)
        s1H_sols1 = expval(self.H_drive_rotated, spin_1_state)
        s0H_sols1 = expval(self.H_drive_rotated, spin_0_state, spin_1_state)
        s1H_sols0 = expval(self.H_drive_rotated, spin_1_state, spin_0_state)

        self.H_Drive_qubit = sp.Matrix([[s0H_sols0, s0H_sols1], [s1H_sols0, s1H_sols1]]) # <00|H_drive|00> where |00> is the ground state of the bosonic system

        self.Detuning_perturbative = np.float64(np.abs(self.qubit_frequency_perturbative - self.omega_drive))

        if not co_moving_frame:
            self.DriveStrength_perturvative = np.float64(np.abs(project_matrix2matrix(self.H_Drive_qubit, self.operators_dict['sx'].matrix).subs(self.subs_dict).subs(self.subs_dict_driving_amplitudes) / h_bar))

        else:
            self.DriveStrength_x = sp.re(sp.N(project_matrix2matrix(self.H_Drive_qubit, self.operators_dict['sx'].matrix).subs(self.subs_dict).subs(self.subs_dict_driving_amplitudes)) / h_bar)
            self.DriveStrength_y = sp.re(sp.N(project_matrix2matrix(self.H_Drive_qubit, self.operators_dict['sy'].matrix).subs(self.subs_dict).subs(self.subs_dict_driving_amplitudes)) / h_bar)

            self.DriveStrength_perturvative = np.float64(self.DriveStrength_x - self.DriveStrength_y)

        self.Rabi_frequency = np.sqrt(self.Detuning_perturbative**2 + self.DriveStrength_perturvative**2)

    def effective_qubit_hamiltonian(self, RWA=False):
        if RWA:
            return self.effective_RWA_qubit_hamiltonian()
        H0_qubit = np.array(sp.N(self.H_qubit.subs(self.subs_dict)), dtype=np.complex128)
        H_drive_t = np.array(sp.N(self.H_Drive_qubit.subs(self.subs_dict_driving_amplitudes).subs(self.subs_dict)).expand(), dtype=np.complex128)
        sx = np.array(self.operators_dict['sx'].matrix, dtype=np.complex128)
        sy = np.array(self.operators_dict['sy'].matrix, dtype=np.complex128)
        H_drive_x = project_matrix2matrix(H_drive_t, sx) * sx 
        H_drive_y = project_matrix2matrix(H_drive_t, sy) * sy 

        return H0_qubit, H_drive_x, H_drive_y
    
    def effective_RWA_qubit_hamiltonian(self):
        H0_RWA = np.array(- h_bar /2 * self.Detuning_perturbative * self.Spin._basis[3].matrix, dtype=np.complex128)
        H_x = np.array(h_bar / 2 * self.DriveStrength_perturvative * self.operators_dict['sx'].matrix, dtype=np.complex128)
        H_y = np.zeros(H_x.shape, dtype=np.complex128)
        return H0_RWA, H_x, H_y

    def full_hamiltonian(self):
        H = np.array(self.get_numeric_hamiltonian(), dtype=np.complex128)
        E0_aux = RDsymbol("E_0", order = 0)
        f1 = self.symbols_dict['f1']
        g2 = self.symbols_dict['g2']
        a1 = self.operators_dict['a_1']
        ad1 = self.operators_dict['ad_1']
        a2 = self.operators_dict['a_2']
        ad2 = self.operators_dict['ad_2']

        subs_drive = {E0_aux : -2*electron_charge*self.E/self.omega} # *sp.cos(self.omega_drive*self.t)

        H_drive = np.array(get_matrix((E0_aux * (f1 * (ad1 + a1) - sp.I * g2 * (ad2 - a2))).subs(self.subs_dict), self.subspaces).subs(subs_drive), dtype=np.complex128)

        Hy = np.zeros(H_drive.shape, dtype=np.complex128)
        return H, H_drive, Hy
    





    # ---------------------------------------------------------------------- Hand Made Hamiltonian ----------------------------------------------------------------------

    def qubit_frequency(self):
        freq_correction_1 = self.delta_plus_1 + 2 * self.Omega_z * (self.f1**2 + self.g1**2) / (self.omega_1**2 - self.Omega_z**2) - (4 * self.omega_1 * self.f1 * self.g1) / (self.omega_1**2 - self.Omega_z**2)
        freq_correction_2 = self.delta_plus_2 + 2 * self.Omega_z * (self.f2**2 + self.g2**2) / (self.omega_2**2 - self.Omega_z**2) + (4 * self.omega_2 * self.f2 * self.g2) / (self.omega_2**2 - self.Omega_z**2)
        freq_correction = freq_correction_1 + freq_correction_2
        return self.Omega_z -  freq_correction
    
    def add_driving(self, E, omega_drive, approximation=False, co_moving_frame=False):
        self.E = E
        self.omega_drive = omega_drive

        self.compute_driving_parameters(approximation, co_moving_frame=co_moving_frame)

    def add_driving_ratio(self, lx_lE, omega_drive, approximation=False, co_moving_frame=False):
        self.lx_lE = lx_lE
        self.lE = self.lx / lx_lE

        E = h_bar **2 / (2 * m * e * (self.lE * NM)**3)

        self.add_driving(E, omega_drive, approximation, co_moving_frame=co_moving_frame)

    def compute_optimal_ratio(self, frequency, R, first_order=False):
        self.CyMagnet.update_geometry(R, self.CyMagnet.L, self.CyMagnet.Delta_z)
        self.compute_parameters()

        coeff = 1/4 * g * mu_b /  h_bar * self.CyMagnet.b_1() * self.lx
        if not first_order:
            coeff += 1/4 * g * mu_b /  h_bar * self.CyMagnet.b_1() *  self.lx * (- self.Omega_z * self.omega_c/ self.omega_y**2 + self.Omega_z**2/self.omega_x**2 + self.Omega_z**2*self.omega_c**2 / (self.omega_x**2 * self.omega_y**2))
        Omega_R = 2 * np.pi * frequency
        return (Omega_R / coeff) ** (1/3)


    def compute_driving_parameters(self, approximation=False, co_moving_frame=False):
        if not co_moving_frame:
            if approximation:
                Bac = electron_charge * self.E /(2 * m) * self.CyMagnet.b_1() * 1/NM / self.omega_x**2 * (1 - self.Omega_z * self.omega_c/ self.omega_y**2 + self.Omega_z**2/self.omega_x**2 + self.Omega_z**2*self.omega_c**2 / (self.omega_x**2 * self.omega_y**2))
            else:
                Bac = electron_charge * self.E /(2 * m) * self.CyMagnet.b_1() * 1/NM / (self.omega_1**2 - self.Omega_z**2)
                Bac *=  (1 + (self.Omega_x**2 - self.Omega_y**2 - self.Omega_3**2)/(2*self.Omega_3**2)  + (self.omega_c + 2 * self.Omega_z)/(2*self.Omega_3**2) * self.omega_c) if self.minimal_coupling and self.B != 0 else 1
                Bac += electron_charge * self.E /(2 * m) * self.CyMagnet.b_1() * 1/NM / (self.omega_2**2 - self.Omega_z**2) * ((self.Omega_3**2 - self.Omega_x**2 + self.Omega_y**2)/(2*self.Omega_3**2) - (self.omega_c + 2*self.Omega_z)/(2*self.Omega_3**2)*self.omega_c ) if self.minimal_coupling and self.B != 0 else 0 
        
        else:
            Bac = - electron_charge * self.E /(2 * m) * self.CyMagnet.b_1() * 1/NM / self.Omega_x_2
            self.DriveStrength_x_ = Bac * (-1 + self.omega_c/2*self.fxy) * g * mu_b / h_bar
            self.DriveStrength_y_ = Bac * (self.omega_drive * self.fyx) * g * mu_b / h_bar
            Bac *= (-1 + self.omega_c/2*self.fxy - self.omega_drive*self.fyx)

        self.DriveStrength = g * mu_b / h_bar * Bac 
        self.Detuning = self.qubit_frequency() - self.omega_drive

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------
        

class Hamiltonian1D:
    def __init__(self, lw, Bz, CyMagnet:CylindricalMagnet):
        self.Bz = Bz
        self.CyMagnet = CyMagnet
        self.lw = lw

        self.omega_w = h_bar / (m * (lw * NM)**2)

        self.pos_coef = np.sqrt(h_bar / (2 * m * self.omega_w ))

        self.compute_parameters()

    def compute_parameters(self):

        self.B = self.Bz + self.CyMagnet.b_0()
        self.Omega_z =  g * mu_b / h_bar * (self.Bz + self.CyMagnet.b_0())
        self.omega_z = self.pos_coef**2 * g * mu_b / h_bar * self.CyMagnet.b_2() * 1/NM**2
        self.f = self.pos_coef * g * mu_b / (2 * h_bar) * self.CyMagnet.b_1() * 1/NM

    def qubit_frequency_correction(self,  activate_2order = True):
        freq_correction = self.omega_z
        if activate_2order:
            freq_correction += 2 * self.Omega_z * self.f**2 / (self.omega_w**2 - self.Omega_z**2)
        return freq_correction

    def qubit_frequency(self, activate_2order = True):
        return self.Omega_z - self.qubit_frequency_correction(activate_2order)
    
    def add_driving(self, E, omega_drive):
        self.E = E
        self.omega_drive = omega_drive

        self.compute_driving_parameters()

    def compute_driving_parameters(self):
        Bac = e * self.E / (2 * m) * self.CyMagnet.b_1() * 1/NM / (self.omega_w**2 - self.Omega_z**2)
        self.DriveStrength = g * mu_b / h_bar * Bac

    
class BasisOperators:
    def __init__(self, space :int=None, name='sigma', max_space=1):
        self.max_space = max_space
        self.sigma_x = Pauli(1, label=name)
        self.sigma_y = Pauli(2, label=name)
        self.sigma_z = Pauli(3, label=name)

        self.Z = Operator('Z') if space is None else Operator('Z_' + str(space))
        self.X = Operator('X') if space is None else Operator('X_' + str(space))
        self.Y = Operator('Y') if space is None else Operator('Y_' + str(space))
        self.I = Operator('I') if space is None else Operator('I_' + str(space))

        self.sigma_0 = sp.Rational(1, 2) * (1 + self.Z)
        self.sigma_1 = sp.Rational(1, 2) * (1 - self.Z)
        self.sigma_plus = sp.Rational(1, 2) * (self.X + sp.I * self.Y)
        self.sigma_minus = sp.Rational(1, 2) * (self.X - sp.I * self.Y)

        self.S0 = Operator('S_0') if space is None else Operator('S_0_' + str(space))
        self.S1 = Operator('S_1') if space is None else Operator('S_1_' + str(space))
        self.Sp = Operator('S_+') if space is None else Operator('S_+_' + str(space))
        self.Sm = Operator('S_-') if space is None else Operator('S_-_' + str(space))

        self.subs_rev_dict = {
            self.sigma_x: self.X,
            self.sigma_y: self.Y,
            self.sigma_z: self.Z,
            self.sigma_0: self.S0,
            self.sigma_1: self.S1,
            self.sigma_plus: self.Sp,
            self.sigma_minus: self.Sm
        }

        self.subs_dict = {v: k for k, v in self.subs_rev_dict.items()}
        self.subs_dict.update({
            self.I : 1})

        self.subs_matrix_dict = {
            self.X: sp.Matrix([[0, 1], [1, 0]]),
            self.Y: sp.Matrix([[0, -sp.I], [sp.I, 0]]),
            self.Z: sp.Matrix([[1, 0], [0, -1]]),
            self.I: sp.eye(2),
            self.S0 : sp.Matrix([[1, 0], [0, 0]]),
            self.S1 : sp.Matrix([[0, 0], [0, 1]]),
            self.Sp : sp.Matrix([[0, 1], [0, 0]]),
            self.Sm : sp.Matrix([[0, 0], [1, 0]])
        }

        for key in self.subs_matrix_dict.keys():
            matrix_result = sp.eye(2)  if space is not None else self.subs_matrix_dict[key]
            for i in range(2, max_space + 1):
                m = sp.eye(2) if space != i else self.subs_matrix_dict[key]
                matrix_result = sp.kronecker_product(matrix_result, m)
            self.subs_matrix_dict[key] = matrix_result

        self.Paulis = [self.I, self.X, self.Y, self.Z]
        self.SBasis = [self.S0, self.S1, self.Sp, self.Sm]

        self.subs_pauli_2_s_dict = {
            self.I: self.S0 + self.S1,
            self.X: self.Sp + self.Sm,
            self.Y: -sp.I * (self.Sp - self.Sm),
            self.Z: self.S0 - self.S1
        }
    
    def subs(self, expr):
        return expr.subs(self.subs_dict)
    
    def subs_pauli_2_s(self, expr):
        new_expr = expr.subs(self.subs_pauli_2_s_dict).expand()
        # If there are self.S0 * self.S1 terms, they should be replaced by 0

        simplification_dict = {
            self.S0 * self.S1: 0,
            self.S1 * self.S0: 0,
        }
        return new_expr.subs(simplification_dict)
    
    def subs_rev(self, expr):
        return expr.subs(self.subs_rev_dict)
    
    def subs_matrix(self, expr):
        return expr.subs(self.subs_matrix_dict).simplify()
    
    def get_matrix(self, pauli):
        return self.subs_matrix_dict[pauli]
    
    def project_to_operator(self, matrix, pauli):
        if pauli in self.Paulis:
            return sp.Rational(1, 2) * sp.trace(matrix @ self.get_matrix(pauli))
        return sp.trace(matrix @ self.get_matrix(pauli).T.conjugate())
    
    def project_to_basis(self, matrix, pauli_basis=True, s_basis=False, other_basis=[]):
        if pauli_basis:
            basis = self.Paulis
        elif s_basis:
            basis = self.SBasis
        else:
            basis = other_basis
            if len(basis) == 0:
                raise ValueError('No basis specified')
            
        return (sp.Matrix([[self.project_to_operator(matrix, b) for b in basis]]) @ sp.Matrix([basis]).T)[0]

    def eval_pauli_product(self, expr):
        return self.subs_rev(evaluate_pauli_product(self.subs(expr).expand()))

class Rabi_Model:
    def __init__(self):
        self.BO = BasisOperators()
        alpha_k = lambda k: sp.symbols(f'alpha_{k}', real=True)
        hbar = sp.symbols('hbar', real=True, positive=True)
        self.alpha_k = alpha_k

        self.B_t = sp.Matrix([
            [0, -alpha_k('z'), alpha_k('y')],
            [alpha_k('z'), 0, -alpha_k('x')],
            [-alpha_k('y'), alpha_k('x'), 0]
        ])

        self.H = sp.expand(sp.Rational(1,2) * hbar * (sp.Matrix([alpha_k('x'), alpha_k('y'), alpha_k('z')]).T @ sp.Matrix(self.BO.Paulis[1:]))[0])
    
    def __str__(self) -> str:
        return sp.latex(self.H)
    
    def __repr__(self) -> str:
        return sp.latex(self.H)

    def define_problem(self, alphas:list):
        alphas = {self.alpha_k(k): v for k,v in zip(['x','y','z'], alphas)}
        self.B = self.B_t.subs(alphas)
        self.alphas = alphas
        self.H = self.H.subs(alphas)
    
    def get_expBt(self):
        t = sp.symbols('t', real=True)
        if not hasattr(self, 'expB'):
            self.expB = sp.simplify(sp.exp(self.B * t))
        return self.expB
    
    def get_sigma_t(self, initial_state):
        expB = self.get_expBt()
        return expB @ initial_state
    
    def get_P0(self, initial_state):
        expval_z = self.get_sigma_t(initial_state)[2]
        return sp.simplify(sp.Rational(1,2) * (1 + expval_z))
    
    def get_P1(self, initial_state):
        return sp.simplify(1 - self.get_P0(initial_state).expand())

