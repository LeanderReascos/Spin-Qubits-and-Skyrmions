import sympy as sp
from sympt import Operator, BosonOp, Dagger, hbar

import numpy as np

from Packages.utils import apply_subs_to_dict

# Defining Constants
hbar_value = 1.054571817e-34  # Planck constant over 2π, in J·s
e_constant = 1.602176634e-19  # Elementary charge magnitude, in C
e_charge = -e_constant        # Electron charge, in C
m_electron = 9.10938356e-31   # Mass of a free electron, in kg
m_si = 0.19 * m_electron  # Effective mass of electron in silicon conduction band


class EllipticalConfinement:
    def __init__(self):

        m = sp.Symbol('m', positive=True, real=True) # effective mass

        # ----------- Definition of frequencies in the x, y, and c directions and the transformed basis -----------

        self.omega_x = sp.symbols('omega_x', real=True, positive=True)
        self.lx = sp.Symbol('l_x', real=True, positive=True)

        self.frequency_to_length = {
            self.omega_x : hbar / (m * self.lx**2),
        }

        self.lenght_to_frequency = {
            self.lx : sp.sqrt(hbar / (m * self.omega_x)),
        }
        
        self.ky = sp.Symbol('k_y', real=True, positive=True)
        self.kc = sp.Symbol('k_c', real=True)
        self.omega_y = self.ky * self.omega_x
        self.omega_c = self.kc * self.omega_x

        self.kp, self.km = sp.symbols('k_+ k_-', real=True, positive=True)
        self.omega_p = self.kp * self.omega_x
        self.omega_m = self.km * self.omega_x

        omega_3 = sp.sqrt(sp.sqrt((self.omega_x**2 - self.omega_y**2)**2 + 2*self.omega_c**2 * (self.omega_x**2 + self.omega_y**2) + self.omega_c**4))
        
        self.eccentricity = sp.Symbol('epsilon', real=True, positive=True)

        self.ky_to_eccentricity = {
            self.ky : sp.sqrt(1 - self.eccentricity**2),
        }

        self.eccentricity_to_ky = {
            self.eccentricity : sp.sqrt(1 - self.ky**2),
        }

        self.omega_pm_to_xyc_subs = {
            self.kp : (sp.sqrt(sp.Rational(1, 2) * (self.omega_x**2 + self.omega_y**2 + self.omega_c**2 + omega_3**2)).collect(self.omega_x) / self.omega_x).subs(self.ky_to_eccentricity),
            self.km : (sp.sqrt(sp.Rational(1, 2) * (self.omega_x**2 + self.omega_y**2 + self.omega_c**2 - omega_3**2)).collect(self.omega_x) / self.omega_x).subs(self.ky_to_eccentricity),
        }


        self.useful_rules = {
            self.kp * self.km : self.ky
        }

        # ----------- Hamiltonian -----------

        self.a_p = BosonOp('a_+')
        self.a_m = BosonOp('a_-')
        self.ad_p = Dagger(self.a_p)
        self.ad_m = Dagger(self.a_m)

        # ---------- Normal Ordering -------------

        self.normal_ordering = {
            self.a_m * self.a_p : self.a_p * self.a_m,
            self.ad_m * self.ad_p : self.ad_p * self.ad_m,
            self.a_m * self.ad_p : self.ad_p * self.a_m,
            self.ad_m * self.a_p : self.a_p * self.ad_m,
            self.a_m * self.ad_m : self.ad_m * self.a_m + 1,
            self.a_p * self.ad_p : self.ad_p * self.a_p + 1,
        }

        # Hamiltonian
        self.H = hbar * self.omega_x * (self.kp * (self.ad_p * self.a_p + sp.Rational(1,2)) + self.km * (self.ad_m * self.a_m + sp.Rational(1,2)))


        # ----------- Definition of the transformation -----------

        Omega_x = sp.sqrt(self.omega_x**2 + sp.Rational(1, 4) * self.omega_c**2).collect(self.omega_x)
        Omega_y = sp.sqrt(self.omega_y**2 + sp.Rational(1, 4) * self.omega_c**2).collect(self.omega_x)
        k3 = sp.Symbol('k_3', real=True, positive=True)
        Omega_3 = k3 * self.omega_x

        subs_rules = {
            2*(1 + sp.Rational(1,2) *(self.kc**2 - self.eccentricity**2 + k3**2)) + 2 - 2*self.eccentricity**2: 2*self.kp**2 + 2*self.kp**2*self.km**2,
            2*(1 + sp.Rational(1,2) *(self.kc**2 - self.eccentricity**2 - k3**2)) + 2: 2*self.km**2 + 2,
            -2*(1 + sp.Rational(1,2) *(self.kc**2 - self.eccentricity**2 + k3**2)) - 2: -2*self.kp**2 - 2,
            -2*(1 + sp.Rational(1,2) *(self.kc**2 - self.eccentricity**2 - k3**2)) - 2 + 2*self.eccentricity**2: -2*self.km**2 - 2*self.kp**2*self.km**2,
            2*(sp.Rational(1,2) * ((self.kp**2 + self.km**2).subs(self.omega_pm_to_xyc_subs)).cancel() + 1 - sp.Rational(1,2) * self.eccentricity**2) :(self.kp**2 + self.km**2) + 1 + self.kp**2*self.km**2,
        }


        c1 = sp.sqrt((Omega_x**2 + 3*Omega_y**2 + Omega_3**2) / (2 * Omega_x**2 + 2 * Omega_y**2)).collect(self.omega_x).subs(self.ky_to_eccentricity).subs(subs_rules).factor()
        c2 = sp.sqrt((3*Omega_x**2 + Omega_y**2 - Omega_3**2) / (2 * Omega_x**2 + 2 * Omega_y**2)).collect(self.omega_x).subs(self.ky_to_eccentricity).subs(subs_rules).factor()
        Omega_1 = sp.sqrt(sp.Rational(1,4) * (3*Omega_x**2 + Omega_y**2 + Omega_3**2)).collect(self.omega_x).factor().subs(self.ky_to_eccentricity).subs(subs_rules).factor()
        Omega_2 = sp.sqrt(sp.Rational(1,4) * (Omega_x**2 + 3*Omega_y**2 - Omega_3**2)).collect(self.omega_x).factor().subs(self.ky_to_eccentricity).subs(subs_rules).factor()

        theta = sp.Symbol('theta', real=True)

        beta = (- m * sp.sqrt((Omega_x**2 + Omega_y**2) / 2)).factor().subs(self.ky_to_eccentricity).subs(subs_rules).factor()
        alpha = 1/beta

        self.x = Operator('x')
        self.y = Operator('y')
        self.px = Operator('p_x')
        self.py = Operator('p_y')

        self.qp = Operator('q_1')
        self.qm = Operator('q_2')
        self.pp = Operator('p_1')
        self.pm = Operator('p_2')


        self.transformation = {
            self.x : (- alpha * sp.sin(theta) * self.pm + sp.cos(theta) * self.qp).subs(self.frequency_to_length),
            self.y : (- alpha * sp.sin(theta) * self.pp + sp.cos(theta) * self.qm).subs(self.frequency_to_length),
            self.px :( beta * sp.sin(theta) * self.qm + sp.cos(theta) * self.pp).subs(self.frequency_to_length),
            self.py :( beta * sp.sin(theta) * self.qp + sp.cos(theta) * self.pm).subs(self.frequency_to_length),
        }


        self.boson_transformation = {
            self.qp : sp.sqrt(hbar / (2*m) * c1/ Omega_1).subs(self.frequency_to_length).simplify() * (self.a_p + self.ad_p),
            self.pp : -sp.I * sp.sqrt(hbar * m / 2 * Omega_1/c1).subs(self.frequency_to_length).simplify() * (self.a_p - self.ad_p),
            self.qm : sp.sqrt(hbar / (2*m) * c2/ Omega_2).subs(self.frequency_to_length).simplify() * (self.a_m + self.ad_m),
            self.pm : -sp.I * sp.sqrt(hbar * m / 2 * Omega_2/c2).subs(self.frequency_to_length).simplify() * (self.a_m - self.ad_m),
        }

        self.position_momentum_to_new_basis = {k: v.subs(self.boson_transformation) for k,v in self.transformation.items()}
        
        self.Kp = sp.Symbol('K_+', real=True, positive=True)
        self.Km = sp.Symbol('K_-', real=True, positive=True)

        self.inv_effective_lengths_subs = {
            self.Kp : sp.sqrt(self.kp / (self.kp**2 + 1)),
            self.Km : sp.sqrt(self.km / (self.km**2 + 1)),
            1 / self.Kp : sp.sqrt((self.kp**2 + 1) / self.kp),
            1 / self.Km : sp.sqrt((self.km**2 + 1) / self.km),
        }

        self.inv_effective_lengths_subs.update({
            self.Kp / sp.sqrt(self.ky) : (self.Kp / sp.sqrt(self.kp * self.km)).subs(self.inv_effective_lengths_subs),
            self.Km / sp.sqrt(self.ky) : (self.Km / sp.sqrt(self.kp * self.km)).subs(self.inv_effective_lengths_subs),
            sp.sqrt(self.ky) / self.Kp : (sp.sqrt(self.kp * self.km) / self.Kp).subs(self.inv_effective_lengths_subs),
            sp.sqrt(self.ky) / self.Km : (sp.sqrt(self.kp * self.km) / self.Km).subs(self.inv_effective_lengths_subs),
        })

        xp, xm, yp, ym = sp.symbols('x_+ x_- y_+ y_-', real=True)
        pxp, pxm, pyp, pym = sp.symbols('p_x_+ p_x_- p_y_+ p_y_-', real=True)

        self.inv_pos_mom_symbols = {
            xp : self.Kp* sp.cos(theta),
            xm : self.Km* sp.sin(theta),
            yp : self.Kp* sp.cos(theta) / sp.sqrt(self.ky),
            ym : self.Km* sp.sin(theta) / sp.sqrt(self.ky),
            pxp : sp.cos(theta) / (self.Kp),
            pxm : sp.sin(theta) / (self.Km),
            pyp : sp.cos(theta) * sp.sqrt(self.ky)/ (self.Kp),
            pym : sp.sin(theta) * sp.sqrt(self.ky)/ (self.Km),
        }

        self.xp = xp 
        self.xm = xm 
        self.yp = yp 
        self.ym = ym 
        self.pxp= pxp
        self.pxm= pxm
        self.pyp= pyp
        self.pym= pym


        self.pos_mom_symbols = {v:k for k,v in self.inv_pos_mom_symbols.items()}

        self.effective_lengths_subs = {v:k for k,v in self.inv_effective_lengths_subs.items()}

        self.pos_mom_to_new_basis_effective_lengths = {k: v.subs(self.effective_lengths_subs) for k,v in self.position_momentum_to_new_basis.items()}

        self.pos_mom_to_new_basis_symbols = {k: v.subs(self.pos_mom_symbols) for k,v in self.pos_mom_to_new_basis_effective_lengths.items()}

        A = (self.omega_c * sp.sqrt(2*(Omega_x**2 + Omega_y**2))).factor().subs(subs_rules).factor()
        B = (self.omega_x**2 - self.omega_y**2).subs(subs_rules).factor()

        subs_theta = {
            theta : sp.atan2(sp.Symbol('A'), sp.Symbol('B')) / 2
        }

        substituition_A2_B2 = {
            sp.Symbol('A')**2 + sp.Symbol('B')**2 : (A**2 + B**2).expand().subs(self.kp**2*self.km**2, 1 - self.eccentricity**2)
        }
        substituition = {
            sp.Symbol('A') : A.subs(self.ky_to_eccentricity),
            sp.Symbol('B') : B.subs(self.ky_to_eccentricity).expand(),
        }

        self.useful_rules.update({
            (self.kp**2 - self.km**2).subs(self.omega_pm_to_xyc_subs).expand() : self.kp**2 - self.km**2,
            ((self.kp**2 + 1)*(self.km**2 + 1)).subs(self.omega_pm_to_xyc_subs).simplify() : (self.kp**2 + 1)*(self.km**2 + 1),
            (self.kp**2 + 1)*(self.km**2 + 1) : ((self.kp**2 + 1)*(self.km**2 + 1)).subs(self.omega_pm_to_xyc_subs).simplify()
        })


        self.trig_subs = {
            sp.sin(theta)**2 : (sp.sin(theta.subs(subs_theta))**2).trigsimp().subs(substituition_A2_B2).simplify().subs(self.ky_to_eccentricity).simplify().subs(self.useful_rules).subs(substituition),
            sp.cos(theta)**2 : (sp.cos(theta.subs(subs_theta))**2).trigsimp().subs(substituition_A2_B2).simplify().subs(self.ky_to_eccentricity).simplify().subs(self.useful_rules).subs(substituition),
            sp.sin(theta)*sp.cos(theta) : (sp.sin(theta.subs(subs_theta))*sp.cos(theta.subs(subs_theta))).trigsimp().subs(substituition_A2_B2).simplify().subs(self.ky_to_eccentricity).simplify().subs(self.useful_rules).subs(substituition).subs(self.useful_rules),
        }

        self.subs_theta = {
            theta : sp.atan2(A, B) / 2
        }


        # ---- Circular Confinement ------

        self.x0 = sp.symbols('x_0', real=True, positive=True)

        circular_confinement_aux = {
            self.kp**2 + 1 : sp.sqrt(self.kc**2 + 4) * self.kp,
            self.km**2 + 1 : sp.sqrt(self.kc**2 + 4) * self.km,
            sp.sqrt(self.km) : 1 / sp.sqrt(self.kp),
            theta : sp.pi / 4,
            (self.kc**2 + 4)**(sp.Rational(1,4)) : 1/(sp.sqrt(2)*self.x0),
        }

        self.circular_confinement = apply_subs_to_dict(self.inv_pos_mom_symbols, self.inv_effective_lengths_subs, circular_confinement_aux)
        self.pos_to_new_basis_circular_confinement = apply_subs_to_dict(self.pos_mom_to_new_basis_symbols, self.circular_confinement)

        Omega_0_symbol = sp.Symbol('Omega_0', real=True, positive=True)  # Rabi frequency prefactor
        omega_c_symbol = sp.Symbol('omega_c', real=True)  # Cyclotron frequency

        omega_0_symbol = sp.Symbol('omega_0', real=True)  # Rabi frequency

        subs_test = {
            Omega_0_symbol : self.omega_x / (4 * self.x0**2),
            sp.Abs(omega_c_symbol) : (self.omega_x * sp.sqrt(1/(4 * self.x0**4) - 4)).simplify()
        }

        self.subs_kp_km_circular = {
            self.kp : ((Omega_0_symbol/self.omega_x + sp.Abs(omega_c_symbol) / (2 * self.omega_x)).subs(subs_test)).expand(),
            self.km : ((Omega_0_symbol/self.omega_x - sp.Abs(omega_c_symbol) / (2 * self.omega_x)).subs(subs_test)).expand()
        }


    def add_values(self, lx=60, eccentricity=1, B=100):
        omega_x = hbar_value / (m_si * (lx * 1e-9)**2)
        omega_c = e_charge * B / m_si
        
        k_c = omega_c / omega_x
        k_y = np.sqrt(1 - eccentricity**2)

        k_p = np.sqrt(1 + 1/2*(k_c**2 - eccentricity**2 + np.sqrt((eccentricity**2 - k_c**2)**2 + 4*k_c**2)))
        k_m = np.sqrt(1 + 1/2*(k_c**2 - eccentricity**2 - np.sqrt((eccentricity**2 - k_c**2)**2 + 4*k_c**2)))

        theta = 1/2 * np.atan2(k_c * np.sqrt((k_p**2 + 1) * (k_m**2 + 1)), eccentricity**2)

        Kp = np.sqrt(k_p / (k_p**2 + 1))
        Km = np.sqrt(k_m / (k_m**2 + 1))

        xp  = Kp * np.cos(theta)
        xm  = Km * np.sin(theta)
        yp  = Kp * np.cos(theta) / np.sqrt(k_y)
        ym  = Km * np.sin(theta) / np.sqrt(k_y)
        pxp = np.cos(theta) / (Kp)
        pxm = np.sin(theta) / (Km)
        pyp = np.cos(theta) * np.sqrt(k_y)/ (Kp)
        pym = np.sin(theta) * np.sqrt(k_y)/ (Km)

        self.subs_numerical_values = {
            self.omega_x : omega_x,
            self.omega_y : k_y * omega_x,
            self.ky : k_y,
            self.eccentricity : eccentricity,
            self.kp : k_p,
            self.km : k_m,
            self.kc : k_c,

            self.xp  : xp ,
            self.xm  : xm ,
            self.yp  : yp ,
            self.ym  : ym ,
            self.pxp : pxp,
            self.pxm : pxm,
            self.pyp : pyp,
            self.pym : pym,
        }
