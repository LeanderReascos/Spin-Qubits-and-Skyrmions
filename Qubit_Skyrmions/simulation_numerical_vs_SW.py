import sympy as sp
import numpy as np

from SymPT import * 

from qutip import basis, destroy, mesolve, qeye, tensor, Qobj

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

Nm = 1e-9


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


class QubitHamiltonianSkyrmion:
    def __init__(self, lx, ly, Bz, CyMagnet:CylindricalMagnet, N_boson_states=10):
        # Numerical Parameters
        self.lx = lx
        self.ly = ly
        self.Bz = Bz
        self.CyMagnet = CyMagnet
        self.N_boson_states = N_boson_states

        # Numerical Frequencies
        self.omega_x = h_bar / (m * (lx * Nm)**2)
        self.omega_y = h_bar / (m * (ly * Nm)**2)
        self.Bz = Bz
    
    def compute_numerical_simulation_values(self):

        self.B = self.Bz + self.CyMagnet.b_0()
        self.omega_c = electron_charge * self.B / (2 * m)
        self.omega_z = g * mu_b / h_bar * self.B

        # Numerical Frequencies in terms of omega_x
        self.ky = self.omega_x / self.omega_y
        self.kB = np.abs(self.omega_x / self.omega_c)
        self.kz = self.omega_x / self.omega_z

        k3_inv_2 = np.sqrt((1 - 1/self.ky**2)**2 + 2/self.kB**2 * (1 + 1/self.ky**2) + 1/self.kB**4)
        self.kp = 1 / (np.sqrt(1/2 * (1 + 1/self.ky**2 + 1/self.kB**2 + k3_inv_2)))
        self.km = 1 / (np.sqrt(1/2 * (1 + 1/self.ky**2 + 1/self.kB**2 - k3_inv_2)))

        self.n1 = self.CyMagnet.b_1() * self.lx
        self.n2 = self.CyMagnet.b_2() * self.lx**2

        Kp_val = 2 * self.kp / (self.kp**2 + 1)
        Km_val = 2 * self.km / (self.km**2 + 1)

        theta_val = 1/2 * np.atan(-(2 / self.kB * np.sqrt(1 / (Kp_val*Km_val*self.ky)))/((self.ky**2 - 1) / self.ky**2))
        self.numerical_values = np.array([self.ky, self.kB, self.kz, self.B, self.n1, self.n2, Kp_val, Km_val, self.kp, self.km, theta_val], dtype=np.float64)

        # Symbols to numerical substitutions
        if not hasattr(self, 'substitutions'):
            self.prepare_symbolic_hamiltonian()

        self.substitutions = dict(zip(self.symbols, self.numerical_values))

    def prepare_symbolic_hamiltonian(self):
        
        # ------------- Defining the symbols for the Hamiltonian ----------------
        hbar = sp.symbols('hbar', positive=True, real=True)
        self.hbar = hbar
        omega_x = sp.symbols('omega_x', positive=True, real=True) # Frequency of x confinement
        self.omega_x_sym = omega_x
        lx = sp.Symbol('l_x', positive=True, real=True)         # Length of x confinement
        self.lx_sym = lx
        m = sp.Symbol('m', positive=True, real=True)                    # Effective mass of the particle on Si
        self.m = m  

        ky = RDSymbol('k_y', positive=True, real=True)  
        kB = RDSymbol('k_B', positive=True, real=True)  
        # Total Magnetic field perpendicular to the plane of xy confinement B = B0 + b0, where b0 is genereted by the skyrmion
        B = sp.symbols('B', real=True)


        # Effective frequencies affter diagonalization of H0 using the minimal coupling substitution
        kp = sp.symbols('k_+', positive=True, real=True)
        km = sp.symbols('k_-', positive=True, real=True)
        # Omega_3_2 = sp.sqrt((omega_x**2 - omega_y**2)**2 + 2 * omega_c**2 * (omega_x**2 + omega_y**2) + omega_c**4)
        omega_p = omega_x / kp              #1/sp.sqrt(2) * sp.sqrt(omega_x**2 + omega_y**2 + omega_c**2 + Omega_3_2)
        omega_m = omega_x / km              #1/sp.sqrt(2) * sp.sqrt(omega_x**2 + omega_y**2 + omega_c**2 - Omega_3_2)

        # Auxilliary variables
        Kp = sp.symbols('K_+', positive=True, real=True)
        Km = sp.symbols('K_-', positive=True, real=True)

        # ------------- Defining the position momentum basis for the Hamiltonian ----------------
        # Defining the bosonic operators of the new basis after diagonalization of H0
        # a_+ and a_- are the bosonic operators of the new basis

        a_1 = BosonOp('a_+')
        ad_1 = Dagger(a_1)
        a_2 = BosonOp('a_-')
        ad_2 = Dagger(a_2)

        self.operators_qutip = {
            a_1: tensor(destroy(self.N_boson_states), qeye(self.N_boson_states), qeye(2)),
            a_2: tensor(qeye(self.N_boson_states), destroy(self.N_boson_states), qeye(2)),
        }
        self.operators_qutip.update({
            ad_1: self.operators_qutip[a_1].dag(),
            ad_2: self.operators_qutip[a_2].dag()
        })

        # Defining the position operators x and y in the new basis after diagonalization of H0 and using the bosonic operators of the new basis
        theta = sp.Symbol('theta', real=True)               # Parameter used on the diagonalization of H0
        beta = - hbar / lx**2 * sp.sqrt(1 / (Kp*Km*ky))     # Parameter used on the diagonalization of H0
        alpha = 1 / beta

        c1_Omega1 = Kp / omega_x
        c2_Omega2 = Kp * ky / omega_x

        # Substitutions for the lenghts of the confinement
        self.subs_lenght = {
            sp.sqrt(hbar / (m * omega_x)): lx,
            sp.sqrt( m * omega_x / hbar): 1 / lx,
            sp.sqrt( m * omega_x * hbar): hbar / lx,
        }

        # Position Operators
        self.x = Operator('x')
        self.y = Operator('y')
        self.px = Operator('p_x')
        self.py = Operator('p_y')

        # Position operators in the new basis
        x_T = (sp.sqrt(hbar/(2*m) * c1_Omega1) * sp.cos(theta) * (ad_1 + a_1) - sp.I * sp.sqrt(hbar*m * 1 / (2 * c2_Omega2)) * alpha * sp.sin(theta) * (ad_2 - a_2)).subs(self.subs_lenght)
        y_T = (sp.sqrt(hbar/(2*m) * c2_Omega2) * sp.cos(theta) * (ad_2 + a_2) - sp.I * sp.sqrt(hbar*m * 1 / (2 * c1_Omega1)) * alpha * sp.sin(theta) * (ad_1 - a_1)).subs(self.subs_lenght)
        px_T = (sp.sqrt(hbar/(2*m)* c2_Omega2) * beta * sp.sin(theta) * (ad_2 + a_2) + sp.I * sp.sqrt(hbar * m * 1 / (2 * c1_Omega1)) * sp.cos(theta) * (ad_1 - a_1)).subs(self.subs_lenght)
        py_T = (sp.sqrt(hbar/(2*m)* c1_Omega1) * beta * sp.sin(theta) * (ad_1 + a_1) + sp.I * sp.sqrt(hbar * m * 1 / (2 * c2_Omega2)) * sp.cos(theta) * (ad_2 - a_2)).subs(self.subs_lenght)

        self.transformation = {
            self.x : x_T,
            self.y : y_T,
            self.px : px_T,
            self.py : py_T
        }

        # ------------- Defining the spin basis for the Hamiltonian ----------------

        self.Spin = RDBasis('sigma', 2)
        s0, sx, sy, sz = self.Spin.basis # Pauli operators
        self.operators_qutip.update({
            s0: tensor(qeye(self.N_boson_states), qeye(self.N_boson_states), qeye(2)),
            sx: tensor(qeye(self.N_boson_states), qeye(self.N_boson_states), Qobj(sx.matrix)),
            sy: tensor(qeye(self.N_boson_states), qeye(self.N_boson_states), Qobj(sy.matrix)),
            sz: tensor(qeye(self.N_boson_states), qeye(self.N_boson_states), Qobj(sz.matrix))
        })

        kz = RDSymbol('k_z', positive=True, real=True)
        Omega_z = omega_x / kz

        n1 = RDSymbol('n_1', order=1, positive=True, integer=True)          # b1=n1/lx
        n2 = RDSymbol('n_2', order=2, positive=True, integer=True)          # b2=n2/lx**2

        # ---------  Symbols -----------
        self.symbols = [ky, kB, kz, B, n1, n2, Kp, Km, kp, km, theta]

        # ---------  Hamiltonian -----------

        self.H0 = ((sp.factor_terms((hbar * omega_p * ad_1 * a_1 + hbar * omega_m * ad_2 * a_2).expand()).expand() - sp.Rational(1,2) * hbar * Omega_z * sz) / (hbar * omega_x)).expand()
        self.V1 = (sp.factor_terms(sp.Rational(1,2) * hbar * Omega_z * n1 / lx * 1/B * (x_T * sx + y_T * sy)) / (hbar * omega_x)).expand()
        self.V2 = (sp.factor_terms(sp.Rational(1,2) * hbar * Omega_z * n2 / lx**2 * 1/B * (x_T**2  + y_T**2) * sz) / (hbar * omega_x)).expand()


        kE = sp.symbols('k_E', positive=True, real=True)
        self.E = - 1 / (2 * lx) * 1/ kE**(3/2) 

        self.drive_symbols = [kE]

        self.H = self.H0 + self.V1 + self.V2

    def add_drive(self, lx_lE):
        kE = 1/(lx_lE ** 2)
        self.drive_subs = dict(zip(self.drive_symbols, [kE]))

    def eff_lab_frame(self, detuning = 0, order=2):
                   
        Eff_frame = EffectiveFrame(self.H, subspaces=[self.Spin])
        Eff_frame.solve(max_order=order, full_diagonalization=True)

        H_dict = Eff_frame.get_H('dict_matrix')
        H_qubit = H_dict[1].subs(self.substitutions) # 0 modes in the bosonic subspace

        H0 = Qobj(H_qubit)
        energies, states = H0.eigenstates()

        qubit_freq = np.abs(energies[1] - energies[0])
        projector_0 = states[0] * states[0].dag()
        projector_1 = states[1] * states[1].dag()

        H_drive = self.E * self.x.subs(self.transformation) # cos(omega * t)
        HDrive = Eff_frame.rotate(H_drive, return_form='dict_matrix')
        HD_qubit = HDrive[1].subs(self.substitutions).subs(self.drive_subs) # 0 modes in the bosonic subspace

        E0 = HD_qubit[0,1]

        drive_freq = qubit_freq + detuning
        Rabi_freq = sp.sqrt(E0**2 + detuning**2)

        self.lab_frame_qubit = {
            'qubit_freq': qubit_freq,
            'drive_freq': drive_freq,
            'Rabi_freq': Rabi_freq,
            'H0' : H0,
            'HDrive': [Qobj(HD_qubit), lambda t, args=None: np.cos(drive_freq * t)],
            'psi0' : basis(2, 0),
            'Projectors' : [projector_0, projector_1]
        }

    def eff_co_moving_frame(self, detuning=0, order=2):
        Eff_frame = EffectiveFrame(self.H, subspaces=[self.Spin])
        Eff_frame.solve(max_order=order, full_diagonalization=True)

        H_dict = Eff_frame.get_H('dict_matrix')
        H_qubit = H_dict[1].subs(self.substitutions)

        H0 = Qobj(H_qubit)
        energies, states = H0.eigenstates()

        qubit_freq = np.abs(energies[1] - energies[0])
        projector_0 = states[0] * states[0].dag()
        projector_1 = states[1] * states[1].dag()

        ky, kB, kz, B, n1, n2, Kp, Km, kp, km, theta = self.symbols
        lx = self.lx_sym
        _, sx, sy, sz = self.Spin.basis

        omega_y = self.omega_x_sym / ky
        omega_c = -self.omega_x_sym / kB
        Omega_z = self.omega_x_sym / kz

        H0_expr = self.px**2/(2*self.m) + self.py**2/(2*self.m) + self.m/2 * (self.omega_x_sym**2 + sp.Rational(1,4) * omega_c**2) * self.x**2 + self.m/2 * (omega_y**2 + sp.Rational(1,4) * omega_c**2) * self.y**2  + sp.Rational(1,2) * omega_c * (self.px * self.y - self.py * self.x)
        H0_sym = Operator('H_0')

        d_sym = sp.Symbol('d', real=True)
        ddt_sym = sp.Symbol('d_t', real=True)

        # Rotating H0
        H_transformed_dict = group_by_operators(H0_expr.subs(self.x, self.x - d_sym).expand().subs(H0_expr.expand(), H0_sym) - ddt_sym * self.px)
        d_val = sp.nsimplify(sp.solve((H_transformed_dict[self.x] / (self.hbar * self.omega_x_sym) + self.E ), d_sym)[0]).subs(self.subs_lenght)
        display(d_val)

        # Rotating the perturbation to the co-moving frame
        V1_sym = Operator('V_1')
        V2_sym = Operator('V_2')

        V1_expr = sp.factor_terms(sp.Rational(1,2) * self.hbar * Omega_z * n1 / lx * 1/B * (self.x * sx + self.y * sy)) 
        V1_transformed_dict = group_by_operators(V1_expr.subs(self.x, self.x - d_sym).expand().subs(V1_expr.expand(), V1_sym))
        V2_expr = sp.factor_terms(sp.Rational(1,2) * self.hbar * Omega_z * n2 / lx**2 * 1/B * (self.x**2 + self.y**2) * sz)
        V2_transformed_dict = group_by_operators(V2_expr.subs(self.x, self.x - d_sym).expand().subs(V2_expr.expand(), V2_sym))

        H_eff_drive_x = V1_transformed_dict[sx]*sx.matrix
        H_drive_transformed = (H_transformed_dict[self.px] * self.px + H_transformed_dict[self.py] * self.py).subs(self.transformation) / (self.hbar * self.omega_x_sym)

        Eff_co_moving_drive = Eff_frame.rotate(H_drive_transformed, return_form='dict_matrix')
        H_eff_drive_dict = group_by_operators(self.Spin.project(Eff_co_moving_drive[1]))

        H_eff_drive_x += H_eff_drive_dict[sx].subs(d_sym, d_val).subs(self.substitutions) * sx.matrix
        H_eff_drive_y = H_eff_drive_dict[sy].subs(ddt_sym, d_val).subs(self.substitutions) * sy.matrix

        display(H_eff_drive_x)
        display(H_eff_drive_y)

        drive_freq = qubit_freq + detuning

        HD_qubit = [[Qobj(H_eff_drive_x), lambda t, args=None: np.cos(drive_freq * t)], [Qobj(H_eff_drive_y), lambda t, args=None: -drive_freq * np.sin(drive_freq * t)]]

        self.co_moving_frame_qubit = {
            'qubit_freq': qubit_freq,
            'drive_freq': drive_freq,
            'H0' : H0,
            'HDrive': HD_qubit,
            'psi0' : basis(2, 0),
            'Projectors' : [projector_0, projector_1]
        }
    
    def numerical_hamiltonian(self, detuning=0):
        H_numerical = self.H.subs(self.substitutions)
        H_numerical_dict = group_by_operators(H_numerical)

        H0_numerical_qutip = 0
        for k, v in H_numerical_dict.items():
            op = 1
            for o in k.as_ordered_factors():
                exponent = 1
                if isinstance(o, sp.Pow):
                    exponent = o.as_base_exp()[1]
                    o = o.base
                qutip_op = self.operators_qutip[o] ** exponent
                op *= qutip_op
            H0_numerical_qutip += v * op
        
        H0 = Qobj(H0_numerical_qutip)
        energies, states = H0.eigenstates()

        qubit_freq = np.abs(energies[1] - energies[0])
        projector_0 = states[0] * states[0].dag()
        projector_1 = states[1] * states[1].dag()

        H_drive = self.E * self.x.subs(self.transformation) # cos(omega * t)
        HDrive_dict = group_by_operators(H_drive.subs(self.substitutions).subs(self.drive_subs))
        HDrive_numerical_qutip = 0
        for k, v in HDrive_dict.items():
            op = 1
            for o in k.as_ordered_factors():
                exponent = 1
                if isinstance(o, sp.Pow):
                    exponent = o.as_base_exp()[1]
                    o = o.base
                qutip_op = self.operators_qutip[o] ** exponent
                op *= qutip_op
            HDrive_numerical_qutip += v * op

        HD_qubit = Qobj(HDrive_numerical_qutip)

        self.numerical_qubit = {
            'qubit_freq': qubit_freq,
            'drive_freq': qubit_freq + detuning,
            'H0' : H0,
            'HDrive': [HD_qubit, lambda t, args=None: np.cos((qubit_freq + detuning) * t)],
            'psi0' : states[0],
            'Projectors' : [projector_0, projector_1]
        }
        
        





