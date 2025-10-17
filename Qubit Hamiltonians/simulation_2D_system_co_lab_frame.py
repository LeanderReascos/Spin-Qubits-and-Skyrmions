from functools import partial
import sympy as sp
import numpy as np

#import PyQt5.QtCore
import os

#plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), 'Qt', 'plugins', 'platforms')
#os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

# Load directory ../Packages/PySW/ to path
import sys
sys.path.append('../Packages/PySW_old/')

from Modules.sympy.classes import *
from Modules.sympy.utils import *
from Modules.sympy.untruncated.solver import *

from Simulation_utils import Hamiltonian2D_Elliptical_Quantum_Well, CylindricalMagnet, h_bar
import json

from parallel_utils import compute_U_tlist, compute_Ut_tlist

simulation_params = json.load(open('simulation_params.json', 'r'))

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#plt.rc('axes', titlesize=18)     # fontsize of the axes title
#plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
#plt.rc('legend', fontsize=16)    # legend fontsize
#
## Transparent background for figures but keep the legend background white but its text color black
#plt.rcParams['legend.facecolor'] = 'white'
#plt.rcParams['legend.edgecolor'] = 'none'
#plt.rcParams['legend.framealpha'] = 0.8
#plt.rcParams['legend.shadow'] = False
#plt.rcParams['figure.facecolor'] = 'none'
#plt.rcParams['axes.facecolor'] = 'none'
## White spins, axis and ticks 
#plt.rcParams['text.color'] = 'black'
#plt.rcParams['axes.labelcolor'] = 'white'
#plt.rcParams['xtick.color'] = 'white'
#plt.rcParams['ytick.color'] = 'white'
#plt.rcParams['axes.labelcolor'] = 'white'
#plt.rcParams['axes.edgecolor'] = 'white'


class HObject:
    def __init__(self, H):
        self.H = H
        eigvals, eigvecs = np.linalg.eig(H)
        indx_sorted_eigvals = np.argsort(eigvals)
        
        self.eigvals = eigvals[indx_sorted_eigvals]
        self.eigvecs = [eigvecs[:, arg] for arg in indx_sorted_eigvals]

        self.projectors = [state.reshape(-1, 1).dot(state.reshape(-1, 1).T.conj()) for state in self.eigvecs]

        self.grond_state = eigvecs[0]

        self.initial_state = np.copy(self.grond_state)

    # latex representation of the object
    def _repr_latex_(self):
        # Whole description, hamiltonian, eigenvalues, eigenvectors, projectors and ground state
        latex_str = r'\begin{align*}'
        latex_str += r'\text{Hamiltonian} & : ' + sp.latex(sp.Matrix(self.H)) + r'\\'
        latex_str += r'\text{Eigenvalues} & : ' + sp.latex(sp.Matrix(self.eigvals)) + r'\\'
        latex_str += r'\text{Eigenvectors} & : ' + sp.latex(sp.Matrix(self.eigvecs)) + r'\\'
        latex_str += r'\text{Projectors} & : ' + r'\\'
        for projector in self.projectors:
            latex_str += sp.latex(sp.Matrix(projector)) + r'\\'

        latex_str += r'\text{Ground State} & : ' + sp.latex(sp.Matrix(self.grond_state)) + r'\\'
        latex_str += r'\end{align*}'
        return latex_str

    def compute_projectors(self, state):
        result_projectors = []
        for projector in self.projectors:
            result_projectors.append(state.T.conj().dot(projector).dot(state))
        return result_projectors
    

def H_qubit(t, H0, Hx, Hy, omega_drive):
    return H0 + Hx * np.cos(omega_drive * t) + Hy * np.sin(omega_drive * t)


def simulate_2D_system_co_lab_frame():

    skyrmion = CylindricalMagnet(simulation_params['M'], 50, simulation_params['L'], simulation_params['Delta z'])
    qubit_co = Hamiltonian2D_Elliptical_Quantum_Well(simulation_params['lx'], simulation_params['ly'], simulation_params['Bz'], skyrmion, minimal_coupling=True)
    qubit_lab = Hamiltonian2D_Elliptical_Quantum_Well(simulation_params['lx'], simulation_params['ly'], simulation_params['Bz'], skyrmion, minimal_coupling=True)

    
    order_perturbation = 2
    bosonic_space_truncation = 3
    order_perturbation_driving = 2
    lx_lE = 0.1
    
    qubit_co.get_symbolic_hamiltonian(bosonic_space_truncation)
    qubit_co.perturbation_expansion(order=order_perturbation, full_diagonal=True)

    driving_frequency = np.float64(qubit_co.qubit_frequency_perturbative)  * 1.0
    # 3.415126257666941 Numerical
    # 3.415625675446527normal
    # 3.4150992415075914
    f_driving_frequency = driving_frequency / (2 * np.pi * 1e9)
    print(f'Driving frequency: {f_driving_frequency} GHz')

    qubit_co.add_driving_expansion(lx_lE, driving_frequency, order=order_perturbation_driving, co_moving_frame=True)

    H0_Eff_co_moving, Hdrive_co_x, Hdrive_co_y = qubit_co.effective_qubit_hamiltonian()
    print('\nEffective Hamiltonian Co-Moving Frame')
    print(H0_Eff_co_moving)
    print(Hdrive_co_x)
    print(Hdrive_co_y)
    H_Eff_co_moving = partial(H_qubit, H0=H0_Eff_co_moving, Hx=Hdrive_co_x, Hy=Hdrive_co_y, omega_drive=driving_frequency)
    H0_Eff_Obj_co_moving = HObject(H0_Eff_co_moving)

    H0_RWA_co_moving, Hdrive_RWA_co_x, Hdrive_RWA_co_y = qubit_co.effective_qubit_hamiltonian(RWA=True)
    print('\nEffective Hamiltonian Co-Moving Frame RWA')
    print(H0_RWA_co_moving)
    print(Hdrive_RWA_co_x)
    print(Hdrive_RWA_co_y)
    H_RWA_co_moving = H0_RWA_co_moving + Hdrive_RWA_co_x + Hdrive_RWA_co_y
    H0_RWA_Obj_co_moving = HObject(H0_RWA_co_moving)

    H0_Full, Hdrive_Full_x, Hdrive_Full_y = qubit_co.full_hamiltonian()
    H_Full = partial(H_qubit, H0=H0_Full, Hx=Hdrive_Full_x, Hy=Hdrive_Full_y, omega_drive=driving_frequency)
    H0_Full_Obj = HObject(H0_Full)

    numerical_qubit_frequency = (H0_Full_Obj.eigvals[1] - H0_Full_Obj.eigvals[0]) / (h_bar * 2 * np.pi * 1e9)

    
    print(f"Qubit Frequency -> Perturbative: {qubit_co.qubit_frequency_perturbative / (2*np.pi * 1e9)} GHz | Prev: {qubit_co.qubit_frequency() / (2*np.pi * 1e9)} GHz")
    print(f"Qubit Frequency -> Numerical: {numerical_qubit_frequency} GHz")

    qubit_lab.get_symbolic_hamiltonian(bosonic_space_truncation)
    qubit_lab.perturbation_expansion(order=order_perturbation, full_diagonal=True)
    qubit_lab.add_driving_expansion(lx_lE, driving_frequency, order=order_perturbation_driving, co_moving_frame=False)

    H0_Eff_lab_frame, Hdrive_lab_x, Hdrive_lab_y = qubit_lab.effective_qubit_hamiltonian()
    print('\nEffective Hamiltonian Lab Frame')
    print(H0_Eff_lab_frame)
    print(Hdrive_lab_x)
    print(Hdrive_lab_y)
    H_Eff_lab_frame = partial(H_qubit, H0=H0_Eff_lab_frame, Hx=Hdrive_lab_x, Hy=Hdrive_lab_y, omega_drive=driving_frequency)
    H0_Eff_Obj_lab_frame = HObject(H0_Eff_lab_frame)

    H0_RWA_lab_frame, Hdrive_RWA_lab_x, Hdrive_RWA_lab_y = qubit_lab.effective_qubit_hamiltonian(RWA=True)
    print('\nEffective Hamiltonian Lab Frame RWA')
    print(H0_RWA_lab_frame)
    print(Hdrive_RWA_lab_x)
    print(Hdrive_RWA_lab_y)
    H_RWA_lab_frame = H0_RWA_lab_frame + Hdrive_RWA_lab_x + Hdrive_RWA_lab_y
    H0_RWA_Obj_lab_frame = HObject(H0_RWA_lab_frame)



    t0 = 0
    tf = 2 * np.pi / qubit_co.Rabi_frequency
    N_points = 100000
    dt = (tf - t0) / N_points

    t = np.linspace(t0, tf, N_points + 1)
    ks = np.arange(0, N_points + 1)

    Us_co_moving = compute_Ut_tlist(ks, dt, H_Eff_co_moving)
    Us_lab = compute_Ut_tlist(ks, dt, H_Eff_lab_frame)
    Us_RWA_co_moving = compute_U_tlist(t, H_RWA_co_moving)
    Us_RWA_lab = compute_U_tlist(t, H_RWA_lab_frame)
    Us_Full = compute_Ut_tlist(ks, dt, H_Full)

    Projects_Eff_co_moving = []
    Projects_RWA_co_moving = []
    Projects_Eff_lab = []
    Projects_RWA_lab = []
    Projects_Full = []

    for k in tqdm(range(0, N_points+1), desc='Computing projectors'):
        Projects_Eff_co_moving.append(H0_Eff_Obj_co_moving.compute_projectors(Us_co_moving[k] @ H0_Eff_Obj_co_moving.grond_state))
        Projects_RWA_co_moving.append(H0_RWA_Obj_co_moving.compute_projectors(Us_RWA_co_moving[k] @ H0_RWA_Obj_co_moving.grond_state))
        Projects_Eff_lab.append(H0_Eff_Obj_lab_frame.compute_projectors(Us_lab[k] @ H0_Eff_Obj_lab_frame.grond_state))
        Projects_RWA_lab.append(H0_RWA_Obj_lab_frame.compute_projectors(Us_RWA_lab[k] @ H0_RWA_Obj_lab_frame.grond_state))
        Projects_Full.append(H0_Full_Obj.compute_projectors(Us_Full[k] @ H0_Full_Obj.grond_state))

    Projects_Eff_co_moving = np.real(np.array(Projects_Eff_co_moving))
    Projects_RWA_co_moving = np.real(np.array(Projects_RWA_co_moving))
    Projects_Eff_lab = np.real(np.array(Projects_Eff_lab))
    Projects_RWA_lab = np.real(np.array(Projects_RWA_lab))
    Projects_Full = np.real(np.array(Projects_Full))
    
    print('Plotting Started')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    print('Plotting')

    p0 = 1-qubit_co.DriveStrength_perturvative**2 * np.sin(qubit_co.Rabi_frequency/2 * t)**2/ qubit_co.Rabi_frequency**2

    N_LEVEL = 0
    ax.plot(t, p0, label=r'$\Omega^2 \sin^2(\frac{\Omega t}{2})$', color='white', linestyle='--')
    ax.plot(t, Projects_Eff_co_moving[:, N_LEVEL], label=r'Eff. Co-Moving Frame', color='blue', marker='o', markersize=1)
    ax.plot(t, Projects_RWA_co_moving[:, N_LEVEL], label=r'RWA Co-Moving Frame', color='red', marker='o', markersize=1)
    ax.plot(t, Projects_Eff_lab[:, N_LEVEL], label=r'Eff. Lab Frame', color='green', marker='d', markersize=1)
    ax.plot(t, Projects_RWA_lab[:, N_LEVEL], label=r'RWA Lab Frame', color='orange')
    #ax.plot(t, Projects_Full[:, N_LEVEL], label=r'Full', color='purple')
    for N_LEVEL in range(1):
        ax.plot(t, Projects_Full[:, N_LEVEL], label=f'$E_{N_LEVEL}$')

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\langle 0 | \psi(t) \rangle$')
    ax.set_ylim(-0.01, 1.01)

    ax.legend()

    fig.savefig('simulation_2D_system_co_lab_frame.png', dpi=300, bbox_inches='tight')
    print('Saved')
    #plt.show()

def parameters():
    skyrmion = CylindricalMagnet(simulation_params['M'], 50, simulation_params['L'], simulation_params['Delta z'])

    lx = simulation_params['lx']
    Ly = np.linspace(1, 2) * lx
    F1 = []
    F2 = []
    G1 = []
    G2 = []

    Dp1 = []
    Dp2 = []
    Dm1 = []
    Dm2 = []
    Tp = []
    Tm = []

    for ly in Ly:
        qubit = Hamiltonian2D_Elliptical_Quantum_Well(lx, ly, simulation_params['Bz'], skyrmion, minimal_coupling=False)
        F1.append(np.abs(qubit.f1))
        F2.append(np.abs(qubit.f2))
        G1.append(np.abs(qubit.g1))
        G2.append(np.abs(qubit.g2))
        Dp1.append(np.abs(qubit.delta_plus_1))
        Dp2.append(np.abs(qubit.delta_plus_2))
        Dm1.append(np.abs(qubit.delta_minus_1))
        Dm2.append(np.abs(qubit.delta_minus_2))
        Tp.append(np.abs(qubit.tp))
        Tm.append(np.abs(qubit.tm))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'black', 'brown', 'pink', 'gray', 'cyan']
    ax.plot(Ly, F1, label=r'$f_1$', color=colors[0])
    ax.plot(Ly, F2, label=r'$f_2$', color=colors[0], linestyle='--')
    ax.plot(Ly, G1, label=r'$g_1$', color=colors[0], linestyle=':')
    ax.plot(Ly, G2, label=r'$g_2$', color=colors[0], linestyle='-.')
    ax.plot(Ly, Dp1, label=r'$\delta_+^{(1)}$', color=colors[1])
    ax.plot(Ly, Dp2, label=r'$\delta_+^{(2)}$', color=colors[1], linestyle='--')
    ax.plot(Ly, Dm1, label=r'$\delta_-^{(1)}$', color=colors[1], linestyle=':')
    ax.plot(Ly, Dm2, label=r'$\delta_-^{(2)}$', color=colors[1], linestyle='-.')
    ax.plot(Ly, Tp, label=r'$t_+$', color=colors[2], linestyle=':', marker='d', markersize=1)
    ax.plot(Ly, Tm, label=r'$t_-$', color=colors[2], linestyle=':', marker='o', markersize=1)

    ax.set_xlabel(r'$L_y$')
    ax.set_ylabel(r'Parameters')
    ax.legend()
    plt.show()


if __name__ == '__main__':

    simulate_2D_system_co_lab_frame()
