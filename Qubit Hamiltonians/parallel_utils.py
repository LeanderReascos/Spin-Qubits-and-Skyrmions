from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
from scipy.linalg import expm
from Simulation_utils import h_bar
import numpy as np

n_process = 10

def Unitary_time_evolution(t, H):
    return expm(-1j / h_bar * H * t)

def compute_U_tlist(t_list, H):
    chunk_size = len(t_list) // (n_process * 10)
    expm_wrapper = partial(Unitary_time_evolution, H=H)
    res = []
    with Pool(n_process) as p:
        res += tqdm(p.imap(expm_wrapper, t_list, chunksize=chunk_size), total=len(t_list), desc='Computing U(t)')

    return res

def Unitary_time_evolution_time_dependant(k, dt, H):
    if k == 0:
        return np.eye(H(0).shape[0], dtype=np.complex128)
    Ht = H(k * dt)
    return expm(-1j / h_bar * Ht * dt)


def compute_Ut_tlist(ks, dt, H):

    chunk_size = len(ks) // (n_process  * 10)

    expm_wrapper = partial(Unitary_time_evolution_time_dependant, dt=dt, H=H)
    res = []
    with Pool(n_process) as p:
        res += tqdm(p.imap(expm_wrapper, ks, chunksize=chunk_size), total=len(ks), desc='Computing U(t)')


    U_tlist = []
    cummulative_mul = np.eye(res[0].shape[0], dtype=np.complex128)
    for U in tqdm(res, desc='Multiplying U(t)'):
        cummulative_mul = U @ cummulative_mul
        U_tlist.append(cummulative_mul)
    
    return U_tlist