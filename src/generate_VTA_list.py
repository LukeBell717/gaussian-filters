import os
import pickle
import numpy as np
from itertools import product
from qutip import qzero, qeye, sigmax, sigmay, sigmaz, tensor
from IPython.display import display, Latex
from spin_chain import properties

# define Pauli matrices
σ_x = sigmax()
σ_y = sigmay()
σ_z = sigmaz()

def exact(N_qubits, Jx, Jy, Jz, periodic_bc, M, α_start, α_end, α_steps, Es):
    
    '''
    computes list of VTA_exact operators for a sweep over α
    '''
    # specify tolerance for "properties" function
    tolerance = 1e-9
    
    # collect properties of spin chain
    H4, _, _, _, _, _ = properties(N_qubits, Jx, Jy, Jz, periodic_bc, tolerance)

    # define array over which we will sweep α and 
    # compute VTA_exact_list
    α_array = np.linspace(α_start, α_end, α_steps)
    return [(-M*(α**2)/2 * (H4 - Es)**2).expm() for α in α_array]

def gaussian_weight_tensor_method(α_start, α_end, α_steps):
    
    '''
    list of VTAs using the Gaussian Weight Tensor approach
    '''
    
    # start the timer for the operation 
    start_time = time.time()
    
    # define array over which we will sweep α
    α_array = np.linspace(α_start, α_end, α_steps)
    
    # calculate the vacuum transition amplitude for N = 4
    VTA_list = []
    for α in α_array: 
        VTA = sum(
                W4(α, k0, k3, k2, k1)*W4(α, k0, k1, k2, k3) * \
                W4(α, k2, k1, k0, k3)*W4(α, k2, k3, k0, k1) * \
                (ν4_bras[k3]*μ4_kets[k2])*(μ4_bras[k2]*ν4_kets[k1]) * \
                (ν4_bras[k1]*μ4_kets[k0])*(ν4_kets[k3]*μ4_bras[k0])  
                for k3 in range(len(d)) \
                for k2 in range(len(c)) \
                for k1 in range(len(b)) \
                for k0 in range(len(a)) \
                )
        # adjust dimensions of VTA and append to VTA_list
        VTA_adjusted = Qobj(VTA, dims = [[2]*4, [2]*4])
        VTA_list.append(VTA_adjusted)
        
    # Saving the list of quantum objects to a file
    path = os.getcwd() + f'/four_site_data/VTA{N}_GWT_{α_start}_{α_end}_{α_steps}.pkl'

    # save VTA_list to directory specified by path 
    with open(path, 'wb') as file:
        pickle.dump(VTA_list, file)
        
    # to upload saved data use following code
#     with open(path, 'rb') as file:
#         VTA_list_new = pickle.load(file)
        
    print("%s seconds" % (time.time() - start_time))
    return VTA_list



def generate_SWAP_operators(N, Jx, Jy, Jz):
    
    '''
    generate list of SWAP operators that needed to compute VTA 
    using the SU(2) expansion method
    '''
    
    if N % 2 != 0: 
        raise ValueError("Please enter an even number of sites.")
        
    # define zero and identity matrices corresponding to dimensions of VTA
    zeros_N = qzero([2]*N)
    I_N = qeye([2]*N)
    
    # define Pauli matrices and constants
    σ_x = sigmax()
    σ_y = sigmay()
    σ_z = sigmaz()
    
    # Interaction coefficients, which we assume are uniform throughout the lattice
    Jx_list = Jx*np.ones(N)
    Jy_list = Jy*np.ones(N)
    Jz_list = Jz*np.ones(N)

    # Setup operators for individual qubits; 
    # here σ_x_list[j] = X_j, σ_y_list[j] = Y_j, and σ_z_list[j] = Z_j
    # since the Pauli matrix occupies the jth location in the tensor product of N terms
    # for which (N-1) terms are the identity
    σ_x_list, σ_y_list, σ_z_list = [], [], []

    for i in range(N):
        op_list = [qeye(2)]*N
        op_list[i] = σ_x
        σ_x_list.append(tensor(op_list))
        op_list[i] = σ_y
        σ_y_list.append(tensor(op_list))
        op_list[i] = σ_z
        σ_z_list.append(tensor(op_list))

    # define empty lists for + and - projection operators
    π_list = []
    
    # collect list of all tuples corresponding to π_p and π_m 
    # SWAP operators
    for k in range(N):

        # find H_ij, the Hamiltonian between the ith and jth sites 
        H_kl = Jx_list[k] * σ_x_list[k] * σ_x_list[(k + 1) % N] + \
               Jy_list[k] * σ_y_list[k] * σ_y_list[(k + 1) % N] + \
               Jz_list[k] * σ_z_list[k] * σ_z_list[(k + 1) % N]
        
        # add π_p to π_m to π_p_list and π_m_list, respectively
        π_p = (3 + H_kl)/4
        π_m = (1 - H_kl)/4
        π_list.append((π_p, π_m))
    
    # check to ensure projectors obey established summation and orthogonality relations
    π_kl_bool_list = []
    for π_kl in π_list: 
        π_kl_bool_list.append(π_kl[0] * π_kl[1] == zeros_N and \
                              π_kl[0] + π_kl[1] == I_N)

    if all(π_kl_bool_list):
#         display(Latex(r'$ \pi^{+}_{kl} \pi^{-}_{kl} = 0 \text{ and } $'
#                       r'$\pi^{+}_{kl} + \pi^{-}_{kl} = \mathbb{1}$' 
#                      rf'$ \ \forall \ k,l \in \{{1, \dots, {N} \}}$'))
        return π_list
    else: 
        display(Latex(r'$ \pi^{+}_{kl} \pi^{-}_{kl} \neq 0 \text{ of } $'
                      r'$\pi^{+}_{kl} + \pi^{-}_{kl} \neq \mathbb{1}$' 
                     rf'$ \ \forall \ k,l \in \{{1, \dots, {N} \}}$'))
        raise ValueError(f'SWAP operators do not obey the desired summation and' + \
                          ' orthogonality conditions')
        
def SU2_manual(N, α_start, α_end, α_steps, Es): 
    
    
    '''
    compute list of VTAs using an approach that explicitly
    defines an octuple for loop 
    '''
    
    # collect list of SWAP operators
    π_list = generate_SWAP_operators(N)

    # define projection operators where π_kl is a tuple 
    # such that π_kl[0] = π^{+}_{kl} and π_kl[1] = π^{-}_{kl}
    π12 = π_list[0]
    π23 = π_list[1]
    π34 = π_list[2]
    π41 = π_list[3]

    # define array over which we will sweep α and 
    # define r
    α_array = np.linspace(α_start, α_end, α_steps)      
    r = 2 + Es/2

    # define 0 matrix corresponding to dimensions of VTA matrix
    zeros_N = qzero([2]*N)

    # compute VTA for each value of α in α_array
    VTA_list_SU2 = []
    for α in α_array:
        VTA4 = zeros_N
        for b7 in range(2):
            for b6 in range(2):
                for b5 in range(2):
                    for b4 in range(2):
                        for b3 in range(2):
                            for b2 in range(2):
                                for b1 in range(2):
                                    for b0 in range(2):
                                        VTA4 += np.exp((-2*α**2) * (
                                                    (r - 4 + 2*(b0 + b3 + b5 + b6))**2 + \
                                                    (r - 4 + 2*(b0 + b2 + b5 + b7))**2 + \
                                                    (r - 4 + 2*(b1 + b2 + b4 + b7))**2 + \
                                                    (r - 4 + 2*(b1 + b3 + b4 + b6))**2)) * \
                                                     π41[b7]*π23[b6]*π34[b5]*π12[b4] * \
                                                     π41[b3]*π23[b2]*π34[b1]*π12[b0]
        VTA_list_SU2.append(VTA4)       
    return VTA_list_SU2
        

def MPO4(π41, π34, π23, π12, α, Es, b7, b6, b5, b4, b3, b2, b1, b0):
    
    '''
    generate matrix product operator corresponding to a unique α 
    and unique tuple of binary variables (b_0, ..., b_7}) for 4 sites
    ''' 

    # define constant q
    q = 2 + Es/2

    # return matrix product operator for a given α and set of indices
    return np.exp((-2*α**2) * (
           (q - 4 + 2*(b0 + b3 + b5 + b6))**2 + \
           (q - 4 + 2*(b0 + b2 + b5 + b7))**2 + \
           (q - 4 + 2*(b1 + b2 + b4 + b7))**2 + \
           (q - 4 + 2*(b1 + b3 + b4 + b6))**2)) * \
            π41[b7]*π23[b6]*π34[b5]*π12[b4] * \
            π41[b3]*π23[b2]*π34[b1]*π12[b0]

def MPO6(π61, π56, π45, π34, π23, π12, α, Es, 
         b17, b16, b15, b14, b13, b12, \
         b11, b10, b9, b8, b7, b6, \
         b5, b4, b3, b2, b1, b0): 
    
    '''
    construct MPO for unique tuple of (b0, ..., b17) 
    to compute GSP(α, λ, Es) for 6 sites
    '''

    # compute w_b prefactor 
    w_b = np.exp((-2*α**2) * \
           ((Es/2 - 3 + 2*sum([b0, b5, b7, b9, b14, b16]))**2 + \
            (Es/2 - 3 + 2*sum([b0, b3, b8, b11, b13, b16]))**2 + \
            (Es/2 - 3 + 2*sum([b1, b3, b8, b10, b12, b17]))**2 + \
            (Es/2 - 3 + 2*sum([b1, b4, b6, b9, b14, b17]))**2 + \
            (Es/2 - 3 + 2*sum([b2, b4, b6, b11, b13, b15]))**2 + \
            (Es/2 - 3 + 2*sum([b2, b5, b7, b10, b12, b15]))**2))
    
    # compute Π operator 
    Π = π61[b17]*π45[b16]*π23[b15]*π56[b14]*π34[b13]*π12[b12] * \
        π61[b11]*π45[b10]*π23[b9]*π56[b8]*π34[b7]*π12[b6] * \
        π61[b5]*π45[b4]*π23[b3]*π56[b2]*π34[b1]*π12[b0]
    
    return w_b*Π


def SU2_automated(N, α_start, α_end, α_steps, Jx, Jy, Jz, Es, savefile, directory): 
    
    '''
    compute list of VTAs using an efficient, automated approach
    '''
    
    # define array over which we will sweep α
    α_array = np.linspace(α_start, α_end, α_steps)
    
    # collect list of SWAP operators
    π_list = generate_SWAP_operators(N, Jx, Jy, Jz)

    # generate all possible combinations of tuples 
    combination_tuples = list(product([0, 1], repeat=int(N*(N/2))))
    
    # check to see if VTA file already exists
    VTA_list_path = directory + f'/data/{N}_sites/VTA_data/' \
           + f'VTA{N}_list_{α_start}_{α_end}_{α_steps}_{round(Es, 2)}'
    
    if os.path.exists(VTA_list_path):

        # save VTA_list to directory specified by path 
        with open(VTA_list_path, 'rb') as file:
            VTA_list = pickle.load(file)
        
        return VTA_list
    
    else: 
    
        if N == 4: 

            # define projection operators where π_kl is a tuple 
            # such that π_kl[0] = π^{+}_{kl} and π_kl[1] = π^{-}_{kl}
            π12 = π_list[0]
            π23 = π_list[1]
            π34 = π_list[2]
            π41 = π_list[3]

            # return the sum of all VTA
            VTA4_list = [sum(MPO4(π41, π34, π23, π12, α, Es, *combination) \
                        for combination in combination_tuples) \
                        for α in α_array]

            if savefile: 

                # create path with which to store data
                filename = f'{directory}/data/{N}_sites/VTA_data/' \
                           f'VTA4_list_{α_start}_{α_end}_{α_steps}_{round(Es, 2)}'

                # Check if the file already exists
                if os.path.exists(filename):
                    # If it exists, remove it
                    os.remove(filename)

                # save VTA_list to directory specified by path 
                with open(filename, 'wb') as file:
                    pickle.dump(VTA4_list, file)

            return VTA4_list

        if N == 6:

            # define projection operators where π_kl is a tuple 
            # such that π_kl[0] = π^{+}_{kl} and π_kl[1] = π^{-}_{kl}
            π12 = π_list[0]
            π23 = π_list[1]
            π34 = π_list[2]
            π45 = π_list[3]
            π56 = π_list[4]
            π61 = π_list[5]

            # return the sum of all VTA
            VTA6_list = [sum(MPO6(π61, π56, π45, π34, π23, π12, α, Es, *combination) \
                        for combination in combination_tuples) \
                        for α in α_array] 

            if savefile: 

                # create path with which to store data
                filename = f'/Users/lukebell/Documents/boson_gang/data/{N}_sites/VTA_data/' \
                           f'VTA6_list_{α_start}_{α_end}_{α_steps}_{round(Es, 2)}'

                # Check if the file already exists
                if os.path.exists(filename):
                    # If it exists, remove it
                    os.remove(filename)

                # save VTA_list to directory specified by path 
                with open(filename, 'wb') as file:
                    pickle.dump(VTA6_list, file)

            return VTA6_list
        