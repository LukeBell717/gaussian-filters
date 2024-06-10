# import quantities function from spin_chain module
import numpy as np
from functools import reduce
from spin_chain import properties
from qutip import qeye, sigmax, sigmay, sigmaz, tensor, Qobj

# define Pauli matrices
σ_x = sigmax()
σ_y = sigmay()
σ_z = sigmaz()

def eigendecompose_spin_operators(N, Jx, Jy, Jz):
    
    '''
    function to find the global eigenkets for blue and red bonds as well
    as the spectrum of each local 2-site Hamiltonian in the basis of these
    global eigenkets
    '''
    
    # find eigenstates of 2 site spin chain with coupling constants specified
    # by Jx, Jy, and Jz
    periodic_bc = False
    tolerance = 1e-12
    _, _, eigenstates2, eigenvalues2, _, _ = properties(2, Jx, Jy, Jz, periodic_bc, tolerance)
    
    # define singlet and triplet states
    S = eigenstates2[0]
    T0 = eigenstates2[3]
    Tn1 = eigenstates2[2]
    T1 = eigenstates2[1]

    # create basis of singlets and triplets that will be used to 
    # construct matrix product states
    ST_basis = [S, Tn1, T0, T1]
    
    # collect indices for H_ik Hamiltonians which will be used to convert 
    # blue global eigenkets to red global eigenkets
    H_ik_indices = [(2 + i, N - i) for i in range(0, int(N/2) - 1)]

    # collect all H_ik matrices, which are Hamiltonians containing 
    # XX + YY + ZZ at indices i and k specified by each tuple in H_SWAP_indices
    H_ik_list = []
    for indices_tuple in H_ik_indices:

        # collect at which sites we will insert Pauli matrices
        site1 = indices_tuple[0]
        site2 = indices_tuple[1]

        # define list of identity operators and insert Pauli matrices
        # at sites specified by indices_tuple
        op_list = [qeye(2)]*N
        op_list[site1 - 1] = σ_x
        op_list[site2 - 1] = σ_x
        HX_couplings = tensor(op_list)
        op_list[site1 - 1] = σ_y
        op_list[site2 - 1] = σ_y
        HY_couplings = tensor(op_list)
        op_list[site1 - 1] = σ_z
        op_list[site2 - 1] = σ_z
        HZ_couplings = tensor(op_list) 

        # append H_SWAP = HX_couplings + HY_couplings + HZ_couplings to H_ik_list
        H_ik_list.append(HX_couplings + HY_couplings + HZ_couplings)

    # collect product of SWAP matrices that will convert red eigenkets to blue eigenkets
    I = tensor([qeye(2)]*N)
    dim = 2**N
    SWAP_operator = Qobj(reduce(lambda x,y: x*y, \
                                [(H_ik + I)/2 for H_ik in H_ik_list]), dims = [[dim], [dim]])

    # construct global eigenkets for red and blue bonds for N sites
    blue_mpsN = [Qobj(eigvec) for eigvec in reduce(np.kron, [ST_basis]*(int(N/2)))]
    red_mpsN = [SWAP_operator(Qobj(eigvec)) for eigvec in reduce(np.kron, [ST_basis]*(int(N/2)))]
    
    # initialize empty list which will contain the spectrum of each local Hamiltonian 
    # in the basis of the blue and red global eigenkets
    H_local_spectrum_list = []

    # find spectra of all 2-site Hamiltonians
    for i in range(1, N+1, 2):

        # compute spectrum from tensor products of index 1 to i 
        Hij_spectrum_1i = np.tile(eigenvalues2, 2**(i-1))

        # compute spectrum of tensor products of inex 1 to N, i.e., the full spectrum
        Hij_spectrum_1N = np.repeat(Hij_spectrum_1i, np.ceil(2**(N - i - 1)))

        # append to Hij_spectrum_list
        H_local_spectrum_list.append(Hij_spectrum_1N)

    # extra spectra for blue and red bonds according to previous ordering of eigenkets
    blue_spectraN = H_local_spectrum_list
    red_spectraN = H_local_spectrum_list[::-1]
    
    return blue_spectraN, blue_mpsN, red_spectraN, red_mpsN


def reconstruct_spin_operators(HN_list, blue_spectraN, blue_mpsN, red_spectraN, red_mpsN):

    '''
    reconstruct list of local Hamiltonians from parameters yielded by the eigendecomposition 
    of all red and blue bonds
    '''

    # calculate the number of sites in spin chain
    N = 2*len(blue_spectraN)

    # reconstruct lists of Hamiltonians corresponding to blue and red bonds
    HN_blue_list_reconstructed = [sum([λ*vec*vec.dag() for λ, vec in \
                                  zip(blue_spectraN[i], blue_mpsN)]) for i in range(int(N/2))]
    HN_red_list_reconstructed = [sum([λ*vec*vec.dag() for λ, vec in \
                                 zip(red_spectraN[i], red_mpsN)]) for i in range(int(N/2))]

    # interleave HN_blue_list and HN_red_list to create an ordered list of local Hamiltonians 
    HN_list_reconstructed = np.concatenate((HN_blue_list_reconstructed, HN_red_list_reconstructed))
    HN_list_reconstructed[::2] = HN_blue_list_reconstructed
    HN_list_reconstructed[1::2] = HN_red_list_reconstructed 

    # extract lists of blue and red Hamiltonians from HN_list
    HN_blue_list = HN_list[::2]
    HN_red_list = HN_list[1::2]

    # verify that HN_list == HN_list_reconstructed using two methods
    verification1 = np.allclose(HN_list, HN_list_reconstructed, atol = 1e-12)

    # use for loop to check whether HN_list[i] == HN_list_reconstructed[i]
    HN_tot_bool = []
    for Hb1, Hb2, Hr1, Hr2 in zip(HN_blue_list_reconstructed, HN_blue_list, \
                                  HN_red_list_reconstructed, HN_red_list):

        HN_tot_bool.append(np.allclose(Hb1, Hb2, atol = 1e-12))
        HN_tot_bool.append(np.allclose(Hr1, Hr2, atol = 1e-12))

    verification2 = all(HN_tot_bool)

    # print whether both methods of verifying equality between Hamiltonians 
    # yield True or False
    if verification1 and verification2: 
        print('Local Hamiltonians are the same!')
    else: 
        print('Hamiltonians are different')


def compute_VTA_list_GWT(α_start, α_end, α_steps):
    
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