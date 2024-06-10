import os 
import numpy as np
from functools import reduce
from qutip import sigmax, sigmay, sigmaz, qeye, tensor, fidelity, simdiag, Qobj

def properties(N, Jx, Jy, Jz, periodic_bc, tolerance):

    
    if N % 2 != 0: 
        raise ValueError("Please enter an even number of sites")
    
    # define Pauli matrices and constants
    σ_x = sigmax()
    σ_y = sigmay()
    σ_z = sigmaz()
    π = np.pi

    # Interaction coefficients, which we assume are uniform throughout the lattice
    Jx_list = Jx*np.ones(N)
    Jy_list = Jy*np.ones(N)
    Jz_list = Jz*np.ones(N)

    # Setup operators for individual qubits; 
    # here sx_list[j] = X_j, sy_list[j] = Y_j, and sz_list[j] = Z_j
    # since the Pauli matrix occupies the jth location in the tensor product of N terms
    # of which N-1 terms are the identity
    sx_list, sy_list, sz_list = [], [], []

    for i in range(N):
        op_list = [qeye(2)]*N
        op_list[i] = σ_x
        sx_list.append(tensor(op_list))
        op_list[i] = σ_y
        sy_list.append(tensor(op_list))
        op_list[i] = σ_z
        sz_list.append(tensor(op_list))

    # define variable for total Hamiltonian H_N and the list of all local 
    # Hamiltonians H_list
    H_N = 0 
    H_list = []
    
    # collect 
    for j in range(N - 1):

        # find H_ij, the Hamiltonian between the ith and jth sites 
        H_ij = Jx_list[j] * sx_list[j] * sx_list[j + 1] + \
               Jy_list[j] * sy_list[j] * sy_list[j + 1] + \
               Jz_list[j] * sz_list[j] * sz_list[j + 1]
        
        # add H_ij to H_N and append H_ij to H_list
        H_N += H_ij
        H_list.append(H_ij)

    # execute if periodic boundary conditions are specified
    if periodic_bc: 
        
        # find H_N1, the Hamiltonian between the Nth and first site
        H_N1 = Jx_list[N-1] * sx_list[N - 1] * sx_list[0] + \
               Jy_list[N-1] * sy_list[N - 1] * sy_list[0] + \
               Jz_list[N-1] * sz_list[N - 1] * sz_list[0]

        # add H_N1 to H_N and append H_N1 to H_list
        H_N += H_N1
        H_list.append(H_N1)

    # find eigenavlues and eigenstates of Hamiltonian 
    eigenvalues, eigenstates = H_N.eigenstates()

    # find indices of smallest eigenvalues, which correspond to the energy(ies) 
    # of the ground state (space in the case of degeneracy); 
    E_0 = min(eigenvalues)
    indices = [index for index, value in enumerate(eigenvalues) \
               if np.allclose(value, E_0, tolerance)]

    # find eigenstates corresponding to ground state
    eigenstates_list = eigenstates[indices]

    # create sum of density matrices of ground states in case ground state is degenerate
    ρ_ground_state = 0 
    for j in range(len(eigenstates_list)):
        ρ_ground_state += (eigenstates_list[j])*(eigenstates_list[j]).dag()

    # return normalized ground state
    return H_N, H_list, eigenstates, eigenvalues, E_0, ρ_ground_state

def orthonormal_eigenstates(eigenstates, tolerance):
    # find the inner product between all eigenstates; 
    # note that if eigenstates are orthonormal this should form 
    # the identity matrix with dimension equal to number of eigenstates
    fidelity_list = []
    for i, eigenstate in enumerate(eigenstates):
        row = []
        for j in range(len(eigenstates)):
            row.append(fidelity(eigenstate, eigenstates[j]))
        fidelity_list.append(np.array(row))


    # round entry in fidelity_matrix according to value of tolerance
    # very whether this matrix if equal to the identity
    if np.array_equal(np.round(fidelity_list, decimals= -int(np.log10(tolerance))), np.eye(len(eigenstates))): 
        print(f'Your eigenstates form an orthonormal basis!')
    else: 
        print(f'Your eigenstates are not orthonormal') 
        
        
def symmetry_operator(N):
    
    '''
    collect symmetry operator used to construct
    symmetry eigenstates through simultaneous diagonalization
    '''
    
    # define Pauli matrices and constants
    σ_x = sigmax()
    σ_y = sigmay()
    σ_z = sigmaz()

    # collect indices for σik operators
    σik_indices = [(2 + i, N - i) for i in range(0, int(N/2) - 1)]

    # collect all σik operators 
    σik_list = []
    for indices_tuple in σik_indices:

        # collect at which sites we will insert Pauli matrices
        site1 = indices_tuple[0] - 1
        site2 = indices_tuple[1] - 1

        # define list of identity operators and insert Pauli matrices
        # at sites specified by indices_tuple
        op_list = [qeye(2)]*N
        op_list[site1] = σ_x
        op_list[site2] = σ_x
        Xik = tensor(op_list)
        op_list[site1] = σ_y
        op_list[site2] = σ_y
        Yik = tensor(op_list)
        op_list[site1] = σ_z
        op_list[site2] = σ_z
        Zik = tensor(op_list) 

        # append SWAP operator to σik_list
        SWAP = (Xik + Yik + Zik + qeye([2]*N))/2
        σik_list.append(SWAP)
    
    # compute symmetry operator
    σik = reduce(lambda x,y: x*y, σik_list)
    
    return σik
            
def collect_spin_ops(N, Jx, Jy, Jz, periodic_bc):

    '''
    constructs and returns H4, S2, Sz, and σik
    '''

    # define Pauli matrices and constants
    σ_x = sigmax()
    σ_y = sigmay()
    σ_z = sigmaz()

    # Interaction coefficients, which we assume are uniform throughout the lattice
    Jx_list = Jx*np.ones(N)
    Jy_list = Jy*np.ones(N)
    Jz_list = Jz*np.ones(N)

    # define empty lists to store weight-four Pauli operators 
    X_list, Y_list, Z_list = [], [], []

    # collect weight-four pauli operators 
    for i in range(N):
        op_list = [qeye(2)]*N
        op_list[i] = σ_x
        X_list.append(tensor(op_list))
        op_list[i] = σ_y
        Y_list.append(tensor(op_list))
        op_list[i] = σ_z
        Z_list.append(tensor(op_list))

    # define variable for total Hamiltonian HN 
    HN = 0 

    # collect 
    for j in range(N - 1):

        # find H_ij, the Hamiltonian between the ith and jth sites 
        H_ij = Jx_list[j] * X_list[j] * X_list[j + 1] + \
               Jy_list[j] * Y_list[j] * Y_list[j + 1] + \
               Jz_list[j] * Z_list[j] * Z_list[j + 1]

        # add H_ij to H_N and append H_ij to H_list
        HN += H_ij

    # execute if periodic boundary conditions are specified
    if periodic_bc: 

        # find H_N1, the Hamiltonian between the Nth and first site
        H_N1 = Jx_list[N-1] * X_list[N - 1] * X_list[0] + \
               Jy_list[N-1] * Y_list[N - 1] * Y_list[0] + \
               Jz_list[N-1] * Z_list[N - 1] * Z_list[0]

        # add H_N1 to H_N and append H_N1 to H_list
        HN += H_N1

    # compute Sx, Sy, Sz
    Sx = sum(X_list)/2
    Sy = sum(Y_list)/2
    Sz = sum(Z_list)/2

    # compute S^2
    S2 = Sx**2 + Sy**2 + Sz**2

    # compute symmetry operator 
    σik = symmetry_operator(N)

    return HN, S2, Sz, σik


def symmetry_eigvecs(N, Jx, Jy, Jz, periodic_bc, return_operators = False):
    
    '''
    find the simultaneous eigenvectors of HN, S^2, Sz, and σik
    '''
    
    # collect spin operators
    HN, S2, Sz, σik = collect_spin_ops(N, Jx, Jy, Jz, periodic_bc)
    
    
    # Loop until the function runs without an error
    successful = False
    attempts = 0
    max_attempts = 100  # Maximum number of attempts to prevent infinite loop

    # find simultaneous (symmetry) eigenvalues and 
    # eigenvectors of H4, S^2, Sz, and σik
    while not successful and attempts < max_attempts:
        try:
            eigEsym_array, eigvecs = simdiag([HN, S2, Sz, σik])
            successful = True  # Exit the loop if no error occurs
        except Exception as e:
            # print(f"Attempt {attempts + 1} failed: {e}")
            attempts += 1
    
#     eigEsym_array_T = eigEsym_array.T
#    
#     # note, this code may need to be adjusted if we wish to analyze 
#     # spin chains with anistropy parameter λ != 1

#     # fix U(1) global phase for each eigenvector
#     eigVsym_fp = eigVsym.copy()  #new obj. to hold the fixed phase eigenvectors
#     for ii in range(2**4):
#         # print(ii)
#         if np.sqrt((eigEsym_array_T[ii,0]-(-8))**2 + \
#                    (eigEsym_array_T[ii,1]-0)**2) < 1e-10:  #|λ0>: E=-8, s(s+1)=0
#             vec = eigVsym_fp[ii].data_as()
#             c0011 = vec[3] #int('0011',base=2)==3
#             eigVsym_fp[ii] *= -np.exp(-1j*np.angle(c0011))  #make c0011<0, see Eq. (513) in the notes
#             vec = eigVsym_fp[ii].data_as()
#             # display(Statevector(vec).draw('latex'))
#         elif np.sqrt((eigEsym_array_T[ii,0]-0)**2 + \
#                      (eigEsym_array_T[ii,1]-0)**2) < 1e-10:  #|λ1>: E=0, s(s+1)=0
#             vec = eigVsym_fp[ii].data_as()
#             c0011 = vec[3] #int('0011',base=2)==3
#             eigVsym_fp[ii] *= np.exp(-1j*np.angle(c0011)) #make c0011>0, see Eq. (514) in the notes
#             vec = eigVsym_fp[ii].data_as()
#             # display(Statevector(vec).draw('latex'))        
#         else:
#             #continue  #uncomment this line to skip the code below that fixes other phases \
#             # (optional, because phases cancel in |E_k><E_k|)
#             vec = eigVsym_fp[ii].data_as()
#             c_idx = np.where(abs(vec)>1e-10)[0][0]  #first nonzero element
#             c = vec[c_idx]
#             eigVsym_fp[ii] *= np.exp(-1j*np.angle(c))
#             vec = eigVsym_fp[ii].data_as()
#             # display(Statevector(vec).draw('latex'))  
            
    # rename eigVsym_fp
    #eigvecs = eigVsym_fp
        
    # find eigenvalues of H4, S^2, Sz, and σik
    HN_eigvals = eigEsym_array[0]
    S2_eigvals = eigEsym_array[1]
    Sz_eigvals = eigEsym_array[2]
    σik_eigvals = eigEsym_array[3]

    # reconstruct H4 through spectral decomposition 
    HN_bool = (HN == sum(λ1*eigvec*eigvec.dag() for λ1, eigvec \
                        in zip(HN_eigvals, eigvecs)))

    # reconstruct S^2 through spectral decomposition 
    S2_bool = (S2 == sum(λ2*eigvec*eigvec.dag() for λ2, eigvec \
                      in zip(S2_eigvals, eigvecs)))

    # reconstruct Sz through spectral decomposition 
    Sz_bool = (Sz == sum(λ3*eigvec*eigvec.dag() for λ3, eigvec \
                      in zip(Sz_eigvals, eigvecs)))

    # reconstruct σik through spectral decomposition 
    σik_bool = (σik == sum(λ4*eigvec*eigvec.dag() for λ4, eigvec \
                       in zip(σik_eigvals, eigvecs)))
    
    # return symmetry eigenvectors only if we can reconstruct
    # H4, S2, Sz, and σik through spectral decomposition
    if HN_bool and S2_bool and Sz_bool and σik_bool: 

        # convert eigenstates from quantum objects into arrays
        eigvecs_array = [v.full() for v in eigvecs]

        # compute change of basis matrix P whose columns 
        # are composed of eigvecs
        P = Qobj(reduce(lambda x, y: np.hstack((x, y)), eigvecs_array), \
                 dims = [[2]*N, [2]*N])

        # compute density matrices of eigvecs
        ρ_list = [eigvec*eigvec.dag() for eigvec in eigvecs]
        
        if return_operators == False: 
            return P, eigvecs, ρ_list
        
        else: 
            return HN, S2, Sz, σik, P, eigvecs, ρ_list

    else: 

        raise ValueError('Unsuccessful diagonalization; please try again. ')