import numpy as np
from qutip import destroy, qeye, tensor

# define qubit identity 
I2 = qeye(2)

def D12_A(α, H2, E_0, cutoff): 
    
    '''
    operator that displaces modes 1 & 2 by weight of A = H12
    '''
    
    # define bosonic operators 
    a = destroy(cutoff)
    Ib = qeye(cutoff)
    
    # define argument of unconditional displacement operator
    α_arg = α*a.dag() - np.conj(α)*a
    
    return (tensor(tensor(α_arg, H2 - E_0).expm(), Ib, Ib, Ib, I2, I2)) \
            .permute([0, 5, 3, 4, 1, 2, 6, 7]) * \
           (tensor(Ib, tensor(α_arg, H2 - E_0).expm(), Ib, Ib, I2, I2)) \
            .permute([0, 1, 5, 4, 2, 3, 6, 7])

def D34_A(α, H2, E_0, cutoff):
      
    '''
    operator that displaces modes 3 & 4 by weight of A = H12
    '''
    
    # define bosonic operators 
    a = destroy(cutoff)
    Ib = qeye(cutoff)
    
    # define argument of unconditional displacement operator
    α_arg = α*a.dag() - np.conj(α)*a
    
    return (tensor(Ib, Ib, tensor(α_arg, H2 - E_0).expm(), Ib, I2, I2)) \
            .permute([0, 1, 2, 5, 3, 4, 6, 7]) * \
           (tensor(Ib, Ib, Ib, tensor(α_arg, H2 - E_0).expm(), I2, I2)) 

def D23_B(α, H2, E_0, cutoff): 
    
    '''
    operator that displaces modes 2 & 3 by weight of B = H23
    '''
    
    # define bosonic operators 
    a = destroy(cutoff)
    Ib = qeye(cutoff)
    
    # define argument of unconditional displacement operator
    α_arg = α*a.dag() - np.conj(α)*a
    
    return (tensor(Ib, tensor(α_arg, H2 - E_0).expm(), Ib, I2, Ib, I2)) \
            .permute([0, 1, 6, 4, 5, 2, 3, 7]) * \
           (tensor(Ib, Ib, tensor(α_arg, H2 - E_0).expm(), I2, Ib, I2)) \
            .permute([0, 1, 2, 6, 5, 3, 4, 7])

def D41_B(α, H2, E_0, cutoff):
    
    '''
    operator that displaces modes 4 & 1 by weight of B = H23
    '''
    
    # define bosonic operators 
    a = destroy(cutoff)
    Ib = qeye(cutoff)
    
    # define argument of unconditional displacement operator
    α_arg = α*a.dag() - np.conj(α)*a
    
    return (tensor(Ib, Ib, Ib, tensor(α_arg, H2 - E_0).expm(), I2, I2)) \
            .permute([0, 1, 2, 3, 6, 4, 5, 7]) * \
           (tensor(tensor(α_arg, H2 - E_0).expm(), Ib, Ib, I2, Ib, I2)) \
            .permute([0, 6, 3, 4, 5, 1, 2, 7])

def D34_C(α, H2, E_0, cutoff): 
    
    '''
    operator that displaces modes 3 & 4 by weight of C = H34
    '''
    
    # define bosonic operators 
    a = destroy(cutoff)
    Ib = qeye(cutoff)
    
    # define argument of unconditional displacement operator
    α_arg = α*a.dag() - np.conj(α)*a
    
    return (tensor(Ib, Ib, tensor(α_arg, H2 - E_0).expm(), I2, I2, Ib)) \
            .permute([0, 1, 2, 7, 5, 6, 3, 4]) * \
           (tensor(Ib, Ib, Ib, tensor(α_arg, H2 - E_0).expm(), I2, I2)) \
            .permute([0, 1, 2, 3, 7, 6, 4, 5])

def D12_C(α, H2, E_0, cutoff):
    
    '''
    operator that displaces modes 1 & 2 by weight of C = H34
    '''
    
    # define bosonic operators 
    a = destroy(cutoff)
    Ib = qeye(cutoff)
    
    # define argument of unconditional displacement operator
    α_arg = α*a.dag() - np.conj(α)*a
    
    return (tensor(tensor(α_arg, H2 - E_0).expm(), Ib, Ib, I2, I2, Ib)) \
            .permute([0, 7, 3, 4, 5, 6, 1, 2]) * \
           (tensor(Ib, tensor(α_arg, H2 - E_0).expm(), Ib, I2, I2, Ib)) \
            .permute([0, 1, 7, 4, 5, 6, 2, 3])

def D41_D(α, H2, E_0, cutoff): 
    
    '''
    operator that displaces modes 4 & 1 by weight of D = H41
    '''
    
    # define bosonic operators 
    a = destroy(cutoff)
    Ib = qeye(cutoff)
    
    # define argument of unconditional displacement operator
    α_arg = α*a.dag() - np.conj(α)*a
    
    return (tensor(Ib, Ib, Ib, tensor(α_arg, H2 - E_0).expm(), I2, I2)) \
            .permute([0, 1, 2, 3, 4, 7, 6, 5]) * \
           (tensor(tensor(α_arg, H2 - E_0).expm(), Ib, Ib, Ib, I2, I2)) \
            .permute([0, 3, 4, 5, 1, 6, 7, 2])

def D23_D(α, H2, E_0, cutoff):
    
    '''
    operator that displaces modes 2 & 3 by weight of D = H41
    '''
    
    # define bosonic operators 
    a = destroy(cutoff)
    Ib = qeye(cutoff)
    
    # define argument of unconditional displacement operator
    α_arg = α*a.dag() - np.conj(α)*a
    
    return (tensor(Ib, tensor(α_arg, H2 - E_0).expm(), Ib, Ib, I2, I2)) \
            .permute([0, 1, 4, 5, 2, 6, 7, 3]) * \
           (tensor(Ib, Ib, tensor(α_arg, H2 - E_0).expm(), Ib, I2, I2)) \
            .permute([0, 1, 2, 5, 3, 6, 7, 4])


def GSP(α, H2, E_0, cutoff): 
    
    '''
    return ground state projection operator for N = 4 sites
    '''
    
    return D23_D(α, H2, E_0, cutoff) * D41_B(α, H2, E_0, cutoff) * \
           D12_C(α, H2, E_0, cutoff) * D34_A(α, H2, E_0, cutoff) * \
           D41_D(α, H2, E_0, cutoff) * D23_B(α, H2, E_0, cutoff) * \
           D34_C(α, H2, E_0, cutoff) * D12_A(α, H2, E_0, cutoff)



