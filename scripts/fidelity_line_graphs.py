import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from qutip import tensor, basis, rand_ket, fidelity, expect
from spin_chain import properties
from generate_VTA_list import generate_SWAP_operators

# font specs for plots
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.size"] = "14"

def generate_initial_states(N_cavities, N_qubits, num_states, cutoff):
    
    '''
    generate list of initial states which is a product of state of 
    the number of cavities and qubits specified by the arguments
    '''
    
    # compute list of all initial qubit states 
    ψ0_list = [tensor([rand_ket(2) \
               for i in range(N_qubits)]) \
               for initial_state in range(num_states)]
    
    if cutoff == 'N/A': 
        
        # return only the list of qubit states that are 
        # not entangled with qumodes
        return None, ψ0_list
   
    else: 
        
        # compute the vacuum state of a qumode
        vacuum_state = tensor([basis(cutoff, 0)]*N_cavities)
    
        # compute list of all initial qubit states with entangled 
        # register of qumodes in vacuum
        Ψ0_list = [tensor(vacuum_state, ψ0) for ψ0 in ψ0_list]
        return Ψ0_list, ψ0_list 
        
        
def VTA_recursive(VTA, ψ, r): 
    
    '''
    function to apply VTA algorithm r times 
    '''
    
    for _ in range(r):
        ψ = (VTA*ψ).unit()
    return ψ

def format_number(num):
    
    '''
    function to format λ_XXZ and Es when plotting so they 
    have 0 decimals if integers and 2 decimals otherwise
    '''
    # round number to 9 decimals 
    num = round(num, 6)
    
    if isinstance(round(num, 6), int) or num.is_integer():
        return f"{int(num)}"
    else: 
        return f"{num:.2f}"
        
def plot_exact_fidelity(N_cavities, α_start, α_end, α_steps, Es, m_start, m_stop, 
                        N_qubits, Jx, Jy, Jz, periodic_bc, tolerance, ψ0_list, \
                        savefile, directory):

    # list of numbers of cavities we will project onto the ground state
    if m_start >= 1 and m_stop <= N_qubits and m_stop <= N_cavities: 
        cavity_list = list(range(m_start, m_stop+1))
    else: 
        raise ValueError("Please enter valid list of the number of cavities to project onto vacuum")

    # create array over which to sweep α
    α_array = np.linspace(α_start, α_end, α_steps)
    
    # compute λ if Jx == Jy
    if Jx == Jy: 
        λ_XXZ = Jz/Jx
        
    # find eigenstates, eigenvalues and ground state energy from spin chain 
    _, _, eigenstates, eigenvalues, E0, _ = properties(N_qubits, Jx, Jy, Jz, periodic_bc, tolerance)

    # calcuate the amount each state will be displaced 
    shifted_eigenvalues = eigenvalues - Es
    
    # find the probability amplitudes for projecting onto each eigenstate in our eigenbasis for every initial state
    prob_amps_list = [[fidelity(ψ0, eigenstate)**2 for eigenstate in eigenstates] \
                      for ψ0 in ψ0_list]
    
    # find indices of eigenvalues corresponding to ground state (or groundspace)
    indices = [index for index, value in enumerate(eigenvalues) \
               if np.allclose(value, E0, tolerance)]

    # collect the probability amplitudes associated with projecting onto the eigenstates in the groundstate 
    # (or groundspace) for every initial state
    gs_prob_amps_list = [[prob_amps[i] for i in indices] for prob_amps in prob_amps_list]

    # calcuate the success probability of being in the ground state after projecting onto vacuum
    fidelity_array_list = [[[np.exp(-m*(α**2)*((E0 - Es)**2))*sum(gs_prob_amps)/(sum([np.exp(-(m*(λ*α)**2))*prob_amp \
                           for λ, prob_amp in zip(shifted_eigenvalues, prob_amps)])) \
                           for α in α_array] \
                           for m in cavity_list]
                           for gs_prob_amps, prob_amps in zip(gs_prob_amps_list, prob_amps_list)] 

    # Generate unique colors using the 'tab10' color map
    colors = ['blue', 'orange', 'green','red', 'purple', 'black', 'pink', 'gray', 'cyan', 'magenta']
        
    # generate custom linestyles to use if we are projecting 4 or more cavities onto vacuum
    dashed_list = [(None, None), (5, 2), (1, 2), (5, 2, 1, 2), (10, 5, 2, 5), \
                   (2, 2, 10, 2), (5, 2, 10, 2, 5, 2), (15, 5, 5, 5), (2, 5, 10, 5), \
                   (0, 3, 5, 1, 5), (0, 3, 1, 1, 1), (0, 1, 1)]
    
    # create plot with axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # graph the theoretical projection of all initial states onto the true ground state 
    for i in range(len(fidelity_array_list)):
        for j, fidelity_array in enumerate(fidelity_array_list[i]):
            ax.plot(α_array, fidelity_array, \
                    label=f'M =  {cavity_list[j]}, $F_0$ = {fidelity_array[0] :.3f}', \
                    color=colors[i], linestyle='-', dashes = dashed_list[(len(cavity_list) - 1) - j])

    ax.set_xlabel(r'$\alpha$', fontsize=15)
    ax.set_ylabel('Fidelity', fontsize=15)
    ax.set_title(fr'$ \mathrm{{VTA}}_{{\mathrm{{exact}}}}(\alpha, $'
                 fr'$ \lambda = {format_number(λ_XXZ)}, E_s = {format_number(Es)}) $ ' \
                 f'for N = {N_qubits}', fontsize=17.5) 
    ax.legend(fontsize = 12.5)
    ax.grid(True)
        
    if savefile: 
        
        # create parent directory for file storage
        parent_dir = f'{directory}/data/{N_qubits}_sites' \
                     + f'/λ={round(λ_XXZ, 3)}/fidelity_line_graphs/exact' 

        # Check if the directory already exists
        if not os.path.exists(parent_dir):
            # Create the directory
            os.makedirs(parent_dir)
            
        # calculate number of states that were simulated
        num_states = len(ψ0_list)
    
        if m_start != m_stop: 

            # create name for png file 
            plot_params = f'/{num_states}_{m_start}_{m_stop}_{round(Es, 2)}.png'

            # save figure under custom filename
            filename = parent_dir + plot_params
            plt.savefig(filename)

        elif m_start == m_stop: 
            
            # create name for png file 
            plot_params = f'/{num_states}_{m_start}_{round(Es, 2)}.png'

            # save figure under custom filename
            filename = parent_dir + plot_params
            plt.savefig(filename)
        
    return fidelity_array_list
        
        
def VTA_fidelity(N, ψ0_list, VTA_list, ρ_ground_stateN, α_start, α_end, α_steps, 
                 Jx, Jy, Jz, Es, r, m_start, m_stop, asymptotes, savefile, directory, \
                 return_fidelity_array = False): 
    
    # define array over which we will plot α
    α_array = np.linspace(α_start, α_end, α_steps)
    
    # compute λ if Jx == Jy
    if Jx == Jy: 
        λ_XXZ = Jz/Jx

    # define list of colors to use for plots 
    colors = ['blue', 'orange', 'green','red', 'purple', 'black', 'pink', 'gray', 'cyan', 'magenta']

    # compute list of projection fidelity for VTA_list for all initial states in ψ0_list
    VTA_fidelity_array_list = [[expect(ρ_ground_stateN, VTA_recursive(VTA, ψ0, r))  \
                                for VTA in VTA_list] \
                                for ψ0 in ψ0_list]

    # generate custom linestyles to use if we are projecting 4 or more cavities onto vacuum
    dashed_list = [(None, None), (5, 2), (1, 2), (5, 2, 1, 2), (10, 5, 2, 5), \
                   (2, 2, 10, 2), (5, 2, 10, 2, 5, 2), (15, 5, 5, 5), (2, 5, 10, 5)]

    fig, ax = plt.subplots(figsize=(8, 6))

    if asymptotes:
        
        π_list = generate_SWAP_operators(4, Jx, Jz, Jz)
        
        # define projection operators where π_kl is a tuple 
        # such that π_kl[0] = π^{+}_{kl} and π_kl[1] = π^{-}_{kl}
        π12 = π_list[0]
        π23 = π_list[1]
        π34 = π_list[2]
        π41 = π_list[3]

        # GSP operator for large α
        G = π41[0]*π23[0]*π34[1]*π12[1]*π41[1]*π23[1]*π34[1]*π12[1] + \
            π41[1]*π23[1]*π34[0]*π12[0]*π41[1]*π23[1]*π34[1]*π12[1] + \
            π41[1]*π23[1]*π34[1]*π12[1]*π41[0]*π23[0]*π34[1]*π12[1] + \
            π41[1]*π23[1]*π34[1]*π12[1]*π41[1]*π23[1]*π34[0]*π12[0] 

        # compute thresholds at which the fidelity will saturate
        threshold_array_list = [[expect(ρ_ground_stateN, (G*ψ0).unit())]*(α_steps) \
                                 for ψ0 in ψ0_list]

        # graph the theoretical projection of all initial states onto the true ground state 
        for i, (VTA_list, threshold_list) in enumerate(zip(VTA_fidelity_array_list, \
                                                           threshold_array_list)): 
            ax.plot(α_array, VTA_list, \
                    label=f'$ | \psi_{i} \\rangle, F_0$ = {VTA_list[0] : .3f}', \
                    color=colors[i], \
                    linestyle='-')
            ax.plot(α_array, threshold_list, \
                    label = f'$\mathcal{{G}}_{i} = {threshold_list[i]:.3f}$', \
                    color = colors[i], \
                    linestyle = '--')

    else: 
        # graph the theoretical projection of all initial states onto the true ground state 
        for i, VTA_list in enumerate(VTA_fidelity_array_list): 
            ax.plot(α_array, VTA_list, \
                    label=f'M = {N}, $F_0$ = {VTA_list[0] : .3f}', \
                    color=colors[i], \
                    linestyle='-')

    ax.set_xlabel(r'$\alpha$', fontsize=15)
    ax.set_ylabel('Fidelity', fontsize=15)
    ax.set_title(fr'$ \mathrm{{VTA}}(\alpha, \lambda = {format_number(λ_XXZ)}, $ ' \
                 fr'$ E_s = {format_number(Es)})$ with r = {r} for N = {N}', fontsize=17.5)  
    ax.legend(fontsize = 12.5)
    ax.grid(True)

    if savefile: 

        # calculate number of states that were simulated
        num_states = len(ψ0_list)

        # create parent directory for file storage
        parent_dir = f'{directory}/data/{N}_sites/λ={round(λ_XXZ, 3)}' \
                     + '/fidelity_line_graphs/approximate' 
        
        # Check if the directory already exists
        if not os.path.exists(parent_dir):
            # Create the directory
            os.makedirs(parent_dir)
            
        # define parameters for filename
        plot_params = f'/{α_start}_{α_end}_{α_steps}_{round(Es, 2)}_{r}.png'

        # save figure under custom filename
        filename = parent_dir + plot_params
        plt.savefig(filename)
        
    if return_fidelity_array:
        return VTA_fidelity_array_list
    else: 
        return None
 
    
def plot_VTA_fidelity(ψ0_list, VTA_list, ρ_ground_stateN, α_start, α_end, α_steps, Jx, Jy, Jz, Es, \
                      r_start, r_end, r_steps, m_start, m_stop, asymptotes, savefile, directory, \
                      return_fidelity_array = False): 
    
    'plots VTA fidelity for a specified amount of r iterations'

    # create a list for the number of times we will apply VTA
    r_list = list(range(r_start, r_end + 1, r_steps))

    # plot simulations
    for r in r_list:

        if r == 1 and asymptotes: 
            VTA_fidelity(ψ0_list, VTA_list, ρ_ground_stateN, \
                         α_start, α_end, α_steps, \
                         Jx, Jy, Jz, Es, r, m_start, m_stop,\
                         asymptotes, savefile, directory, return_fidelity_array = \
                         return_fidelity_array)
        else: 
            more_asymptotes = False
            VTA_fidelity(ψ0_list, VTA_list, ρ_ground_stateN, \
                         α_start, α_end, α_steps, \
                         Jx, Jy, Jz, Es, r, m_start, m_stop,\
                         more_asymptotes, savefile, directory, return_fidelity_array = \
                         return_fidelity_array)
    
