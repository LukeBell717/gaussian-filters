import os
import numpy as np
import matplotlib.pyplot as plt
from qutip import expect
from fidelity_line_graphs import format_number

# G(α, Es) basis functions
e0 = lambda α, Es: np.exp(-2*(α**2)*((Es + 12)**2)) 
e1 = lambda α, Es: np.exp(-2*(α**2)*((Es + 8)**2)) 
e2 = lambda α, Es: np.exp(-2*(α**2)*((Es + 4)**2)) 
e3 = lambda α, Es: np.exp(-2*(α**2)*(Es**2)) 
e4 = lambda α, Es: np.exp(-2*(α**2)*((Es - 4)**2))
e5 = lambda α, Es: np.exp(-2*(α**2)*((Es + 2)**2 + 4))
e6 = lambda α, Es: np.exp(-2*(α**2)*((Es + 2)**2 + 20))
e7 = lambda α, Es: np.exp(-2*(α**2)*(Es**2 + 8))
e8 = lambda α, Es: np.exp(-2*(α**2)*((Es + 4)**2 + 16))
e9 = lambda α, Es: np.exp(-2*(α**2)*(Es**2 + 16))
e10= lambda α, Es: np.exp(-2*(α**2)*((Es + 4)**2 + 32))

# G(α, Es) coefficients
g00 = lambda α, Es: (3*e0(α, Es) + 24*e1(α, Es) \
                     + 6*e2(α, Es) - e4(α, Es))/32
g44 = lambda α, Es: -(e0(α, Es) - 6*e2(α, Es) \
                      - 24*e3(α, Es) - 3*e4(α, Es))/32
g04 = lambda α, Es: (np.sqrt(3)*(e0(α, Es) + 4*e1(α, Es) - \
                     10*e2(α, Es) + 4*e3(α, Es) + e4(α, Es)))/32
g40 = lambda α, Es: - g04(α, Es)
gt = lambda α, Es: (e10(α, Es) + e2(α, Es) + 4*e5(α, Es) - 4*e6(α, Es) + \
                    2*e8(α, Es))/4
gs = lambda α, Es: (e3(α, Es) + e5(α, Es) - e6(α, Es) + e9(α, Es))/2
gq = lambda α, Es: e4(α, Es)

def VTA_exact(eigvecs, ρ_list, ρ_ground_stateN, ψ, α, Es, r): 

    '''
    compute single VTA_exact for a given α, Es, and r
    '''
    
    # analytic expression for the coefficients of F(α; Es)
    f = lambda x, α: np.exp(-32*(α**2)*(x - 1 - Es/4)**2)

    # compute F
    F = (f(-1, α)*ρ_list[0] + \
         f(0, α)*sum(ρ_list[1:3 + 1]) + \
         f(1, α)*sum(ρ_list[4:10 + 1]) + \
         f(2, α)*sum(ρ_list[11:15 + 1])).tidyup(atol = 1e-9)
    
    # iterate algorithm r times 
    for _ in range(r):
        ψ = (F*ψ).unit()

    # return projection fidelity of algorithm 
    return expect(ρ_ground_stateN, ψ)

def VTA(eigvecs, ρ_list, ρ_ground_stateN, ψ, α, Es, r): 

    '''
    compute single VTA for a given α, Es, and r
    '''

    # define operator that acts on the logical qubit subspace
    GL_op = lambda α, Es: g00(α, Es)*ρ_list[0] + \
                          g04(α, Es)*eigvecs[0]*eigvecs[4].dag() + \
                          g40(α, Es)*eigvecs[4]*eigvecs[0].dag() + \
                          g44(α, Es)*ρ_list[4]

    # compute G
    G = (GL_op(α, Es) + \
        gt(α, Es)*sum(ρ_list[1:4]) + \
        gs(α, Es)*sum(ρ_list[5:11]) + \
        gq(α, Es)*sum(ρ_list[11:16])).tidyup(atol = 1e-9)
    
    # iterate algorithm r times 
    for _ in range(r):
        ψ = (G*ψ).unit()

    # return projection fidelity of algorithm 
    return expect(ρ_ground_stateN, ψ)

def plot_sweep(N_qubits, VTA_type, eigvecs, ρ_list, ρ_ground_stateN, \
               ψ0, Jx, Jy, Jz, α_start, α_end, α_steps, \
               Es_start, Es_end, Es_steps, r_start, r_end, \
               savefile = False, directory = None): 
    
    '''
    plots heatmap for sweep over α vs Es or r
    '''
    
    # define λ
    if Jx == Jy: 
        λ = Jz/Jx
    else: 
        raise ValueError('Please enter valid Jx and Jy')
    
    # calculate initial fidelity
    F0 = expect(ρ_ground_stateN, ψ0)
    
    # examine parameters for Es sweep and r sweep
    Es_params = len(set([Es_start, Es_end]))
    r_params = len(set([r_start, r_end]))
    
    if Es_params != 1 and r_params == 1: 
        
        # define r value
        r = r_start

        # construct α_array and Es_array
        α_array = np.linspace(α_start, α_end, α_steps)
        Es_array = np.linspace(Es_start, Es_end, Es_steps)

        # create grids from α_array and Es_array
        α_grid, Es_grid = np.meshgrid(α_array, Es_array)

        # compute data for sweep over α and Es
        fidelity_values = np.zeros((len(Es_array), len(α_array)))
        
        # calculate projection fidelities using either VTA_exact or VTA
        if VTA_type == 'exact': 
        
            for i, Es, in enumerate(Es_array):
                for j, α in enumerate(α_array): 
                    fidelity_values[i, j] = VTA_exact(eigvecs, ρ_list, ρ_ground_stateN, ψ0, α, Es, r)
                    
            # define title string
            title_str = fr'$\mathrm{{VTA}}_{{\mathrm{{exact}}}}(\alpha, \lambda = {format_number(Jz/Jy)}, E_s)$ ' \
                 fr'with r = {r} and $F_0 = {round(F0, 3)}$'
                    
        elif VTA_type == 'approximate':
            
            for i, Es, in enumerate(Es_array):
                for j, α in enumerate(α_array): 
                    fidelity_values[i, j] = VTA(eigvecs, ρ_list, ρ_ground_stateN, ψ0, α, Es, r)
                    
            # define title string
            title_str = fr'$\mathrm{{VTA}}(\alpha, \lambda = {format_number(Jz/Jy)}, E_s)$ ' \
                 fr'with r = {r} and $F_0 = {round(F0, 3)}$'
            
        else: 
            raise ValueError('Please enter valid VTA_type')
                
        # define window parameters for plot
        extent = [α_start, α_end, Es_start, Es_end]
        
        # define y axis
        ylabel_str = r'$E_s$'
        
        # name used to store file 
        sweep_type = 'Es_sweeps'
        
    elif r_params != 1 and Es_params == 1:
        
        # define Es value
        Es = Es_start

        # construct α_array and Es_array
        α_array = np.linspace(α_start, α_end, α_steps)
        r_array = list(range(r_start, r_end+1))

        # create grids from α_array and Es_array
        α_grid, r_grid = np.meshgrid(α_array, r_array)

        # compute data for sweep over α and Es
        fidelity_values = np.zeros((len(r_array), len(α_array)))
        
        # calculate projection fidelities using either VTA_exact or VTA
        if VTA_type == 'exact': 
        
            for i, r, in enumerate(r_array):
                for j, α in enumerate(α_array): 
                    fidelity_values[i, j] = VTA_exact(eigvecs, ρ_list, ρ_ground_stateN, ψ0, α, Es, r)
                    
            # define title string
            title_str = r'$\mathrm{{VTA}}_{{\mathrm{{exact}}}}(\alpha, ' \
                        + fr'\lambda = {format_number(Jz/Jy)}, E_s = {Es})$' \
                        +  fr' for $F_0 = {round(F0, 3)}$'
            
            # name used to store file 
            sweep_type = 'r_sweeps'

        elif VTA_type == 'approximate':
            
            for i, r, in enumerate(r_array):
                for j, α in enumerate(α_array): 
                    fidelity_values[i, j] = VTA(eigvecs, ρ_list, ρ_ground_stateN, ψ0, α, Es, r)
                    
            # define title string
            title_str = r'$\mathrm{{VTA}}(\alpha, ' \
                        + fr'\lambda = {format_number(Jz/Jy)}, E_s = {Es})$' \
                        +  fr' for $F_0 = {round(F0, 3)}$'
            
        else: 
            raise ValueError('Please enter valid VTA_type')
                
        # define window parameters for plot
        extent = [α_start, α_end, r_start, r_end]
        
        # define y axis
        ylabel_str = 'r'
    
    else: 
        raise ValueError('Please enter valid parameters for sweep over Es or r')
        
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the heatmap
    cax = ax.imshow(fidelity_values, extent=extent, \
                    origin='lower', cmap='coolwarm', aspect='auto')

    # Add a colorbar
    cbar = fig.colorbar(cax, ax=ax, label='Fidelity')

    # Set title and labels
    ax.set_xlabel('α')
    ax.set_ylabel(ylabel_str)
    ax.set_title(title_str)
    
    if savefile: 
        
        if r_params == 1: 
            
            # name used to store file 
            sweep_type = 'Es_sweeps'
            
            # fixed parameter
            fixed_param = f'r={r}'
            
            # figure information 
            plot_params = f'/α={α_start}_{α_end}_{α_steps}_Es={Es_start}_{Es_end}_{Es_steps}.png'
            
        elif Es_params == 1: 
            
            # name used to store file 
            sweep_type = 'r_sweeps'
            
            # fixed parameter
            fixed_param = f'Es={round(Es, 3)}'
            
            # figure information 
            plot_params = f'/α={α_start}_{α_end}_{α_steps}_r={r_start}_{r_end}.png'

        # create name for png file 
        parent_dir = f'{directory}/{N_qubits}_sites/λ={round(λ, 3)}' \
                     f'/fidelity_heatmaps/{sweep_type}/{VTA_type}_{fixed_param}'

        # Check if the directory already exists
        if not os.path.exists(parent_dir):
            # Create the directory
            os.makedirs(parent_dir)

        # save figure under custom filename
        filename = parent_dir + plot_params
        plt.savefig(filename)