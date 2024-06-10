# import relevant modules and functions
import numpy as np
import matplotlib.pyplot as plt
import os
from qutip import Qobj
from scipy.linalg import logm
from simulate_algorithm import format_number
from generate_VTA_list import exact, SU2_automated

def spectrum(operator_list, Jx, Jy, Jz, \
             α_start, α_end, α_steps, Es, \
             k_start, k_end, k_steps, \
             tolerance, asymptotes, \
             current_directory, \
             loc = 'upper right', \
             VTA_type = 'approximate', \
             log = True, fontsize = 9, ncol = 3, \
             savefile = False): 

    # compute array over which we will sweep α
    α_array = np.linspace(α_start, α_end, α_steps)

    # create list for rank ordering list of eigenvalues
    # according to size of modulus
    k_list = list(range(k_start, k_end + 1, k_steps))
    k_length = len(k_list)

    # compute λ if Jx == Jy
    if Jx == Jy: 
        λ = Jz/Jx

    # initialize lists of eigenvalues as well as
    # real and imaginary parts
    eigvals = []

    # find the list of largest eigenvalues for each operator
    for O in operator_list: 

        eigvals_array = np.array(sorted(Qobj(O).eigenenergies(), reverse = True, \
                           key=lambda x: abs(x))[k_start:k_end + 1: k_steps])

        eigvals.append(eigvals_array)
        
    # convert 'eigvals' into eigvals_array
    eigvals_array = np.array(eigvals)

    # find arrays of real and imaginary parts of eigenvalues
    real_eigvals_array = eigvals_array.real
    imag_eigvals_array = eigvals_array.imag

    # find lowest eigenvalue across all α for each eigenvalue 
    # category, i.e., first largest, second largest, etc. 
    asymptotes_list = []
    for i in range(k_length):
        asymptotes_list.append(real_eigvals_array[:, i][-1])

    # construct array of thresholds for plotting
    asymptotes_array = np.tile(asymptotes_list, reps = α_steps).reshape(α_steps,  k_length)

    # Generate unique colors to use for plotting
    colors = ['red', 'green', 'lightblue', 'orange', 'purple', \
              'magenta', 'pink', 'gray', 'cyan', 'coral', \
              'black', 'silver', 'olive', 'teal', 'maroon', 'blue']

    fig, ax = plt.subplots(figsize=(8, 6))
    
    if VTA_type == 'exact' and log == True:
        title_str = r'Spectrum of $ \log(\mathrm{{VTA}}_{\mathrm{exact}} $' \
                    + fr'$(\alpha, \lambda = {format_number(λ)},$ ' \
                    + fr'$ Es = {format_number(Es)}))$'
        
    elif VTA_type == 'exact': 
        title_str = r'Spectrum of $ \mathrm{{VTA}}_{\mathrm{exact}} $' \
                    + fr'$(\alpha, \lambda = {format_number(λ)},$ ' \
                    + fr'$ Es = {format_number(Es)})$'
    elif VTA_type == 'approximate' and log == True: 
        title_str = r' Spectrum of $ \log(\mathrm{{VTA}} $' \
                    + fr'$(\alpha, \lambda = {format_number(λ)}$, ' \
                    + fr'$Es = {format_number(Es)}))$'
    elif VTA_type == 'approximate': 
        title_str = r' Spectrum of $ \mathrm{{VTA}} $' \
                    + fr'$(\alpha, \lambda = {format_number(λ)}$, ' \
                    + fr'$Es = {format_number(Es)})$'

    for i, j in enumerate(k_list): 

        # plot real part of eigenvalue
        ax.plot(α_array, real_eigvals_array[:, i], \
                color = colors[j], \
                label = f'$\mathcal{{Re}}(\gamma_{{{j}}})$')

        # plot imaginary parts 
        if any(imag_eigvals_array[:, i] > tolerance): 
            ax.plot(α_array, imag_eigvals_array[:, i], \
                color = colors[j], \
                label = f'$\mathcal{{Im}}(\gamma_{{{j}}})$', 
                linestyle = '--')
        
        # plots asymptotes
        if asymptotes and any(asymptotes_array[:, i] > tolerance): 
            ax.plot(α_array, asymptotes_array[:, i], \
                    color = colors[j % len(colors)], \
                    label = r'$\lim_{{\alpha \to \infty}}$ ' \
                    + fr'$\gamma_{{{j}}} = {asymptotes_array[0, i] : .3f}$', \
                    linestyle = ':')
        ax.set_xlabel(r'$\alpha$', fontsize=15)
        ax.set_title(title_str, fontsize=17.5)
    ax.legend(fontsize = fontsize, ncol=ncol, loc = loc)
    ax.grid(True)
    
    if savefile: 
        
        if VTA_type == 'exact' and log == True:
            sub_folder = 'log_VTA_exact'
        elif VTA_type == 'exact' and log == False: 
            sub_folder = 'VTA_exact'
        elif VTA_type == 'approximate' and log == True: 
            sub_folder = 'log_VTA'
        elif VTA_type == 'approximate' and log == False: 
            sub_folder = 'VTA'
        else: 
            raise ValueError('Please enter a valid VTA_type and log type')

        # create name for png file 
        parent_dir = f'{current_directory}/data/λ={round(λ, 3)}/{sub_folder}'

        # Check if the directory already exists
        if not os.path.exists(parent_dir):
            # Create the directory
            os.makedirs(parent_dir)

        # save figure under custom filename
        filename = parent_dir + f'/E0={round(Es, 3)}_{k_start}_{k_end}_{k_steps}.png'
        plt.savefig(filename)

        
def sweep_spectra(N_qubits, Jx, Jy, Jz, periodic_bc, \
                     M, α_start, α_end, α_steps, \
                     Es_start, Es_end, Es_gradation, \
                     k_start, k_end, k_steps, \
                     tolerance, asymptotes, \
                     current_directory, \
                     VTA_type = 'approximate', log = True, \
                     fontsize = 9, ncol = 3, loc = 'upper right', \
                     savefile = False, return_op_list = True):
    
    '''
    plots the spectrum of VTA (or VTA_exact) or the spectrum of
    log of VTA (or VTA_exact) for a sweep over Es
    '''

    # generate list over which we will sweep Es
    if Es_start <= Es_end: 
        Es_list = np.arange(Es_start, Es_end + Es_gradation, Es_gradation)
    else: 
        raise ValueError('Please enter valid values of Es_start and Es_end')
        
    # create list to fill with operators
    op_list = []

    if VTA_type == 'exact' and log == True: 

        for Es in Es_list: 

            # compute list of VTA_exact
            VTA_exact_list = exact(N_qubits, Jx, Jy, Jz, periodic_bc, \
                                   M, α_start, α_end, α_steps, Es)

            # compute list of effective Hamiltonians
            H_eff_exact_list = [logm(VTA.full()) for VTA in VTA_exact_list]
               
            
            # append H_eff_exact_list to 
            op_list.append(H_eff_exact_list)

            # compute spectrum for each H_eff_exact
            spectrum(H_eff_exact_list, Jx, Jy, Jz, \
                     α_start, α_end, α_steps, Es, \
                     k_start, k_end, k_steps, \
                     tolerance, asymptotes, \
                     current_directory, \
                     VTA_type = VTA_type, log = log, \
                     fontsize = fontsize, ncol = ncol, loc = loc, \
                     savefile = savefile)

    if VTA_type == 'exact' and log == False: 

        for Es in Es_list: 

            # compute list of VTA_exact
            VTA_exact_list = exact(N_qubits, Jx, Jy, Jz, periodic_bc, \
                                   M, α_start, α_end, α_steps, Es)
            
            # append VTA_exact_list to 
            op_list.append(VTA_exact_list)

            # compute spectrum for each VTA_exact
            spectrum(VTA_exact_list, Jx, Jy, Jz, \
                     α_start, α_end, α_steps, Es, \
                     k_start, k_end, k_steps, \
                     tolerance, asymptotes, \
                     current_directory, \
                     VTA_type = VTA_type, log = log, \
                     fontsize = fontsize, ncol = ncol, loc = loc, \
                     savefile = savefile)

    elif VTA_type == 'approximate' and log == True:

        for Es in Es_list:

            # compute list of VTAs
            VTA_list = SU2_automated(N_qubits, α_start, α_end, α_steps, \
                                     Jx, Jy, Jz, Es)

            # compute list of effective Hamiltonians
            H_eff_list = [logm(VTA.full()) for VTA in VTA_list]
            
            # append H_eff_list to 
            op_list.append(H_eff_list)

            # plot spectrum of each H_eff
            spectrum(H_eff_list, Jx, Jy, Jz, \
                     α_start, α_end, α_steps, Es, \
                     k_start, k_end, k_steps, \
                     tolerance, asymptotes, \
                     current_directory, \
                     VTA_type = VTA_type, log = log, \
                     fontsize = fontsize, ncol = ncol, loc = loc, \
                     savefile = savefile)

    elif VTA_type == 'approximate' and log == False:

        for Es in Es_list:

            # compute list of VTAs
            VTA_list = SU2_automated(N_qubits, α_start, α_end, α_steps, \
                                     Jx, Jy, Jz, Es)
            
            # append H_eff_exact_list to 
            op_list.append(VTA_list)

            # plot spectrum of each H_eff
            spectrum(VTA_list, Jx, Jy, Jz, \
                     α_start, α_end, α_steps, Es, \
                     k_start, k_end, k_steps, \
                     tolerance, asymptotes, \
                     current_directory, \
                     VTA_type = VTA_type, log = log, \
                     fontsize = fontsize, ncol = ncol, loc = loc, \
                     savefile = savefile)

    if return_op_list: 
        return op_list[0]
            