# import relevant modules 
import os
import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, expect
from scipy.linalg import logm
from functools import reduce
from collections import Counter
from spin_chain import properties, collect_spin_ops, symmetry_eigvecs
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

def sort_eigvals(ranking, eigvals, k_start, k_end, k_steps): 
    
    '''
    returns eigenvalues depending on method of ranking
    '''
    
    if ranking == 'energy' or ranking == 'spin': 
        
        return eigvals
    
    elif ranking == 'modulus': 
        
        # define key that sorts according to modulus 
        key = lambda x: abs(x)
        
        return np.array(sorted(eigvals, reverse = True, key=key) \
                        [k_start:k_end + 1: k_steps])
    
def find_s(γ, tolerance = 1e-9):
    
    '''
    function that will find s eigval from 
    total angular momentum eigenvalue γ = s(s+1)
    '''
    
    # compute s and round to nearest integer
    s = (-1 + np.sqrt(1 + 4*γ))/2
    int_s = np.round(s, decimals = abs(np.log10(tolerance)).astype(int))
    
    # return s if its is sufficiently close to an integer
    if np.all(abs(s - int_s)) < tolerance: 
        return int_s
    else: 
        raise ValueError('s is not an integer')


def plot_spectra(VTA_type, log, N, Jx, Jy, Jz, periodic_bc, \
                 α_start, α_end, α_steps, Es, \
                 ranking, k_start, k_end, k_steps, asymptotes, \
                 tolerance, directory, r = 1, Δ = 0, fontsize = 9, \
                 ncol = 4, loc = 'best', loc_spin = None, \
                 return_op_list = False, savefile = False): 

    '''
    plot analytic expression for VTA_exact or VTA 
    as a function of α and Es
    '''

    # define λ
    if Jx == Jy: 
        λ = Jz/Jx
    else: 
        raise ValueError('Please enter valid Jx and Jy')

    # compute array over which we will sweep α
    α_array = np.linspace(α_start, α_end, α_steps)

    # create list of eigenvalues to plot
    k_list = list(range(k_start, k_end + 1, k_steps))


    if ranking == 'spin' or Δ != 0: 

        # compute simulatenous eigenvectors H4, S2, Sz, and σ24
        H4, S2, Sz, σ24, P, eigvecs, ρ_list = symmetry_eigvecs(N, Jx, Jy, Jz, \
                                                           periodic_bc, return_operators = True)

        # transform S2, Sz, and σ24 into correct bases
        S2_P = S2.transform(P.dag()).tidyup(atol = tolerance)
        Sz_P = Sz.transform(P.dag()).tidyup(atol = tolerance)
        σ24_P = σ24.transform(P.dag()).tidyup(atol = tolerance)

        # construct perturbation 
        V = Δ*S2

    else:
        P, eigvecs, ρ_list = symmetry_eigvecs(N, Jx, Jy, Jz, periodic_bc)

        V = 0

    # initialize empty list in which to store spectra 
    # and eigenvectors of for F (G) or log(F) (log(G))
    op_list = []
    spectra_list = []
    eigvecs_list = []

    if VTA_type == 'exact':

        if log == True: 

            # analytic expression for the coefficients of log(F(α; Es))
            f = lambda x, α: (-32*(α**2)*(x - 1 - Es/4)**2)

            # define title for graph
            title_str = fr'Spectrum of $\log(\mathcal{{F}}(\alpha, E_s = {format_number(Es)}))$'

            # define subfolder 
            sub_folder = 'log_F'

        elif log == False: 

            # analytic expression for the coefficients of F(α; Es)
            f = lambda x, α: np.exp(-32*(α**2)*(x - 1 - Es/4)**2)

            # define title for graph
            title_str = fr'Spectrum of $\mathcal{{F}}(\alpha, E_s = {format_number(Es)})$' 

            # define subfolder
            sub_folder = 'F'

        for α in α_array: 

            # generate array of coefficients to compute F(α; Es) or log(F(α; Es))
            coeff_array = [f(-1, α)] + [f(0, α)]*3 + \
                          [f(1, α)]*7 + [f(2, α)]*5

            # compute F(α; Es) or log(F(α; Es))
            F_op = (sum(f*ρ for f, ρ in zip(coeff_array, ρ_list)) + V)**r
            
            # transform F_op to P basis
            F_op_P = F_op.transform(P.dag()).tidyup(atol = tolerance)

            # append F_op to F_list
            op_list.append(F_op)
            
            # find the eigenvalues and eigenvectors of F_op
            F_op_P_eigvals, F_op_P_eigvecs = F_op_P.eigenstates()
            
            # compute spectrum of F_op
            F_op_P_eigvals_sorted = sort_eigvals(ranking, F_op_P_eigvals, k_start, k_end, k_steps)

            # append spectrum to list 
            spectra_list.append(F_op_P_eigvals_sorted)

            # append eigenvectors to list
            eigvecs_list.append(F_op_P_eigvecs)

    elif VTA_type == 'approximate': 

        if log == True: 

            for α in α_array: 

                # compute coefficients of log(GL)
                log_GL = lambda α, Es: logm([[g00(α, Es), g04(α, Es)], [g40(α, Es), g44(α, Es)]])
                log_GL00 = lambda α, Es: log_GL(α, Es)[0, 0]
                log_GL01 = lambda α, Es: log_GL(α, Es)[0, 1]
                log_GL10 = lambda α, Es: log_GL(α, Es)[1, 0]
                log_GL11 = lambda α, Es: log_GL(α, Es)[1, 1]

                # compute operator that acts on logical subspace
                GL_op = lambda α, Es: log_GL00(α, Es)*ρ_list[0] + \
                                      log_GL01(α, Es)*eigvecs[0]*eigvecs[4].dag() + \
                                      log_GL10(α, Es)*eigvecs[4]*eigvecs[0].dag() + \
                                      log_GL11(α, Es)*ρ_list[4]

                # compute log(G)
                log_G = (GL_op(α, Es) + \
                        np.log(gt(α, Es))*sum(ρ_list[1:4]) + \
                        np.log(gs(α, Es))*sum(ρ_list[5:11]) + \
                        np.log(gq(α, Es))*sum(ρ_list[11:16]) + \
                        V)**r

                # convert log(G) to P basis
                log_G_P = log_G.transform(P.dag()).tidyup(atol = tolerance)

                # append log_G to op_list
                op_list.append(log_G_P)

                # find eigenvalues and eigenvectors of log(G)
                log_G_P_eigvals, log_G_P_eigvecs = log_G_P.eigenstates()

                # compute spectrum of log(G) 
                log_G_P_eigvals_sorted = sort_eigvals(ranking, log_G_P_eigvals, k_start, k_end, k_steps)

                # append spectrum to list 
                spectra_list.append(log_G_P_eigvals_sorted)

                # append eigenvectors to list
                eigvecs_list.append(log_G_P_eigvecs)

                # compute string for title of plot
                title_str = fr'Spectrum of $\log(\mathcal{{G}}(\alpha, E_s = {format_number(Es)}))$'

                # define subfolder
                sub_folder = 'log_G'

        if log == False: 

            for α in α_array: 

                # define operator that acts on the logical qubit subspace
                GL_op = lambda α, Es: g00(α, Es)*ρ_list[0] + \
                                      g04(α, Es)*eigvecs[0]*eigvecs[4].dag() + \
                                      g40(α, Es)*eigvecs[4]*eigvecs[0].dag() + \
                                      g44(α, Es)*ρ_list[4]

                # compute G
                G = (GL_op(α, Es) + \
                    gt(α, Es)*sum(ρ_list[1:4]) + \
                    gs(α, Es)*sum(ρ_list[5:11]) + \
                    gq(α, Es)*sum(ρ_list[11:16]) + \
                    V)**r

                # transform G to P basis
                G_P = G.transform(P.dag()).tidyup(atol = tolerance)

                # append G to op_list
                op_list.append(G_P)

                # find eigenvalues and eigenvectors of log(G)
                G_P_eigvals, G_P_eigvecs = G_P.eigenstates()

                # compute spectrum of log(G) 
                G_P_eigvals_sorted = sort_eigvals(ranking, G_P_eigvals, k_start, k_end, k_steps)

                # append spectrum to list 
                spectra_list.append(G_P_eigvals_sorted)

                # append eigenvectors to list
                eigvecs_list.append(G_P_eigvecs)

                # compute string for title of plot
                title_str = fr'Spectrum of $\mathcal{{G}}(\alpha, E_s = {format_number(Es)})$'

                # define subfolder 
                sub_folder = 'G'

    else: 
        raise ValueError('Please enter a valid VTA_type and log type')


    # convert list into array and create lists 
    # of real and imaginary components
    spectra_array = np.array(spectra_list)
    spectra_array_real = spectra_array.real
    spectra_array_imag = spectra_array.imag

    # convert eigvecs list into array 
    eigvecs_array = np.array(eigvecs_list)

    # find lowest eigenvalue across all α for each eigenvalue 
    # category, i.e., first largest, second largest, etc. 
    asymptotes_list = []

    for i in range(len(k_list)):
        asymptotes_list.append(spectra_array_real[:, i][-1])

    # construct array of thresholds for plotting
    asymptotes_array = np.tile(asymptotes_list, reps = α_steps).reshape(α_steps, len(k_list))

    # modify title string if Δ is not equal to 0 
    if Δ != 0: 
        title_str = title_str + fr' for $\Delta = {format_number(Δ)}$'

    fig, ax = plt.subplots(figsize=(8, 6))

    # customize features of graph if we desire to rank 
    # eigenvalues according to energy or modulus 
    if ranking == 'energy' or ranking == 'modulus': 

        # colors to use in plots
        colors = ['red', 'teal', 'purple', 'black', 'magenta', \
                  'blue', 'orange', 'gray', 'cyan', 'coral', \
                  'lightblue', 'silver', 'olive', 'pink', 'maroon', 'green']

        for i, j in enumerate(k_list):   

            # plots asymptotes
            if asymptotes and any(asymptotes_array[:, i] > tolerance): 
                ax.plot(α_array, asymptotes_array[:, i], \
                        color = colors[j], \
                        label = r'$\lim_{{\alpha \to \infty}}$ ' \
                        + fr'$\gamma_{{{j}}} = {asymptotes_array[0, i] : .3f}$', \
                        linestyle = ':')

            # check if there are nonnegligible imaginary parts
            if any(abs(spectra_array_imag[:, i]) > tolerance):

                # plots imaginary parts of spectrum
                ax.plot(α_array, spectra_array_imag[:, i], color = colors[j], \
                        label = f'$\mathcal{{Im}}(\gamma_{{{j}}})$', linestyle = '--')

                # plot real parts of spectrum
                ax.plot(α_array, spectra_array_real[:, i], color = colors[j], \
                        label = f'$\mathcal{{Re}}(\gamma_{{{j}}})$')

            # plots imaginary parts of spectrum
            elif all(abs(spectra_array_imag[:, i]) <= tolerance):
                # plot real parts of spectrum
                ax.plot(α_array, spectra_array_real[:, i], color = colors[j], \
                        label = f'$\gamma_{{{j}}}$')

            else: 
                raise ValueError('Further examine real and imaginary parts ' \
                                 + f'of eigenvalue {j}')

        ax.legend(fontsize = fontsize, ncol = ncol, loc = loc)

    # customize features of graph if we desire to rank 
    # eigenvalues according to spin
    elif ranking == 'spin': 

        # compute array of expectation values (<g_i|S^2|G_i>) for each α
        expectation_list = []
        for i in range(α_steps):

            # compute expectation of S^2 for each g_i
            expectation_list.append(expect(S2_P, eigvecs_array[i]))

        # compute s for every expectation value in expectation_list
        s_array = find_s(np.array(expectation_list))

        # define colors for plots
        color_map = {0: 'red', 1: 'blue', 2: 'green'}

        legend_elements = []

        if loc_spin != None: 

            # calculate eigenvalues of S^2
            s_eigvals = np.round(S2_P.diag(), 9)

            # convert to integers
            if np.all(s_eigvals - s_eigvals.astype(int)) < tolerance: 
                s_eigvals = s_eigvals.astype(int)
            else: 
                raise ValueError ('There is an eigenvalue that is not an integer')

            # collect data on angular momenta eigenvalues
            s_counts = Counter(s_eigvals)
            s_max = max(s_eigvals)

            # generates handles and labels for spin legend
            spin_handles = [plt.Line2D([0], [0], color = color) for color in color_map.values()]
            spin_labels = [f's = {i}' for i in range(s_max + 1)]

            # find the coordinates of the top center of leg1
            spin_legend = ax.legend(spin_handles, spin_labels, \
                                    fontsize = fontsize, loc = loc_spin)

            # Add the legend for all lines to the axes
            ax.add_artist(spin_legend)

        for i, j in enumerate(k_list):

            # check if there are nonnegligible imaginary parts
            if any(abs(spectra_array_imag[:, i]) > tolerance):

                for p in range(α_steps - 1):

                    # plots imaginary parts of spectrum
                    ax.plot(α_array[p:p+2], spectra_array_imag[:, i][p:p+2], \
                            color = color_map[s_array[:, j][p]], linestyle = '--')

                    # plot real parts of spectrum
                    ax.plot(α_array[p:p+2], spectra_array_real[:, i][p:p+2], \
                            color = color_map[s_array[:, j][p]])

            # plots  spectrum
            elif all(abs(spectra_array_imag[:, i]) < tolerance):

                for p in range(α_steps - 1):
                    # plot real parts of spectrum
                    ax.plot(α_array[p:p+2], spectra_array[:, i][p:p+2], \
                            color = color_map[s_array[:, j][p]])

            else: 
                raise ValueError('Further examine real and imaginary parts ' \
                                 + f'of eigenvalue {j}')

    ax.set_xlabel(r'$\alpha$', fontsize = 15)
    ax.set_title(title_str, fontsize=17.5)
    ax.grid(True)

    if savefile: 

        # create name for png file 
        parent_dir = f'{directory}/{N}_sites/λ={round(λ, 3)}/{sub_folder}/{ranking}'

        # Check if the directory already exists
        if not os.path.exists(parent_dir):
            # Create the directory
            os.makedirs(parent_dir)

        # save figure under custom filename
        filename = parent_dir + f'/Es={round(Es, 3)}_{k_start}_{k_end}_{k_steps}_{Δ}.png'
        plt.savefig(filename)

    if return_op_list: 
        return op_list
        
        
def sweep_spectra(VTA_type, log, N, Jx, Jy, Jz, periodic_bc, \
                 α_start, α_end, α_steps, \
                 Es_start, Es_end, Es_gradation, \
                 ranking, k_start, k_end, k_steps, asymptotes, \
                 tolerance, directory, Δ = 0, r = 1, fontsize = 9, \
                 ncol = 4, loc = 'best', loc_spin = None, \
                 return_op_list = False, savefile = True):
    
    '''
    plots the spectrum of VTA (or VTA_exact) or the spectrum of
    log of VTA (or VTA_exact) for a sweep over Es
    '''

    # generate list over which we will sweep Es
    if Es_start <= Es_end: 
        Es_list = np.arange(Es_start, Es_end + Es_gradation, Es_gradation)
    else: 
        raise ValueError('Please enter valid values of Es_start and Es_end')
    
    if return_op_list:
        
        # create list to fill with operators
        op_list = []
    
        for Es in Es_list:  
        
            # plot VTA spectrum
            op_list_analytic = plot_spectra(VTA_type, log, N, Jx, Jy, Jz, periodic_bc, \
                                            α_start, α_end, α_steps, Es, \
                                            ranking, k_start, k_end, k_steps, asymptotes, \
                                            tolerance, directory, Δ = Δ, r = r, fontsize = fontsize, \
                                            ncol = ncol, loc = loc, loc_spin = loc_spin, \
                                            return_op_list = return_op_list, savefile = savefile)

            # append list to op_list
            op_list.append(op_list_analytic)
            
        return op_list[0]
    
    else:
        
        for Es in Es_list:  
        
            # plot VTA spectrum
            plot_spectra(VTA_type, log, N, Jx, Jy, Jz, periodic_bc, \
                         α_start, α_end, α_steps, Es, \
                         ranking, k_start, k_end, k_steps, asymptotes, \
                         tolerance, directory, Δ = Δ, r = r, fontsize = fontsize, \
                         ncol = ncol, loc = loc, loc_spin = loc_spin, \
                         return_op_list = return_op_list, savefile = savefile)