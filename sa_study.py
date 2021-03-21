#sensitivity analysis study
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze
import numpy as np
from scipy.integrate import solve_ivp, simps
from numba import njit
from tqdm import tqdm
import pandas as pd
import os
import matplotlib.pyplot as plt
from scikits.odes.odeint import ode


### Parameters from model ###
# Mustafa et al. (2014)
mu_max = 0.23
mu_maxd = 0.22
alpha = 1.74
beta = 2.5
Ks = 20.
phi =  0.2
Kss = 150.
Ksp = 9.5
Kssp = 200.
mp = 1.9
ms = 3.5
Pc = 250.
Pcd = 350.
Yxs = 0.03
Yxp = 0.375
Din = 0.06

S0 = 140.
P0 = 0.
Xnv0 = 0.
Xd0 = 0.
Xv0 = 2.5

T0=0
TF=20.
PSET=65.

times_for_sa = [T0,
                0.2*TF,
                0.4*TF,
                0.6*TF,
                0.8*TF,
                TF]

def wrapper_for_sa(parameters):
    mu_max = parameters[0]
    mu_maxd = parameters[1]
    alpha = parameters[2]
    beta = parameters[3]
    Ks = parameters[4]
    phi = parameters[5]
    Kss = parameters[6]
    Ksp = parameters[7]
    Kssp = parameters[8]
    mp = parameters[9]
    ms = parameters[10]
    Pc = parameters[11]
    Pcd = parameters[12]
    Yxs = parameters[13]
    Yxp = parameters[14]
    Din = parameters[15]
    S0 =  parameters[16]
    P0 =  parameters[17]
    Xnv0 = parameters[18]
    Xd0 =  parameters[19]
    Xv0 =  parameters[20]

    
    @njit
    def ZymFerm_wrapped(t, u, du):
        Xv = u[0]
        Xnv = u[1]
        Xd = u[2]
        P = u[3]
        S = u[4]

        mu_v = ((mu_max*S)/(Ks+S+(S**2)/Kss))*((1-(P/Pc))**alpha)
        mu_nv = ((mu_maxd*S)/(Ksp+S+(S**2)/Kssp))*((1-(P/Pcd))**beta) - mu_v
        mu_d =  phi*mu_nv

        #dX=Yx*((S*E)/(Ks+S)) + Din*X0 - Dout*X
        #dS=(-Yx/Ysx)*((S*E)/(Ks+S)) - ms*X + Din*S0 - Dout*S
        #dE=(k1-k2*P+k3*(P**2))*((S*E)/(Ks+S)) + Din*E0 - Dout*E
        #dP=(Yx/Ypx)*((S*E)/(Ks+S)) + mp*X + Din*P0 - Dout*P

        dXv = Din*(Xv0 - Xv) + Xv*(mu_v - mu_nv)
        dXnv = Din*(Xnv0 - Xnv) + Xv*mu_nv - Xnv*mu_d
        dXd = Din*(Xd0 - Xd) + Xnv*mu_d
        dP = Din*(P0-P) + Xv*(mu_v/Yxp) + mp*Xnv
        dS = Din*(S0-S) - Xv*(mu_v/Yxs) - ms*Xnv

        du[:] = [dXv, dXnv, dXd, dP, dS]
    

    y_0 = [Xv0, Xnv0, Xd0, P0, S0]
    #sol = solve_ivp(fun=ZymFerm_wrapped, 
    #                t_span=(T0, TF),
    #                y0=np.array(y_0),
    #                method='BDF',
    #                #jac=Jac_ZymFerm,
    #                #atol=1e-8,
    #                #rtol=1e-8,
    #                t_eval=np.linspace(T0, TF, 100))
    options= {'max_steps': 100e3}
    sol = ode('cvode', 
            ZymFerm_wrapped, 
            old_api=False, **options).solve(times_for_sa, y_0)
    return sol.values



def sensitivity_analysis() -> dict:
    """
    """
    #List parameters and their range
    mu_max_range = [.5*mu_max, 1.5*mu_max]
    mu_maxd_range = [.5*mu_maxd, 1.5*mu_maxd]
    alpha_range = [.5*alpha, 1.5*alpha]
    beta_range = [.5*beta, 1.5*beta] 
    Ks_range = [.5*Ks, 1.5*Ks] 
    phi_range = [0., 0.4]
    Kss_range = [.5*Kss, 1.5*Kss] 
    Ksp_range = [.5*Ksp, 1.5*Ksp]
    Kssp_range = [.5*Kssp, 1.5*Kssp]
    mp_range = [.5*mp, 1.5*mp] 
    ms_range = [.5*ms, 1.5*ms] 
    Pc_range = [.5*Pc, 1.5*Pc] 
    Pcd_range = [.5*Pcd, 1.5*Pcd] 
    Yxs_range = [.5*Yxs, 1.5*Yxs] 
    Yxp_range = [.5*Yxp, 1.5*Yxp] 
    Din_range = [1e-3, 0.25] 
    S0_range = [50., 200.]
    P0_range = [0., 5.]
    Xnv0_range = [0., 1e-2]
    Xd0_range = [0., 1.e-2]
    Xv0_range = [0., 5.]
    #Times for evaluation

    problem={'num_vars':21,
             'names':['mu_max', 'mu_maxd', 'alpha', 'beta', 'Ks', 'phi', 'Kss', 'Ksp', 'Kssp', 'mp', 'ms', 'Pc', 'Pcd', 'Yxs', 'Yxp', 'D_in', 'S_0', 'P_0', 'X_nv0', 'X_d0', 'Xv_0'],
             'bounds':[mu_max_range , mu_maxd_range , alpha_range, beta_range, Ks_range, phi_range, Kss_range, Ksp_range , Kssp_range, mp_range, ms_range, Pc_range, Pcd_range, Yxs_range, Yxp_range, Din_range, S0_range , P0_range, Xnv0_range, Xd0_range, Xv0_range]   
    }

    print("\n Problem bounds =", problem['bounds'])

    #Sample values    
    print("Starting SA.\n")
    print("\t Sampling parameters.\n")    
    param_values = morris_sample.sample(problem, 500, num_levels=10)
    output = np.zeros((param_values.shape[0], len(times_for_sa)))
    print("\t Gathering output from parameter samples for time points.\n")
    for i, X in tqdm(enumerate(param_values)):
        #Evaluate model for different times and store it accordingly
        #Calculate with  dedicated wrapper
        sol = wrapper_for_sa(X)
        y, t = sol.y, sol.t
        P = y[:, 3]
        output[i, :] = P #simps((P-PSET)**2, t)
    
    sa_results_over_time = np.zeros([problem['num_vars'], len(times_for_sa)])
    print("\t Performing elementary effects analysis for each time point.\n")
    for j in tqdm(range(output.shape[1])):
        #Get SA results for current time point
        sa_results_this_time_point = morris_analyze.analyze(problem, param_values, output[:, j], print_to_console=False, num_levels=10)
        #Save only  normalized mu_star and sigma_star
        sa_results_over_time[:, j] = sa_results_this_time_point["mu_star"]/np.sum(sa_results_this_time_point["mu_star"])
    print("\n End of SA.")
    #Fill up DataFrame
    column_names = [str("t=")+str(p) for p in times_for_sa]
    sa_df = pd.DataFrame(data=sa_results_over_time, columns=column_names, index=problem['names'])
    
    print(sa_df)

    #Making plot for visual identification
    fig, ax = plt.subplots(figsize=(9,7))
    _bottom = 0
    #_var_names = problem['names']
    _var_names = []
    for v, var_name in enumerate(problem['names']):
        if var_name not in ['D_in', 'S_0', 'P_0', 'X_nv0', 'X_d0', 'Xv_0', 'Ks', 'Kss', 'Ksp', 'Kssp', 'mp', 'ms', 'Pc', 'Pcd', 'Yxs', 'Yxp', 'mu_max', 'mu_maxd']:
            _var_names.append("$"+"\\"+var_name+"$")
        elif var_name == 'mu_max':
            _var_names.append("$\mu_{max}$")
        elif var_name == 'mu_maxd':
            _var_names.append("$\mu_{maxd}$")
        elif var_name == 'D_in':
            _var_names.append("$D_{in}$")
        elif var_name == 'S_0':
            _var_names.append("$S_{0}$")
        elif var_name == 'P_0':
            _var_names.append("$P_{0}$")
        elif var_name == 'X_v0':
            _var_names.append("$Xv_0$")
        elif var_name == 'X_nv_0':
            _var_names.append("$Xnv_0$")
        elif var_name == 'X_d0':
            _var_names.append("$Xd_0$")
        elif var_name == 'Ks':
            _var_names.append("$K_s$")
        elif var_name == 'Kss':
            _var_names.append("$K_{ss}$")
        elif var_name == 'Ksp':
            _var_names.append("$K_{sp}$")
        elif var_name == 'Kssp':
            _var_names.append("$K_{ssp}$")
        elif var_name == 'mp':
            _var_names.append("$m_p$")
        elif var_name == 'ms':
            _var_names.append("$m_s$")
        elif var_name == 'Pc':
            _var_names.append("$P_c$")
        elif var_name == 'Pcd':
            _var_names.append("$P_{cd}$")
        elif var_name == 'Yxs':
            _var_names.append("$Y_{xs}$")
        elif var_name == 'Yxp':
            _var_names.append("$Y_{xp}$")
        else:
            _var_names.append("$"+var_name+"$")
    color_for_vars = ["gray", "lightgrey", "rosybrown", "red", "tomato", "chocolate", "saddlebrown", "darkorange", "olive", "darkolivegreen", "green", "teal", "aqua", "steelblue", "navy", "indigo", "darkviolet", "violet", "pink", "mediumseagreen", "purple"]
    #color_for_vars = [np.random.rand(3,) for c in range(problem['num_vars'])]
    
    for i in range(problem['num_vars']):
        print("# Var = ", problem['names'][i])
        print("S.A. over time = ", sa_results_over_time[i, :])
        print("Times for S.A. = ", times_for_sa)

        ax.bar(times_for_sa, sa_results_over_time[i, :], width=1, bottom=_bottom, label=_var_names[i], color=color_for_vars[i])#color_for_vars[i])
        _bottom += sa_results_over_time[i, :]
    
    ax.set_ylabel(r"Absolute sensitivity coefficient ($\mu^*$)")
    ax.set_xlabel(r"Time (h)")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fontsize=10)
    #ax.legend(loc='upper left', bbox_to_anchor=[0.07, 0.5, 0.5, 0.5], ncol=10, fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(times_for_sa)
    ax.tick_params(axis='x', which='major', labelsize=9, labelrotation=90)
    #plt.tight_layout()
    plt.savefig("sa_analysis.png", dpi=300, bbox_inches='tight')


sensitivity_analysis()