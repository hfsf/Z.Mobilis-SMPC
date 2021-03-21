#MPC formulation

import numpy as np
from numba import njit
import random
from scipy.integrate import solve_ivp, simps
from scipy.optimize import differential_evolution, Bounds, minimize
from scipy.stats import truncnorm
from matplotlib import pyplot as plt
from configurations import *

import solve_ocp

def get_new_U_step(t, tf, current_states):
    #Perform optimization for t=[t; TF], obtaining new U
    #-> call for optimize_U function
    #-> return U_step for implementation
    U_projected = optimize_U(t, tf, current_states)
    #Return only the first control action
    return U_projected

def optimize_U(t, tf, current_states):
    #Perform optimization for U, for t=[t; TF] obtaining new U
    #-> return U
    U_new = solve_ocp.perform_optimization(t0=t, 
                                           tf=TF, 
                                           nint=int((TF-t)/DT),
                                           x0=current_states[-1], 
                                           is_stochastic=False, 
                                           open_loop=False)
    return U_new

def implement_U_step(system_states, u_next, initial_time, end_time):
    #Implement U_step received from get_new_U_step
    #-> return new state for the system [initial_time; end_time]
    new_states_ = solve_ocp.solve_fermentation(1, #Just implement one control action
                                              initial_time,
                                              end_time,
                                              [u_next],
                                              initial_state=system_states[-1],
                                              is_stochastic=True,
                                              is_optimizing=False)
    #Get new states from Solution object
    new_states = new_states_.y[-1]
    return np.vstack((system_states, new_states))
    

def deterministic_MPC():
    #Main function for MPC
    #Initialize system states (Xv, Xnv, Xd, P, S)
    system_states = np.zeros((1, 5), dtype=np.float64)
    #Start with X_0, T_0
    system_states[:] = Xv0, Xnv0, Xd0, P0, S0
    control_actions = []
    current_time = T0
    Delta_t = (TF-T0)/NINT
    #Start receding horizon
    print(f"\n # Starting MPC [{T0};{TF}], with Delta_t = {Delta_t}")
    print(f"\n\t - initial state: {system_states}")
    while(current_time < TF):
        print("\n->Time: ", current_time)
        print(f"\n\t * Calculating new U (t=[{current_time}; {TF}]).")
        next_control_step = get_new_U_step(current_time,
                                           TF,
                                           system_states)                                   
        #print("\n\t * Calculated U = ", next_control_step)
        next_control_step = next_control_step[0]
        print("\n\t * Implementing control action (U =",next_control_step,").")
        system_states = implement_U_step(system_states, 
                                         next_control_step,
                                         current_time,
                                         current_time + Delta_t)
        #Store control action performed
        control_actions.append(next_control_step)
        #Finalize iterating time
        print("\n * System states = ", system_states)
        print("\n * Control actions = ", control_actions)
        print("\n\t * Iterating time.")
        current_time+=Delta_t
    print("\n # End of MPC.")
    control_actions = np.array(control_actions, dtype=np.float64)
    system_states = np.array(system_states, dtype=np.float64)
    print("\n * Control actions = ", control_actions)
    print("\n * System states = ", system_states)
    print("\n * Ploting solution")

    plot_mpc_solution(system_states, control_actions)

def plot_mpc_solution(states, control_actions):
    t_ = np.linspace(T0, TF, int((TF-T0)/DT)+1)
    #SHOULD_USE_SCIPY = True
    #if SHOULD_USE_SCIPY is True:
    #    X = sol.y[0]
    #    S = sol.y[4]
    #    Xnv = sol.y[1]
    #    P = sol.y[3]
    #IF USING SCIKIT.ODES ODEINT
    #if SHOULD_USE_SCIPY is False:
    #    X = sol.y[:,0]
    #    S = sol.y[:, 4]
    #    Xnv = sol.y[:, 1]
    #    P = sol.y[:, 3]
    Xv = states[:, 0]
    P = states[:, 3]
    S = states[:, 4]
    S[S<0] = 0.

    fig, axs = plt.subplots(nrows=1,ncols=2, figsize=(12,4))
    axs[0].plot(t_, Xv, label=r"$X_v\,(t)$", linestyle='-', lw=2., color='orange')
    axs[0].plot(t_, S, label=r"$S\,(t)$", linestyle='-', lw=2., color='blue')  
    axs[0].plot(t_, P, label=r"$P\,(t)$", linestyle='-', lw=2., color='red')  
    Pset = np.full(len(t_), PSET)
    axs[0].plot(t_, Pset, label=r"$P_{set}$", linestyle='--', lw=2., color='lightgreen')    
    #axs[0].plot(t_, Xnv, label=r"$X_{nv}\,(t)$", linestyle='-', lw=2., color='green')    
    axs[0].grid(b=True, which='major')
    axs[0].set_xlim((T0, TF))
    axs[0].set_ylim(bottom=0.)
    axs[0].set_xticks(np.arange(T0, TF+2., 2.))
    axs[0].set_xlabel(r"Time $(h)$", fontsize=12)
    axs[0].set_ylabel(r"Concentration $(g\,L^{-1})$", fontsize=12)
    axs[0].legend(fontsize=11)
    plt.xticks(rotation=80)
    #---SECOND SUBPLOT---#
    Din_t = np.concatenate(([control_actions[0]], control_actions))
    axs[1].step(t_, Din_t, label=r"$D_{in}\,(t)$", linestyle='-', lw=2., color='purple')
    axs[1].grid(b=True, which='major')    
    axs[1].set_xlim((T0, TF))
    #axs[1].set_ylim((DIN_LOWER_BOUND, DIN_UPPER_BOUND))
    axs[1].set_xticks(np.arange(T0, TF+2., 2.))
    axs[1].set_xlabel(r"Time $(h)$", fontsize=12)
    axs[1].set_ylabel(r"Input dilution factor $(h^{-1})$", fontsize=12)
    #axs[1].legend(fontsize=11)
    for ax_n in [0, 1]:
        for tick in axs[ax_n].get_xticklabels():
            tick.set_rotation(80)
    #-------------
    fig.tight_layout()
    fig.savefig("solution_OCP_with_control_var_DMPC.png")
    plt.clf()

deterministic_MPC()