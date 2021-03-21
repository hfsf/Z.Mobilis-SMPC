#SCMPC formulation

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp, simps
from scipy.optimize import differential_evolution, Bounds, minimize
from scipy.stats import truncnorm
from matplotlib import pyplot as plt
import pygmo as pg
from math import ceil
from tqdm import tqdm

import solve_ocp
from configurations import *

conf_level = 1e-7
violation_prob = 0.05
number_manipulated_vars = 1

#Set random seed for reproducibility
RANDOM_SEED=12345
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(seed=RANDOM_SEED)

#Optimization problem for pygmo
class optimization_problem:
    def __init__(self, bounds, Fun, args):
        self.bounds=bounds
        self.Fun=Fun
        self.args=args

    def fitness(self, x):
        return [self.Fun(x, *self.args)]

    def get_bounds(self):
        #Pass to pygmo format

        #bounds_l=[np.array(b[0]) for b in self.bounds]
        #bounds_u=[np.array(b[1]) for b in self.bounds]
        #print("bounds=", (bounds_l, bounds_u))

        return tuple([np.array(i) for i in self.bounds])

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)

#Function for cost calculation over multiple scenarios
def multi_scenario_fermentation_cost(x, t0=T0, tf=TF, nint=NINT, x0=0, parameter_scenarios=[]):

    #Iterate over all  scenarios
    #For each column (representing an specific scenario), run the cost function and evaluate maximum output
    #print("-> Parameter scenarios = ", parameter_scenarios)
    J_for_scenarios=[]
    for ns in tqdm(range(parameter_scenarios.shape[1]), leave=False):
        sol = solve_ocp.solve_fermentation(nint,  #N=Number of intervals
                                           t0,    #t0=Initial time
                                           tf,    #tf = End time 
                                           x,     #Din = Input dilution factor
                                           x0,    #S0 = Initial substrate concentration
                                           1,     #is_stochastic = Activate  stochastic parameters
                                           1.,    #is_optimizing = Din is being passed as Array
                                           None,
                                           parameter_scenarios[:, ns])
        
        t_ = np.linspace(t0, tf, nint+1)
        #IF USING SCIKIT.ODES
        if SHOULD_USE_SCIPY is False:
            P = sol.y[:, 3]
            S = sol.y[:, 4]
        #IF USING SCIPY ODEINT
        if SHOULD_USE_SCIPY is True:
            P = sol.y[3, :]
            S = sol.y[4, :]
        x=np.concatenate(([x[0]], x))
        #For preventing to take solutions with premature conclusion (tf < TF) and raising and error in integration
        x = x[:len(t_)]
        x_ = np.zeros(x.shape)
        x_[0] = x[0]
        x_[1:] = x[:-1]
        if P[-1] < 1 or np.isnan(P[-1]) == True or len(x-x_) != len(t_) or len(P) != len(t_):
            if IS_VERBOSE:            
                print("\n\t->Error in solution. Penalizing with J=1e10")
                print("\n\t->P = ", P)
                print("\n\t->S = ", S)
                print("\n\t->x = ", x)
                print("\n\t->Δx = ", (x-x_))
            J_ = 1e100
        else:
            #J is modified for minimization
            #try:
                #    ------ ISE ------     + ------ Δx² ------               
            J_ = simps((P-PSET)**2, t_)# + simps((x-x_)**2, t_)# - P[-1]/(tf-t0)

        #If the solution was penalized,  do not include it
        #if J_ != 1e100:
        J_for_scenarios.append(J_)
    
    #Avoid returning penalized solutions
    return min(max(J_for_scenarios), 1e100) 

def scenario_optimization(t0, tf, nint, x0, is_stochastic=False, number_of_scenarios=None):
    #Sample parameters
    parameter_scenarios = sample_scenarios(number_of_scenarios)
    #Perform calculations for the scenarios
    print("-> Performing optimization.")
    bounds = ([DIN_LOWER_BOUND]*nint, [DIN_UPPER_BOUND]*nint)
    opt_problem=optimization_problem(bounds, multi_scenario_fermentation_cost, [t0, tf, nint, x0, parameter_scenarios])
    prob=pg.problem(opt_problem)

    if USE_HEURISTIC is False:
        algo=pg.algorithm(pg.nlopt("slsqp"))
        pop=pg.population(prob, 1)
        pop.problem.c_tol = [1e-8] * prob.get_nc()
        pop=algo.evolve(pop)
        result_x=pop.champion_x
        result_fun=pop.champion_f[0]

    else:
        algo=pg.algorithm(uda=pg.cmaes(gen=N_GEN,
                                       #F=.6,
                                       #CR=.9,
                                       force_bounds=True,
                                       #allowed_variants=[6, 2, 3, 7, 10, 13, 14, 15, 16, 18],
                                       #allowed_variants=[2, 6, 7, 13, 17, 16],
                                       #variant_adptv=2,
                                       seed=RANDOM_SEED,
                                       ftol=1e-7,
                                       xtol=1e-7))
        algo.set_verbosity(10)
        pop_=pg.population(prob, POP_SIZE)
        pop=algo.evolve(pop_)
        result_x_old=pop.champion_x
        result_fun=pop.champion_f[0]

        print("\n \t-> Polishing result with local optimization...")
        try:
            algo_=pg.algorithm(uda=pg.compass_search(max_fevals=1000,
                                                    start_range=1e-2,
                                                    stop_range=1e-10))
                
            algo_.set_verbosity(10)
            algo.set_verbosity(10)
            pop_=algo_.evolve(pop)
            result_x=pop_.champion_x
            result_fun=pop_.champion_f[0]

        except:
            print("\t\t Some error ocurred. Aborted local optimization.")

    print("\n -> Optimization ended.")                 
    print("\n -> Optimization result = ", result_x, "(J = ",result_fun,")")
    
    return result_x


def sample_scenarios(number_of_scenarios):
    #Sample stochastic parameters for an specific number of scenarios
    #-> If the number of scenarios was not provided, calculate it [ González et al. (2020) - A Comparative Study of Stochastic Model Predictive Controllers.  Electronics 2020, 9(12), 2078]
    if number_of_scenarios is None:
        number_of_scenarios = ceil(( NINT*number_manipulated_vars + 1 + np.log(1./conf_level) + np.sqrt(2*(NINT*number_manipulated_vars + 1)*np.log(1./conf_level)) )/violation_prob)
    #Draw samples for each stochastic variable

    Yxp_samples = rng.uniform((1.-UNIFORM_DEV)*YXP_MEAN, (1.+UNIFORM_DEV)*YXP_MEAN, size=number_of_scenarios)
    mp_samples = rng.uniform((1.-UNIFORM_DEV)*MP_MEAN, (1.+UNIFORM_DEV)*MP_MEAN, size=number_of_scenarios)
    mu_max_samples = rng.uniform((1.-UNIFORM_DEV)*MU_MAX_MEAN, (1.+UNIFORM_DEV)*MU_MAX_MEAN, size=number_of_scenarios)
    mu_maxd_samples = rng.uniform((1.-UNIFORM_DEV)*MU_MAXD_MEAN, (1.+UNIFORM_DEV)*MU_MAXD_MEAN,  size=number_of_scenarios)
    
    parameters_for_scenarios = np.vstack((Yxp_samples, mp_samples, mu_max_samples, mu_maxd_samples))

    return parameters_for_scenarios

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
    U_new = scenario_optimization(t0=t, 
                                tf=TF, 
                                nint=int((TF-t)/DT),
                                x0=current_states[-1], 
                                is_stochastic=False, 
                                number_of_scenarios=None)
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
    

def SCMPC(number_of_scenarios=None):
    #Main function for MPC
    #Initialize system states (Xv, Xnv, Xd, P, S)
    system_states = np.zeros((1, 5), dtype=np.float64)
    #Start with X_0, T_0
    system_states[:] = Xv0, Xnv0, Xd0, P0, S0
    control_actions = []
    current_time = T0
    Delta_t = (TF-T0)/NINT

    M = ceil(( NINT*number_manipulated_vars + 1 + np.log(1./conf_level) + np.sqrt(2.*(NINT*number_manipulated_vars + 1)*np.log(1./conf_level)) )/violation_prob)

    #Start receding horizon
    print(f"\n # Starting SCMPC [{T0};{TF}], with Delta_t = {Delta_t}, M = {M}")
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

    plot_scmpc_solution(system_states, control_actions)

def plot_scmpc_solution(states, control_actions):
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
    fig.savefig("solution_OCP_with_control_var_SCMPC.png")
    plt.clf()

SCMPC()