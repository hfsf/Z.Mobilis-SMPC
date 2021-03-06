#Solve the OCP problem

import numpy as np
from numba import njit
import random
from scipy.integrate import solve_ivp, simps
from scipy.optimize import differential_evolution, Bounds, minimize, LinearConstraint
from scipy.stats import truncnorm
from matplotlib import pyplot as plt
#import pyomo.environ as pyo
#from diffeqpy import de
from scikits.odes.odeint import odeint, ode
import pygmo as pg

#Import model parameters
from configurations import *
#Set random seed for reproducibility
RANDOM_SEED=12345
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(seed=RANDOM_SEED)

#User Defined Problem class for pygmo
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

'''
@njit
def Jac_ZymFerm(t, u, p):
    Din_vec = p[:NINT]
    Yx, ms, k1 = p[NINT:NINT+3]
    is_stochastic = p[-3]
    is_optimizing = p[-2]
    dt = p[-1]

    if is_stochastic == 1.:
        Yx = np.random.uniform(YX_LOWER, YX_UPPER)
        ms = np.random.uniform(MS_LOWER, MS_UPPER)
        k1 = np.random.uniform(K1_LOWER, K1_UPPER)    
    
    #Hard-coding the restrictions in the states (>0)
    X = u[0]
    S = max(0., u[1])
    E = u[2]
    P = u[3]

    Jac = np.zeros((4, 4), dtype=np.float64)    
    Jac[0,0] = -Dout   
    Jac[0,1] = (- E*S*Yx)/((Ks+S)**2)  + E*Yx/(Ks+S)
    Jac[0,2] = S*Yx/(Ks+S)    
    Jac[0,3] = 0.

    Jac[1,0] = -ms   
    Jac[1,1] = -Dout + (- E*S*Yx)/(Ysx*(Ks+S)**2) - E*Yx/(Ysx*(Ks+S))   
    Jac[1,2] =  -S*Yx/(Ysx*(Ks+S))   
    Jac[1,3] = 0.
    
    Jac[2,0] = 0.   
    Jac[2,1] = ( E*S*(k3*(P**2) - P*k2 + k1)/(Ks+S)**2 ) + E*(k3*(P**2) - P*k2 + k1)/(Ks+S)       
    Jac[2,2] = -Dout + ( S*(k3*(P**2) - P*k2 + k1)/(Ks+S) )    
    Jac[2,3] = E*S*(2.*P*k3 - k2)/(Ks+S)
    
    Jac[3,0] = mp     
    Jac[3,1] = (- E*S*Yx)/(Ypx*(Ks+S)**2) + E*Yx/(Ypx*(Ks+S))      
    Jac[3,2] = - Dout + S*Yx/(Ypx*(Ks+S))      
    Jac[3,3] = 0.

    return Jac       
'''

@njit
def ZymFerm(t, u, p):
    
    #Bring variables to local namespace
    Yxp = YXP_MEAN
    mu_max = MU_MAX_MEAN
    mu_maxd = MU_MAXD_MEAN
    mp = MP_MEAN

    #Mustafa et al. (2014) -  Structured Mathematical Modeling, Bifurcation, and Simulation for the Bioethanol Fermentation Process Using Zymomonas mobilis. doi:10.1021/ie402361b    
    #p = np.concatenate((Din, [mu_max_, alpha_, S0_], [is_stochastic, is_optimizing, dt, N]))
    N = int(p[-1])
    Din_vec = p[:N]
    dt = p[-2]
    is_optimizing = p[-3]    
    is_stochastic = p[-4]
    V4 = p[-5]
    V3 = p[-6]
    V2 = p[-7]
    V1 = p[-8]
    #Employ the ni-th element from Din array
    ni = np.array([t // dt, N-1], dtype=np.int64).min()
    if is_stochastic == 1.:
        Yxp = V1
        mp = V2
        mu_max = V3
        mu_maxd = V4

    if is_optimizing == 1.:
        Din = Din_vec[ni]
    else:
        Din = Din_vec[0]
    Xv = u[0]
    Xnv = u[1]
    Xd = u[2]
    P = u[3]
    S = u[4]

    if Yxp * mp * mu_max * mu_maxd == 0:
        print("Yxp = ", Yxp)
        print("V1 = ", V1)
        print("mp = ", mp)
        print("mu_max = ", mu_max)
        print("mu_maxd = ", mu_maxd)
        print("N = ", N)
        print("p = ", p)
        print("len(p) = ", len(p))

    #dX=Yx*((S*E)/(Ks+S)) + Din*X0 - Dout*X
    #dS=(-Yx/Ysx)*((S*E)/(Ks+S)) - ms*X + Din*S0 - Dout*S
    #dE=(k1-k2*P+k3*(P**2))*((S*E)/(Ks+S)) + Din*E0 - Dout*E
    #dP=(Yx/Ypx)*((S*E)/(Ks+S)) + mp*X + Din*P0 - Dout*P

    mu_v = ((mu_max*S)/(Ks+S+(S**2)/Kss))*((1-(P/Pc))**alpha)
    mu_nv = ((mu_maxd*S)/(Ksp+S+(S**2)/Kssp))*((1-(P/Pcd))**beta) - mu_v
    mu_d =  phi*mu_nv

    dXv = Din*(Xv0 - Xv) + Xv*(mu_v - mu_nv)
    dXnv = Din*(Xnv0 - Xnv) + Xv*mu_nv - Xnv*mu_d
    dXd = Din*(Xd0 - Xd) + Xnv*mu_d
    dP = Din*(P0-P) + Xv*(mu_v/Yxp) + mp*Xnv
    dS = Din*(S0-S) - Xv*(mu_v/Yxs) - ms*Xnv

    #print("->")

    return [dXv, dXnv, dXd, dP, dS]


def calback_fun(xk, convergence=False):
    if IS_VERBOSE:
        print(f"\n x = {xk} \n \t convergence = {convergence}")

#Function for resolution of the OCP
def fermentation_cost(x, t0=T0, tf=TF, nint=NINT, x0=0., is_stochastic=False):
    sol = solve_fermentation(nint,          #N=Number of intervals
                            t0,             #t0=Initial time
                            tf,             #tf = End time 
                            x,              #Din = Input dilution factor
                            x0,             #S0 = Initial substrate concentration
                            is_stochastic,  #is_stochastic = Activate  stochastic parameters
                            1.,             #is_optimizing = Din is being passed as Array
                            None,
                            [])
    t_ = sol.t
    #IF USING SCIKIT.ODES
    if SHOULD_USE_SCIPY is False:
        P = sol.y[:, 3]
        S = sol.y[:, 4]
        Xnv = sol.y[:,1]
        Xv = sol.y[:,0]
        Xd = sol.y[:,2]
    #IF USING SCIPY ODEINT
    if SHOULD_USE_SCIPY is True:
        P = sol.y[3, :]
        S = sol.y[4, :]
        Xnv = sol.y[1, :]
        Xv = sol.y[0, :]
        Xd = sol.y[2, :]
    x=np.concatenate(([x[0]], x))
    #For preventing to take solutions with premature conclusion (tf < TF) and raising and error in integration
    x = x[:len(t_)]
    x_ = np.zeros(x.shape)
    x_[0] = x[0]
    x_[1:] = x[:-1]
    if P[-1] < 1 or np.isnan(P[-1]) == True or len(x-x_) != len(t_) or len(P) != len(t_) or []:
        if IS_VERBOSE:            
            print("\n\t->Error in solution. Penalizing with J=1e10")
            print("\n\t->P = ", P)
            print("\n\t->S = ", S)
            print("\n\t->x = ", x)
            print("\n\t->??x = ", (x-x_))
        J = 1e100
    else:
        #J is modified for minimization
        #try:
            #    ------ ISE ------     + ------ ??x?? ------               
        J = simps((P-PSET)**2, t_)# + simps((x-x_)**2, t_)# - P[-1]/(tf-t0)

    return J        

def solve_fermentation(N, t0, tf, Din, initial_state, is_stochastic, is_optimizing, last_run=None, predefined_stochastic_parameters=[]):

    if last_run is True:
        USE_SCIPY = True
    else:
        USE_SCIPY = SHOULD_USE_SCIPY

    if IS_VERBOSE:
        if USE_SCIPY is True:
            print("\t->Solving fermentation problem. (Using SCIPY)")
        else:    
            print("\t->Solving fermentation problem. (Not using SCIPY)")
    dt = (tf-t0)/N

    if predefined_stochastic_parameters == []:

        Yxp_ = rng.uniform((1.-UNIFORM_DEV)*YXP_MEAN, (1.+UNIFORM_DEV)*YXP_MEAN)
        mp_ = rng.uniform((1.-UNIFORM_DEV)*MP_MEAN, (1.+UNIFORM_DEV)*MP_MEAN)
        mu_max_ = rng.uniform((1.-UNIFORM_DEV)*MU_MAX_MEAN, (1.+UNIFORM_DEV)*MU_MAX_MEAN)
        mu_maxd_ = rng.uniform((1.-UNIFORM_DEV)*MU_MAXD_MEAN, (1.+UNIFORM_DEV)*MU_MAXD_MEAN)
    #If the parameter scenarios are passed, use them as 
    else:

        Yxp_, mp_, mu_max_, mu_maxd_ = predefined_stochastic_parameters

    #Concatenate all arguments to pass to a flattene vector (Numba requires it)
    p = np.concatenate((Din, [Yxp_, mp_, mu_max_, mu_maxd_], [is_stochastic, is_optimizing, dt, N]))

    def ZymFerm_wrapper_for_odes(t, X, ydot):
        ydot[:] =  ZymFerm(t, X, p)

    #Pass the predefined parameter scenarios for resolution of the equations. Only relevant if stochasticity is considered (is_stochastic = 1 )    

    if t0 == 0.:
        y_0 = [Xv0, Xnv0, Xd0, P0, S0]
    else:
        y_0 = initial_state
    #=== SOLVING WITH SCIPY ===
    #S0=np.random.normal(loc=S0_MEAN, scale=S0_STDDEV)
    if USE_SCIPY is True:
        sol = solve_ivp(fun=ZymFerm, 
                        t_span=(t0, tf),
                        y0=np.array(y_0),
                        args=(p,),
                        method='BDF',
                        #jac=Jac_ZymFerm,
                        #atol=1e-8,
                        rtol=1e-8,
                        t_eval=np.linspace(t0, tf, N+1))
        
    #=== SOLVING WITH SCIKIT.ODES ===
    if USE_SCIPY is False:
        y0 =  np.array(y_0)
        options= {'max_steps': 100e3}
        sol_ = ode('cvode', 
                    ZymFerm_wrapper_for_odes, 
                    old_api=False, **options).solve(np.linspace(t0, tf, N+1), y0)
        #sol_ = odeint(ZymFerm_wrapper_for_odes, np.linspace(t0, tf, N+1), y0, method='bdf')
        sol = sol_.values
    return(sol)
def constraint_lower(x):
    if all([xi > DIN_LOWER_BOUND for xi in x]) == True:
        return 0.
    else:
        return -1 
def constraint_upper(x):
    if all([xi < DIN_UPPER_BOUND for xi in x]) == True:
        return 0.
    else:
        return -1
def perform_optimization(t0=T0, tf=TF, nint=NINT, x0=0., is_stochastic=False, open_loop=True):
    print("-> Performing optimization.")
    bounds = ([DIN_LOWER_BOUND]*NINT, [DIN_UPPER_BOUND]*NINT)
    opt_problem=optimization_problem(bounds, fermentation_cost, [t0, tf, nint, x0, is_stochastic])
    prob=pg.problem(opt_problem)
    if USE_HEURISTIC is False:
        '''
        bounds = Bounds(np.full(NINT, DIN_LOWER_BOUND, dtype=np.float64), 
                        np.full(NINT, DIN_UPPER_BOUND, dtype=np.float64))
        cons = LinearConstraint(np.eye(NINT), DIN_LOWER_BOUND, DIN_UPPER_BOUND)

        result = minimize(fermentation_cost,
                        x0=rng.choice(np.linspace(DIN_LOWER_BOUND, 
                                                    DIN_UPPER_BOUND,
                                                    10), NINT),#np.full(NINT, 0.25),   
                        bounds=bounds,
                        constraints=[{'type':'eq', 'fun':constraint_lower},
                                     {'type':'eq', 'fun':constraint_upper}],
                        method="SLSQP",
                        #options={'catol':1e-8},
                        #popsize=3,
                        #seed=12345,
                      callback=calback_fun)
        '''

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
    if open_loop is True:
        print("\n -> Running simulation with obtained manipulated variable profile.")
        #input("...")
        sol = solve_fermentation(NINT, 
                                T0, 
                                TF, 
                                result_x, 
                                [Xv0, Xnv0, Xd0, P0, S0], 
                                1., 
                                1., 
                                last_run=True)

        plot_solution(sol, result_x)
    return result_x

def plot_solution(sol, Din_t):
    print("\n Obtained solution (y)= ", sol.y)
    print("\n Obtained solution (t)= ", sol.t)
    
    SHOULD_USE_SCIPY = True

    t_ = sol.t
    #IF USING SCIPY ODEINT
    if SHOULD_USE_SCIPY is True:
        Xv = sol.y[0]
        S = sol.y[4]
        Xnv = sol.y[1]
        P = sol.y[3]
    #IF USING SCIKIT.ODES ODEINT
    if SHOULD_USE_SCIPY is False:
        Xv = sol.y[:,0]
        S = sol.y[:, 4]
        Xnv = sol.y[:, 1]
        P = sol.y[:, 3]

    #Filter out residual S <0
    S[S<0] = 0.
    #print("x=",X)
    #print("t=", sol.t)
    #print("x=",sol.y[:,0])

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
    Din_t = np.concatenate(([Din_t[0]], Din_t))
    axs[1].step(t_, Din_t, label=r"$D_{in}\,(t)$", linestyle='-', lw=1.2, color='purple')
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
    fig.savefig("solution_OCP_with_control_var_OLO.png")
    plt.clf()

#perform_optimization()