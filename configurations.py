#CONFIGURATION PARAMETERS
#------------------------------

#Mustafa et al (2014) -  Structured Mathematical Modeling, Bifurcation, and Simulation for the Bioethanol Fermentation Process Using Zymomonas mobilis. doi:10.1021/ie402361b 

##### General parameters #####

#Use SCIPY (True) or Scikit.Odes (False) for IVP solving
SHOULD_USE_SCIPY=False
#Use heuristic methods (True) or local optimization (False) for NLP incidental problem solving
USE_HEURISTIC=True
#Control verbosity of the output
IS_VERBOSE=False
#Set random seed for reproducibility
RANDOM_SEED=12345
#Initial time
T0=0.
#End time
TF=50.
IS_STOCHASTIC=1.
#Control and prediction time interval
DT = 1.
#Number of time intervals
NINT=int((TF-T0)/DT)
#Set-point for product (bioethanol)
PSET=65.
#Lower bound for the manipulated variable (D_in)
DIN_LOWER_BOUND=0.
#Upper bound for the manipulated variable (D_in)
DIN_UPPER_BOUND=0.1
#Number of generations for optimization
N_GEN=500
#Population size for metaheuristic algorithms
POP_SIZE=100
#Yx=1.
#ms=2.16
#k1=16.

##### Model parameters #####
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

S0 = 150.
P0 = 0.
Xnv0 = 0.
Xd0 = 0.
Xv0 = 2.5

#Deviation for uniform sampling
UNIFORM_DEV  = .2
YXP_MEAN = 0.375
MP_MEAN = 1.1
MU_MAX_MEAN = 0.23
MU_MAXD_MEAN = 0.22
