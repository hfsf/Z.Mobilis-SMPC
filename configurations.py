#Configurations for problem solving
'''
Yx=1.
YX_MEAN = 1.
YX_STD = .1 * YX_MEAN
Ysx=0.0244
Ypx=0.0526
Ks=0.5
ms=2.16
MS_MEAN = 2.16
MS_STD = .1 * MS_MEAN
mp=1.1
#Din=0.
DIN_LOWER_BOUND=0.001
DIN_UPPER_BOUND=0.1
k1=16.
K1_MEAN = 16
K1_STD = .1 * K1_MEAN
k2=0.497
k3=0.0038
X0=0.08
S0=150.3
P0=4.0
E0=0.25
Vf=0.003
T0=0.
TF=30.
IS_STOCHASTIC=1.
NINT=100
PSET=65.
S0_MEAN=150.3
S0_STDDEV=0.#0.005

YX_DEV = 0.
MS_DEV = 0.05
K1_DEV = 0.15
YX_LOWER = (1.-YX_DEV)*YX_MEAN
YX_UPPER = (1.+YX_DEV)*YX_MEAN
MS_LOWER = (1.-MS_DEV)*MS_MEAN
MS_UPPER = (1.+YX_DEV)*MS_MEAN
K1_LOWER = (1.-K1_DEV)*K1_MEAN
K1_UPPER = (1.+K1_DEV)*K1_MEAN 
'''
#------------------------------

#Mustafa et al (2014) -  Structured Mathematical Modeling, Bifurcation, and Simulation for the Bioethanol Fermentation Process Using Zymomonas mobilis. doi:10.1021/ie402361b 

T0=0.
TF=20.
IS_STOCHASTIC=1.
NINT=100
PSET=65.
DIN_LOWER_BOUND=0.0
DIN_UPPER_BOUND=0.1
#Yx=1.
#ms=2.16
#k1=16.

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

MU_MAX_MEAN = 0.23
MU_MAX_LOWER = 0.8*MU_MAX_MEAN
MU_MAX_UPPER = 1.2*MU_MAX_MEAN
ALPHA_MEAN = 1.74
ALPHA_LOWER = 0.8*ALPHA_MEAN
ALPHA_UPPER = 1.2*ALPHA_MEAN
S0_MEAN = S0
S0_LOWER = 0.8*S0_MEAN
S0_UPPER = 1.2*S0_MEAN