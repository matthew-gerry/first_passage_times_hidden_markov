'''
chain_fpt.py

Consider the case of transport along one-dimensional chains, calculate and plot the first few moments of the FPT distribution.

Begin by comparing a uniform chain to a modular chain which should have analogous averaged properties as per Phys Rev E 108, 024135 (2023).  
'''

import numpy as np
import matplotlib.pyplot as plt

from _utils import *
import rate_matrices as rm

# Set up time array
dt = 0.03; tmax = 150
time = np.arange(0, tmax, dt)

# We will analyze how FPT statistics vary with N so set a range of chain lengths to analyze
Nrange = range(2, 7)
k = 1 # Rate for site-site transitions in the uniform chain
kA = 2/3; kB = 2 # Rates in the two different domains of the modular chain
m = 1 # Segment length parameter for modular chains
b = 0 # bias
# k0 = 1 # Leak rate

FPTD_N_uni = np.zeros([len(Nrange), len(time)]) # Array to store full FPTDs at every N for uniform chains
FPTD_N_mod = np.zeros([len(Nrange), len(time)]) # Array to store full FPTDs at every N for modular chains
FPTD_N_imp = np.zeros([len(Nrange), len(time)]) # Array to store full FPTDs at every N for impurity chains

 # Arrays to store first and second moments of FPTD at each N for each chain type
M1_N_uni, M2_N_uni, M3_N_uni = np.zeros(len(Nrange)), np.zeros(len(Nrange)), np.zeros(len(Nrange))
M1_N_mod, M2_N_mod, M3_N_mod = np.zeros(len(Nrange)), np.zeros(len(Nrange)), np.zeros(len(Nrange))
M1_N_imp, M2_N_imp, M3_N_imp = np.zeros(len(Nrange)), np.zeros(len(Nrange)), np.zeros(len(Nrange))

# Calculations for uniform chains
for i in range(len(Nrange)):
    N = Nrange[i]

    L = rm.L_uniform(N, k, b)
    start_site = 0; leak_site = N - 1 # Choice of start and leak sites

    # Calculate the exact FPTD for this rate matrix and choice of start/leak sites    
    FPTD = first_passage_time_dist(L, start_site, leak_site, k, time)
    FPTD_N_uni[i, :] = FPTD

    # Save the first two moments
    M1_N_uni[i] = fpt_moments(time, FPTD, 1)
    M2_N_uni[i] = fpt_moments(time, FPTD, 2)
    M3_N_uni[i] = fpt_moments(time, FPTD, 3)


# Calculations for modular chains
for i in range(len(Nrange)):
    N = Nrange[i]

    L = rm.L_modular(N, m, kA, kB, b)
    start_site = 0; leak_site = N - 1 # Choice of start and leak sites

    # Calculate the exact FPTD for this rate matrix and choice of start/leak sites
    if (N+1)//m%2==0:
        k_leak = kA
    elif (N+1)//m%2==1:
        k_leak = kB    
    FPTD = first_passage_time_dist(L, start_site, leak_site, k_leak, time)
    FPTD_N_mod[i, :] = FPTD

    # Save the first two moments
    M1_N_mod[i] = fpt_moments(time, FPTD, 1)
    M2_N_mod[i] = fpt_moments(time, FPTD, 2)
    M3_N_mod[i] = fpt_moments(time, FPTD, 3)


# Calculations for chains with an impurity
for i in range(len(Nrange)):
    N = Nrange[i]

    L = rm.L_impurity(N, 0, k, 0.5*k, b) # Impurity placed at site 0 so it works for all site lengths
    start_site = 0; leak_site = N - 1 # Choice of start and leak sites

    # Calculate the exact FPTD for this rate matrix and choice of start/leak sites    
    FPTD = first_passage_time_dist(L, start_site, leak_site, k, time)
    FPTD_N_imp[i, :] = FPTD

    # Save the first two moments
    M1_N_imp[i] = fpt_moments(time, FPTD, 1)
    M2_N_imp[i] = fpt_moments(time, FPTD, 2)
    M3_N_imp[i] = fpt_moments(time, FPTD, 3)


# Calculate randomness parameter
r_uni = (M2_N_uni - M1_N_uni**2)/M1_N_uni**2
r_mod = (M2_N_mod - M1_N_mod**2)/M1_N_mod**2
r_imp = (M2_N_imp - M1_N_imp**2)/M1_N_imp**2

sk_uni = (M3_N_uni - M1_N_uni**3)/M1_N_uni**3
sk_mod = (M3_N_mod - M1_N_mod**3)/M1_N_mod**3
sk_imp = (M3_N_imp - M1_N_imp**3)/M1_N_imp**3


# Plot the first passage time distributions at each N as a function of time for each chain type
plt.subplot(1,3,1)
for i in range(len(Nrange)):
    plt.plot(time, FPTD_N_uni[i, :], '-', label = "$N = $" + str(Nrange[i]))
plt.legend()
plt.subplot(1,3,2)
for i in range(len(Nrange)):
    plt.plot(time, FPTD_N_mod[i, :], '-', label = "$N = $" + str(Nrange[i]))
plt.subplot(1,3,3)
for i in range(len(Nrange)):
    plt.plot(time, FPTD_N_imp[i, :], '-', label = "$N = $" + str(Nrange[i]))
plt.show()

# Show a scatter plot of the first two moments
fig, ax = plt.subplots()
ax.scatter(M1_N_uni, r_uni, label="Uniform")
for i, N in enumerate(Nrange):
    ax.annotate("N = "+str(N), (M1_N_uni[i], r_uni[i]))
ax.scatter(M1_N_mod, r_mod, label="Modular")
ax.scatter(M1_N_imp, r_imp, label="Impurity")
ax.set_xlabel(r"$\langle t\rangle$")
# ax.set_ylabel(r"$\langle t^2\rangle$")
ax.set_ylabel("$r$")
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.scatter(M1_N_uni, sk_uni, label="Uniform")
for i, N in enumerate(Nrange):
    ax.annotate("N = "+str(N), (M1_N_uni[i], sk_uni[i]))
ax.scatter(M1_N_mod, sk_mod, label="Modular")
ax.scatter(M1_N_imp, sk_imp, label="Impurity")
ax.set_xlabel(r"$\langle t\rangle$")
ax.set_ylabel(r"$\langle\langle t^3\rangle\rangle$")
plt.legend()
plt.show()