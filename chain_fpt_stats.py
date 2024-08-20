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
dt = 0.01; tmax = 100
time = np.arange(0, tmax, dt)

# We will analyze how FPT statistics vary with N so set a range of chain lengths to analyze
Nrange = range(2, 7)
k = 1 # Rate for site-site transitions in the uniform chain
kA = 2/3; kB = 2 # Rates in the two different domains of the modular chain
m = 1 # Segment length parameter for modular chains
b = 0 # bias
k0 = 1 # Leak rate

FPTD_N_uni = np.zeros([len(Nrange), len(time)]) # Array to store full FPTDs at every N for uniform chains
FPTD_N_mod = np.zeros([len(Nrange), len(time)]) # Array to store full FPTDs at every N for modular chains

 # Arrays to store first and second moments of FPTD at each N for each chain type
M1_N_uni, M2_N_uni = np.zeros(len(Nrange)), np.zeros(len(Nrange))
M1_N_mod, M2_N_mod = np.zeros(len(Nrange)), np.zeros(len(Nrange))

# Calculations for uniform chains
for i in range(len(Nrange)):
    N = Nrange[i]

    L = rm.L_uniform(N, k, b)
    start_site = 0; leak_site = N - 1 # Choice of start and leak sites

    # Calculate the exact FPTD for this rate matrix and choice of start/leak sites    
    FPTD = first_passage_time_dist(L, start_site, leak_site, k0, time)
    FPTD_N_uni[i, :] = FPTD

    # Save the first two moments
    M1_N_uni[i] = fpt_moments(time, FPTD, 1)
    M2_N_uni[i] = fpt_moments(time, FPTD, 2)


# Calculations for modular chains
for i in range(len(Nrange)):
    N = Nrange[i]

    L = rm.L_modular(N, m, kA, kB, b)
    start_site = 0; leak_site = N - 1 # Choice of start and leak sites

    # Calculate the exact FPTD for this rate matrix and choice of start/leak sites    
    FPTD = first_passage_time_dist(L, start_site, leak_site, k0, time)
    FPTD_N_mod[i, :] = FPTD

    # Save the first two moments
    M1_N_mod[i] = fpt_moments(time, FPTD, 1)
    M2_N_mod[i] = fpt_moments(time, FPTD, 2)

# Plot the first passage time distributions at each N as a function of time for each chain type
plt.subplot(1,2,1)
for i in range(len(Nrange)):
    plt.plot(time, FPTD_N_uni[i, :], '-', label = "$N = $" + str(Nrange[i]))
plt.legend()
plt.subplot(1,2,2)
for i in range(len(Nrange)):
    plt.plot(time, FPTD_N_mod[i, :], '-', label = "$N = $" + str(Nrange[i]))
plt.show()

# Show a scatter plot of the first two moments
fig, ax = plt.subplots()
ax.scatter(M1_N_uni, M2_N_uni, label="Uniform")
for i, N in enumerate(Nrange):
    ax.annotate("N = "+str(N), (M1_N_uni[i], M2_N_uni[i]+4))
ax.scatter(M1_N_mod, M2_N_mod, label="Modular")
ax.set_xlabel(r"$\langle t\rangle$")
ax.set_ylabel(r"$\langle t^2\rangle$")
plt.show()