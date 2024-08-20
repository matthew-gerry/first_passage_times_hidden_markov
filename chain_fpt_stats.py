'''
chain_fpt.py

Consider the case of transport along one-dimensional chains with side-chains, carry out first-passage time analysis.
'''

import numpy as np
import matplotlib.pyplot as plt

from _utils import *

# Set up time array
dt = 0.01; tmax = 100
time = np.arange(0, tmax, dt)

# Start simple - define the rate matrix for a N-site chains with a leak on the last site, no side-chains or impurities
# Analyze how FPT statistics vary with N
Nrange = range(2, 7)
k = 1 # Rate for site-site transitions
k0 = 1 # Leak rate

FPTD_N = np.zeros([len(Nrange), len(time)]) # Array to store full FPTDs at every N
M1_N, M2_N = np.zeros(len(Nrange)), np.zeros(len(Nrange)) # Arrays to store first and second moments of FPTD at each N

for i in range(len(Nrange)):
    N = Nrange[i]

    L = np.diagflat(k*np.ones(N-1), -1) # Forward transitions
    L += np.diagflat(k*np.ones(N-1), 1) # Reverse transitions

    L -= np.diagflat(sum(L), 0) # Set diagonal elements to ensure all columns sum to zero (the leak is accounted for later)


    # Calculate the first passage time distribution based on this rate matrix

    start_site = 0; leak_site = N - 1 # Choice of start and leak sites

    # Calculate the exact FPTD for this rate matrix and choice of start/leak sites    
    FPTD = first_passage_time_dist(L, start_site, leak_site, k0, time)
    FPTD_N[i, :] = FPTD

    # Save the first two moments
    M1_N[i] = fpt_moments(time, FPTD, 1)
    M2_N[i] = fpt_moments(time, FPTD, 2)


# Plot the first passage time distributions at each N as a function of time
for i in range(len(Nrange)):
    # Plot the true FPTD with the fit
    plt.plot(time, FPTD_N[i, :], '-', label = "$N = $" + str(Nrange[i]))
    # plt.plot(time, fit3, label = "N=3")
plt.legend()
plt.show()

# Show a scatter plot of the first two moments
plt.scatter(M1_N, M2_N)
plt.xlabel(r"$\langle t\rangle$")
plt.ylabel(r"$\langle t^2\rangle$")
plt.show()