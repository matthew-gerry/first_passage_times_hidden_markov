'''
gated_tree.py

Fit the first passage time distribution as determined by a rate matrix with a specified "leak state" to a sum of exponentials. Evaluate the quality of such a fit if the number of terms in the model function is less than the number of states in the network.

Focus on a network structure with no cycles (tree), where there is a single transition between two neighourhoods of sites (through a gate) that occurs on one timescale, while transitions within each neighbourhood occur on another timescale.

Matthew Gerry, August 2024
'''

import numpy as np
import matplotlib.pyplot as plt

from _utils import *

# Define the rate matrix according to the below network structure. The gate is represented by the double dash
#           2    5
#           |    |       
#   0 - 1 - 3 -- 4 - 6
#                |
#                7

N = 8 # In our case there are eight states
L = np.zeros([8, 8]) # Allocate rate matrix

# Populate rate matrix
kfast = 10
kslow = 1

L[1,0] = kfast; L[3,1] = kfast; L[3,2] = kfast # Left neighbourhood
L[4,5] = kfast; L[4,6] = kfast; L[4,7] = kfast # Right neighbourhood
L[4,3] = kslow # Gate
L += L.transpose() # All transitions symmetric
L -= np.diagflat(sum(L)) # Set diagonal elements to ensure all columns sum to zero (the leak is accounted for later)

print("Rate matrix:\n", L)

# Calculate the first passage time distribution based on this rate matrix
dt = 0.01; tmax = 15
time = np.arange(0, tmax, dt)

start_site = 0; leak_site = N - 1 # Choice of start and leak sites
k0 = 0.5 # Leak rate

# Calculate the exact FPTD for this rate matrix and choice of start/leak sites
FPTD = first_passage_time_dist(L, start_site, leak_site, k0, time)

# Fit the FPTD to sums of two and eight exponentials
fit_result_2 = fit_exponentials(time, FPTD, 2, x0=[0.1, -0.5, -0.1], num_guesses=4)
fit_result_8 = fit_exponentials(time, FPTD, 8, x0=[0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1, -1, -1, -1, -1, -1, -0.5, -0.5, -0.1], num_guesses=10)

print(fit_result_2)
print(fit_result_8)

# Extract the parameter values estimated by the fits
theta2 = fit_result_2["x"]; theta8 = fit_result_8["x"]

# Pass the time values through the model functions with the fit parameters applied
fit2 = f_model(time, theta2[:1], theta2[1:])
fit8 = f_model(time, theta8[:7], theta8[7:])

# Display the mean squared error associated with each fit
# print("MSE with N=2: " + str(np.mean(np.power(fit_result_2["fun"], 2))))
# print("MSE with N=8: " + str(np.mean(np.power(fit_result_8["fun"], 2))))


# Compare the fit parameters to the true eigenvalues in the case of a coarse-grained fit and an exact fit
V, D = eig_L(L, leak_site, k0)
eigs = np.diag(D)

print("True eigenvalues: " + str(eigs))
print("Predicted eigenvalues, N=8: " + str(theta8[7:]))
print("Predicted eigenvalues, N=2: " + str(theta2[1:]))

# Plot the true FPTD with both fits
plt.plot(time, FPTD, '.', label = 'Exact')
plt.plot(time, fit2, label = "N=2")
plt.plot(time, fit8, label = "N=8")
plt.legend()
plt.show()
