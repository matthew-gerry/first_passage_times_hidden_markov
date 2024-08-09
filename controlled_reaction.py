'''
fpt_fitting.py

Fit the first passage time distribution as determined by a rate matrix with a specified "leak state" to a sum of exponentials. Evaluate the quality of such a fit if the number of terms in the model function is less than the number of states in the network.

Matthew Gerry, August 2024
'''

import numpy as np
import matplotlib.pyplot as plt

from _utils import *

# Define a rate matrix = here we consider a "controlled reaction": two local "neighbourhoods" of three sites, from each of which there is a relatively slow transition to a counterpart site in the other neighbourhood
# Network structure - sites on the same horizontal line are in a common "neighbourhood"
#    0 - 1 - 2
#    |   |   |
#    3 - 4 - 5

# Define rates for the fast and slow processes, respectively
kfast = 10
kslow = 1

N = 6 # Total number of sites

L = np.diagflat(kslow*np.ones(N-3), 3) + np.diagflat(kslow*np.ones(N-3), -3) # Slow transition rates
L += np.diagflat(kfast*np.ones(N-1), 1) + np.diagflat(kfast*np.ones(N-1), -1) # Fast transition rates
L[int(N/2-1), int(N/2)] = 0; L[int(N/2), int(N/2-1)] = 0 # The previous line added two elements that shouldn't be there--remove

impurities = False # Toggle whether we want to introduce impurities to the rate matrix
if impurities:
    # Add some impurities to see if the quality of the fit is affected at all
    L[4, 1] = 0.1; L[5, 2] = 0.6 # Reduce speed to a few of the inter-neighbourhood transitions in one direction
    L[1, 4] = 8 # Speed up one in the other direction
    L[2, 1] = 0.5; L[5, 4] = 0.6; L[4, 5] = 0.6 # Reduce some speeds within the neighbourhoods too to muddle the timescale distinction

L -= np.diagflat(sum(L)) # Set diagonal elements to ensure all columns sum to zero (the leak is accounted for later)

print("Rate matrix:\n", L)

# Calculate the first passage time distribution based on this rate matrix
dt = 0.01; tmax = 15
time = np.arange(0, tmax, dt)

start_site = 0; leak_site = N-1 # Choice of start and leak sites
k0 = 0.5 # Leak rate

# Calculate the exact FPTD for this rate matrix and choice of start/leak sites
FPTD = first_passage_time_dist(L, start_site, leak_site, k0, time)

# Fit the FPTD to sums of two and six exponentials
fit_result_2 = fit_exponentials(time, FPTD, 2, p0=[0.1, -0.5, -0.1])
fit_result_6 = fit_exponentials(time, FPTD, 6, p0=[0.1, 0.1, 0.1, -0.1, -0.1, -10, -1, -1, -0.5, -0.5, -0.1])

# Extract the parameter values estimated by the fits
theta2 = fit_result_2["x"]; theta6 = fit_result_6["x"]

# Pass the time values through the model functions with the fit parameters applied
fit2 = f_model(time, theta2[:1], theta2[1:])
fit6 = f_model(time, theta6[:5], theta6[5:])

# Display the mean squared error associated with each fit
print("MSE with N=2: " + str(np.mean(np.power(fit_result_2["fun"], 2))))
print("MSE with N=6: " + str(np.mean(np.power(fit_result_6["fun"], 2))))

# Compare the fit parameters to the true eigenvalues in the case of a coarse-grained fit and an exact fit
V, D = eig_L(L, leak_site, k0)
eigs = np.diag(D)

print("True eigenvalues: " + str(eigs))
print("Predicted eigenvalues, N=6: " + str(theta6[5:]))
print("Predicted eigenvalues, N=2: " + str(theta2[1:]))

# Plot the true FPTD with both fits
plt.plot(time, FPTD, '.', label = 'Exact')
plt.plot(time, fit2, label = "N=2")
plt.plot(time, fit6, label = "N=6")
plt.legend()
plt.show()