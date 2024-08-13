'''
chain_fpt.py

Consider the case of transport along one-dimensional chains with side-chains, carry out first-passage time analysis.
'''

import numpy as np
import matplotlib.pyplot as plt

from _utils import *

# Start simple - define the rate matrix for an N-site chain with a leak on the last site, no side-chains or impurities
N=3
# k = 1 # Rate for site-site transitions
k0 = 0.5 # Leak rate

L = np.zeros([N, N])
L[0, 1] = 1.4
L[1, 0] = 0.4
L[1, 2] = 1.0
L[2, 1] = 2.1
# L = np.diagflat(k*np.ones(N-1), -1) # Forward transitions
# L += np.diagflat(k*np.ones(N-1), 1) # Reverse transitions

L -= np.diagflat(sum(L), 0) # Set diagonal elements to ensure all columns sum to zero (the leak is accounted for later)

print("Rate matrix:\n", L)

# Calculate the first passage time distribution based on this rate matrix
dt = 0.01; tmax = 25
time = np.arange(0, tmax, dt)

start_site = 0; leak_site = N - 1 # Choice of start and leak sites

# Calculate the exact FPTD for this rate matrix and choice of start/leak sites
FPTD = first_passage_time_dist(L, start_site, leak_site, k0, time)

# Fit the FPTD to sum of three exponentials
fit_result_3 = fit_exponentials(time, FPTD, 3, x0=[1, 1, -4, -2, -0.1], weight_short=True, num_guesses=10)
# fit_result_3 = fit_exponentials_adam(time, FPTD, 3, x0=[1, 1, -4, -2, -0.1], weight_short=False)

print(fit_result_3)

# Extract the parameter values estimated by the fits
theta3 = fit_result_3["x"]

# Pass the time values through the model functions with the fit parameters applied
fit3 = f_model(time, theta3[:2], theta3[2:])

# Calculate the eigenvalues straight from the rate matrix
V, D = eig_L(L, leak_site, k0)
eigs = np.diag(D)

print("True eigenvalues: " + str(eigs))
print("Predicted eigenvalues, N=3: " + str(theta3[2:]))


# Plot the true FPTD with the fit
plt.plot(time, FPTD, '.', label = 'Exact')
plt.plot(time, fit3, label = "N=3")
plt.legend()
plt.show()

