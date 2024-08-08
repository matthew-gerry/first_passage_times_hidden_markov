'''
fpt_fitting.py

Fit the first passage time distribution as determined by a rate matrix with a specified "leak state" to a sum of exponentials. Evaluate the quality of such a fit if the number of terms in the model function is less than the number of states in the network.

Matthew Gerry, August 2024
'''

import numpy as np
import matplotlib.pyplot as plt

from _utils import *

# Define a rate matrix = here we consider a "controlled reaction": two local "neighbourhoods" of three sites, from each of which there is a relatively slow transition to a counterpart site in the other neighbourhood
kfast = 10
kslow = 1

N = 6 # Total number of sites

L = np.diagflat(kslow*np.ones(N-3), 3) + np.diagflat(kslow*np.ones(N-3), -3) # Slow transition rates
L += np.diagflat(kfast*np.ones(N-1), 1) + np.diagflat(kfast*np.ones(N-1), -1) # Fast transition rates
L[int(N/2-1), int(N/2)] = 0; L[int(N/2), int(N/2-1)] = 0 # The previous line added two elements that shouldn't be there--remove
L -= np.diagflat(sum(L)) #

# Calculate the first passage time distribution based on this rate matrix
dt = 0.01; tmax = 20
time = np.arange(0, tmax, dt)

start_site = 0; leak_site = N-1 # Choice of start and leak sites
k0 = 0.5 # Leak rate
FPTD = first_passage_time_dist(L, start_site, leak_site, k0, time)

fit_result_2 = fit_exponentials(time, FPTD, 2, p0=[0.1, -0.5, -0.1])
fit_result_6 = fit_exponentials(time, FPTD, 6, p0=[0.1, 0.1, -0.1, -0.1, -0.1, -1, -1, -1, -1, -0.5, -0.1])

theta2 = fit_result_2["x"]
theta6 = fit_result_6["x"]

fit2 = f_model(time, theta2[:1], theta2[1:])
fit6 = f_model(time, theta6[:5], theta6[5:])

print("MSE with N=2: " + str(MSE_exponential_fit(time, FPTD, theta2)))
print("MSE with N=6: " + str(MSE_exponential_fit(time, FPTD, theta6)))

plt.plot(time, FPTD, '.', label = 'Exact')
plt.plot(time, fit2, label = "N=2")
plt.plot(time, fit6, label = "N=6")
plt.legend()
plt.show()

# IN PROGRESS: COMPARE THE FIT PARAMETERS TO THE TRUE EIGENVALUES, INVESTIGATE WHAT FEATURES OF THE DYNAMICS ARE PICKED OUT BY THE REDUCED DIMENSIONALITY FIT