'''
fpt_fitting.py

Fit the first passage time distribution as determined by a rate matrix with a specified "leak state" to a sum of exponentials. Evaluate the quality of such a fit if the number of terms in the model function is less than the number of states in the network.

Matthew Gerry, August 2024
'''

import numpy as np

# Define a rate matrix = here we consider a "controlled reaction": two local "neighbourhoods" of three sites, from each of which there is a relatively slow transition to a counterpart site in the other neighbourhood
L = np.zeros([6, 6])

# IN PROGRESS