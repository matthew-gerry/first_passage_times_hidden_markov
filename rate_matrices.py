'''
rate_matrices.py

Generate rate matrices for various useful models of Markov processes on kinetic networks.

All of these functions return a rate matrix with every column summing to zero.
If the network includes leak, that is accounted for in other functions downstream in the analysis.

Matthew Gerry, August 2024
'''

import numpy as np


def L_uniform(N, k, b):
    ''' UNIFORM CHAIN WITH NEAREST NEIGHBOUR HOPPING AND BIAS b '''

    k_rev = k*np.exp(-b)

    L = np.diagflat(k*np.ones(N-1), -1) # Forward transitions
    L += np.diagflat(k_rev*np.ones(N-1), 1) # Reverse transitions

    L -= np.diagflat(sum(L), 0) # Set diagonal elements to ensure all columns sum to zero
    return L


def L_modular(N, m, kA, kB, b):
    ''' MODULAR CHAIN WITH SEGMENTS OF LENGTH m, ALTERNATING BETWEEN FORWARD TRANSITION RATES kA AND kB '''

    L = np.zeros([N, N])

    for i in range(N-1): # Set transition rates for forward and reverse transitions according to position in the chain
        if int(i/m)%2==0:
            L[i+1, i] = kA
            L[i, i+1] = kA*np.exp(-b)
        elif int(i/m)%2==1:
            L[i+1, i] = kB
            L[i, i+1] = kB*np.exp(-b)

    L -= np.diagflat(sum(L), 0) # Set diagonal elements to ensure all columns sum to zero
    return L


def L_impurity(N, n_impurity, k, k_impurity, b):
    ''' A UNIFORM CHAIN WITH ONE IMPURITY--SPECIFY THE SITE FROM WHICH THE TRANSITION TO THE LEFT HAS A DIFFERENT RATE '''

    if n_impurity > N-2:
        raise ValueError("There must be at least one site to the right of the site of the impurity.")
    
    # Start with L of a uniform chain
    L = L_uniform(N, k, b)

    # Replace the rate at the impurity
    L[n_impurity+1, n_impurity] = k_impurity
    L[n_impurity, n_impurity+1] = k_impurity*np.exp(-b)

    # Reset the diagonal elements affected by the insertion of the impurity
    L[n_impurity, n_impurity] = 0; L[n_impurity, n_impurity] = -sum(L[:, n_impurity])
    L[n_impurity+1, n_impurity+1] = 0; L[n_impurity+1, n_impurity+1] = -sum(L[:, n_impurity+1])

    return L


def L_side_chains(N, chain_lengths, k, k_side, b):
    ''' A CHAIN WITH A UNIFORM BACKBONE OF LENGTH N, OFF OF WHICH NON-BRANCHING SIDE-CHAINS MAY EMERGE AT EACH SITE '''

    # chain_lengths is a list of length N, specifying the length of the side chain emerging from each site of the back bone
    # For example, for a "comb" with one-site-long side chains at each site, chain_lengths should be a list with N entries, all ones
    # For a site, n, with no side chain, ensure chain_lengths[n]==0
    # Assume that bias acts along the backbone, suppressing the leftward transitions, but not along the side chains, which contain only symmetric transitions but which may have a distinct transition rate from the backbone

    if len(chain_lengths) != N:
        raise ValueError("Ensure the list of side chain lengths has N entries.")

    # Since the backbone is of length N, and the sum of chain_lengths corresponds to the number of sites in all side chains,
    # the number of columns in the rate matrix is N + sum(chain_lengths).
    # The backbone sites correspond to indices 0 through N-1, the remaining indices are assigned to side chain sites increasing from
    # left to right along the backbone, and away from the backbone

    L = np.zeros([N + sum(chain_lengths), N + sum(chain_lengths)]) # Pre-allocate matrix

    # Set the nearest neighbour hopping rates along the backbone first
    L[0:N, 0:N] += np.diagflat(k*np.ones(N-1), -1) # Forward transitions
    L[0:N, 0:N] += np.diagflat(k*np.exp(-b)*np.ones(N-1), 1) # Reverse transitions

    # Specify the transition rates within the side chains
    for n in range(len(chain_lengths)):
        if chain_lengths[n]==0:
            continue
        else:
            # Note that n here corresponds to the index of the relevant backbone site
            # Ensure that we label side chain sites in a consistent way by calculating what the next index should be for labelling the side chain sites off of each subsequent backbone site as we iterate through chain_lengths
            next_index = N + sum(chain_lengths[0:n]) 

            # Set the transition rates for the bond from the backbone to the first site in the side chain
            L[next_index, n] = k_side; L[n, next_index] = k_side

            # If the side chain has length greater than 1, continue filling in all the rates between sites within the side chain
            if chain_lengths[n] > 1:
                for i in range(chain_lengths[n] - 1):
                    L[next_index + i, next_index + i + 1] = k_side
                    L[next_index + i + 1, next_index + i] = k_side

    L -= np.diagflat(sum(L), 0) # Set diagonal elements to ensure all columns sum to zero
    return L