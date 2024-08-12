'''
Useful functions towards studying the relationship between first passage time distributions and Markov process network structure.

Matthew Gerry, August 2024
'''

import numpy as np

from scipy.linalg import expm
from scipy.optimize import least_squares

from tensorflow import Variable, GradientTape
from tensorflow.math import pow, reduce_sum
import tensorflow.keras.optimizers as ko


def first_passage_time_dist(L, start, leak, k0, time):
    '''
    FIRST PASSAGE TIME DISTRIBUTION GIVEN THE RATE MATRIX
    
    THE START AND LEAK SITES MUST BE SPECIFIED (INDEXING STARTING AT ZERO).
    
    THE ARGUMENT L MUST BE A RATE MATRIX OF THE HIDDEN NETWORK WITH ALL COLUMNS SUMMING TO ZERO.
    '''

    # Substract the leak rate k0 from the diagonal element associated with the leak site
    L[leak, leak] -= k0

    N = np.shape(L)[0] # Size of the hidden network

    # Distributions associated with the system at the start and leak site, respectively
    p_start, p_leak = np.zeros(N), np.zeros(N)
    np.put(p_start, start, 1)
    np.put(p_leak, leak, 1)

    # Pre-allocate array to hold FPTD values
    FPTD = np.zeros(len(time))
    for i in range(len(time)):
        surv = expm(L*time[i]) # Survivial matrix

        # FPTD representing -d/dt of survival probability given start and leak sites
        FPTD[i] = k0*p_leak.dot(surv.dot(p_start.transpose()))
    
    return FPTD


def f_model(t, coefficients, decay_rates):
    '''
    MODEL FUNCTION FOR FPTD - SUMMATION OF N EXPONENTIAL DECAY TERMS
    
    ARGUMENTS INCLUDE AN ARRAY OF N DECAY RATES (ONE FOR EACH TERM) AND N-1 COEFFICIENTS (THE LAST COEFFICIENT IS FIXED BY NORMALIZATION).
    '''

    if len(decay_rates) - len(coefficients) != 1:
        raise ValueError("decay_rates must include exactly one more value than coefficients")
    else:
        # Determine the last coefficient as fixed by normalization
        last_coeff = -decay_rates[-1]*(1 + sum([x/y for x,y in zip(coefficients, decay_rates[:-1])]))
        
        # Output is the sum of exponential decay terms
        return sum([x*np.exp(y*t) for x,y in zip(coefficients, decay_rates[:-1])]) + last_coeff*np.exp(decay_rates[-1]*t)


def fit_exponentials(t, y, N, x0=None, gtol=None, num_guesses=1):
    '''
    USE scipy_optimize.least_squares TO FIT DATA TO A SUM OF EXPONENTIAL TERMS
    
    INPUT AN INITIAL GUESS AND OPTIONALLY TRY A SPECIFIED NUMBER OF ADDITIONAL RANDOM INITIAL GUESSES AND SELECT THE BEST ONE
    '''
    
    # Set the initial guess to a list of ones if not specified
    if x0==None:
        x0 = (2*N - 1)*[1]
    elif len(x0) != 2*N - 1:
        raise ValueError("Initial guess must include 2*N - 1 parameter values.")

    # Define a function to calculate the residuals
    def residuals(theta):
        coefficients = theta[:N-1]
        decay_rates = theta[N-1:]
        ft = f_model(t, coefficients, decay_rates)
        return ft - y
    
    result = least_squares(residuals, x0, gtol=gtol)

    # Carry out optimizaton additional times with randomized intial guesses
    if num_guesses > 1:
        for i in range(num_guesses-1):
            coeff_rand_guess = 2*np.random.rand(N - 1) - 1 # Guess random values between -1 and 1 for coefficients
            rate_rand_guess = -5*np.random.rand(N) # Guess random values -5 and 0 for decay rates
            rand_guess = np.append(coeff_rand_guess, rate_rand_guess)

            # Carry out optimization with randomly guessed initial values
            result_temp = least_squares(residuals, rand_guess, gtol=gtol)

            if result_temp["cost"] < result["cost"]:
                result = result_temp # If the random guess minimizes the cost function more effectively, replace the old result

    return result


def fit_exponentials_adam(t, y, N, x0=None, num_steps=200, learning_rate=0.1, epsilon=1e-8):
    '''
    CARRY OUT OPTIMIZATION USING THE ADAM ALGORITHM TO FIT THE DISTRIBUTION TO A SUMMATION OF EXPONENTIALS
    '''
    
    # Set the initial guess to a list of ones if not specified
    if x0==None:
        x0 = (2*N - 1)*[1.]
    elif len(x0) != 2*N - 1:
        raise ValueError("Initial guess must include 2*N - 1 parameter values.")
    
    # Define the cost function - sum of square error
    def sse(theta):
        coefficients = theta[:N-1]
        decay_rates = theta[N-1:]
        ft = f_model(t, coefficients, decay_rates)
        return reduce_sum(pow(ft - y, 2))

    # Initialize Adam optimizer
    optimizer = ko.Adam(learning_rate=learning_rate, epsilon=epsilon,)
    # Set starting point for optimization
    theta = Variable(x0) # x0 should is a list of 2N-1 values
    
    # Perform the optimization
    for i in range(num_steps):
        with GradientTape() as tape:
            # Calculate the value of the cost function and its gradient, store values
            cost_value = sse(theta)
            gradient = tape.gradient(cost_value, theta)

        #  Use the Adam optimizer to update the value of theta
        optimizer.apply_gradients([(gradient, theta)])
    
    x = list(theta.numpy()) # Optimized parameter values
    cost = sse(list(theta.numpy())) # Cost function at the end of optimization

    # Output the parameter values as the end of optimization as well as the cost function
    return dict(zip(["x", "cost"], [x, cost]))


def eig_L(L, leak, k0):
    '''
    EIGENVALUES AND NORMALIZED EIGENVECTORS OF A GIVEN RATE MATRIX INCLUDING THE LEAK
    '''

    # Substract the leak rate k0 from the diagonal element associated with the leak site
    L[leak, leak] -= k0

    N = np.shape(L)[0] # Size of the hidden network

    # Numpy function to calculate eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(L)

    # Create a matrix whose columns will hold the normalized eigenvectors (elements sum to unity)
    V = np.zeros([N,N])
    for i in range(N):
        V[:,i] = eigvecs[:,i]/sum(eigvecs[:,i])

    # Diagonal matrix whose elements are the eigenvalues
    D = np.diag(eigvals)

    return V, D


def eig_decomposition(p, V):
    '''
    DECOMPOSE A POPULATION DISTRIBUTION p IN TERMS OF THE COLUMNS OF V (NORMALIZED EIGENVECTORS)
    
    RETURNS BOTH THE COEFFICIENTS OF THE DECOMPOSITION AND THE 
    '''

    N = np.shape(V)[0] # Size of the hidden network

    if sum(V).any() != 1 or np.shape(V)[0] != np.shape(V)[1]:
        raise ValueError("V must be a square matrix of eigenvectors whose columns each sum to unity")
        
    components = p.dot(V) # Components of p along each eigenvector
    gramian = V.transpose().dot(V) # Gramian matrix (matrix of dot products of eigenvectors)

    return np.linalg.inv(gramian).dot(components)
