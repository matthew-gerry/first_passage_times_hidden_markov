'''
Useful functions towards studying the relationship between first passage time distributions and Markov process network structure.

Matthew Gerry, August 2024
'''

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

import matplotlib.pyplot as plt

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


def fit_exponentials(x, y, N, p0=None):
    ''' USE scipy_optimize.least_squares TO FIT DATA TO A SUM OF EXPONENTIAL TERMS '''
    
    # Set the initial guess to a list of ones if not specified
    if p0==None:
        p0 = (2*N - 1)*[1]
    elif len(p0) != 2*N - 1:
        raise ValueError("Initial guess must include 2*N - 1 parameter values.")

    # Define a function to calculate the residuals
    def residuals(theta):
        coefficients = theta[:N-1]
        decay_rates = theta[N-1:]
        fx = f_model(x, coefficients, decay_rates)
        return fx - y
    
    result = least_squares(residuals, p0)
    return result


def MSE_exponential_fit(x, y, theta):
    '''
    MEAN SQUARED ERROR OF THE FIT GIVEN A SET THETA OF PARAMETER VALUES AS PREDICTED BY FITTING
    '''

    N = int((len(theta) + 1)/2)

    # Output should be a numpy array (ensure x is a numpy array)
    fit_result = f_model(x, theta[:N-1], theta[N-1:])


    return np.mean(np.power(fit_result - y, 2))


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






time = np.arange(0, 30, 0.01)

coeff_true = [0.85]
decay_rates_true = [-1.65, -0.21]

y = f_model(time, coeff_true, decay_rates_true) + 0.1*np.random.rand(time.shape[0])

result = fit_exponentials(time, y, 2, p0=[0.1, -1, -0.1])
print(result)

