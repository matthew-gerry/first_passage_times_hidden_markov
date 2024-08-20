'''
_utils_advanced.py

Functions involved in the first passage time statistics analysis that are not needed by every script in the project and that require tensorflow, which is slow to import. Separated into a distinct utilities file so they can be loaded only when needed.

Matthew Gerry, August 2024
'''

from tensorflow import Variable, GradientTape, cast, float32
import tensorflow.math as tfmath
import tensorflow.keras.optimizers as ko


def fit_exponentials_adam(t, y, N, x0=None, num_steps=200, learning_rate=0.1, epsilon=1e-8, weight_short=False):
    '''
    CARRY OUT OPTIMIZATION USING THE ADAM ALGORITHM TO FIT THE DISTRIBUTION TO A SUMMATION OF EXPONENTIALS.

    INCLUDES OPTION TO WEIGHT SHORT TIME VALUES MORE HEAVILY
    '''
    
    # Set the initial guess to a list of ones if not specified
    if x0==None:
        x0 = (2*N - 1)*[1.]
    elif len(x0) != 2*N - 1:
        raise ValueError("Initial guess must include 2*N - 1 parameter values.")
    
    # Define the cost function - sum of square error
    def cost(theta):
        coefficients = theta[:N-1]
        decay_rates = theta[N-1:]
        ft = f_model(t, coefficients, decay_rates)
        res = ft - y # Residuals

        if weight_short:
            kernel = 50.0*tfmath.exp(-0.2*t) + 1.0
            kernel = cast(kernel, float32)
            weighted_res = tfmath.multiply(res, kernel)
            output = tfmath.reduce_sum(tfmath.pow(weighted_res, 2))
        else:
            output = tfmath.reduce_sum(tfmath.pow(res, 2))
        
        return output

    # Initialize Adam optimizer
    optimizer = ko.Adam(learning_rate=learning_rate, epsilon=epsilon,)
    # Set starting point for optimization
    theta = Variable(x0) # x0 should is a list of 2N-1 values
    
    # Perform the optimization
    for i in range(num_steps):
        with GradientTape() as tape:
            # Calculate the value of the cost function and its gradient, store values
            cost_value = cost(theta)
            gradient = tape.gradient(cost_value, theta)

        #  Use the Adam optimizer to update the value of theta
        optimizer.apply_gradients([(gradient, theta)])
    
    x = list(theta.numpy()) # Optimized parameter values
    costval = cost(list(theta.numpy())) # Cost function at the end of optimization

    # Output the parameter values as the end of optimization as well as the cost function
    return dict(zip(["x", "cost"], [x, costval]))
