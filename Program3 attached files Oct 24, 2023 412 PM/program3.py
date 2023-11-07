from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange
import numpy as np
import re
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
from pylab import plot, show, xlabel, ylabel


def read_data(filename):
    with open(filename, 'r') as f:
        data = []
        for line in f:
            items = re.split('[ ,]', line.strip())
            float_items = []
            for item in items:
                try:
                    float_items.append(float(item))
                except ValueError:
                    pass  # or handle non-numeric data as required
            data.append(float_items)  # <-- Moved inside the loop

    return np.array(data)

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    sqrErrors = (predictions - y) ** 2
    J = 1 / (2 * m) * np.sum(sqrErrors)
    return J

def gradient_descent(X, y, theta, alpha, num_iterations):
    m = len(y)
    J_history = []
    
    for i in range(num_iterations):
        error = X.dot(theta) - y
        temp = theta.copy()  # Copy current theta values
        for j in range(len(theta)):
            temp[j] -= (alpha / m) * np.sum(error * X[:, j])
        theta = temp  # Update all theta values simultaneously
        J_history.append(compute_cost(X, y, theta))
    return theta, J_history

def main(filename):
    # Load the data
    data = read_data(filename)
    X = data[:, :-1]
    y = data[:, -1]
    
    # Add a column of ones to X for the bias term
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    
    # Set initial values for parameters and hyperparameters
    theta = np.array([0.0 for _ in range(X.shape[1])])
    alpha = 0.01
    iterations = 100
    
    # Run gradient descent
    new_theta, J_history = gradient_descent(X, y, theta, alpha, iterations)
    

	#Init Theta and Run Gradient Descent
	#theta = zeros(shape=(3, 1))


	#theta, J_history = gradientDescent(it, y, theta, alpha, iterations)

    # Print results
    print("Initial weights:", theta)
    print()
    print("Final weights after", iterations, "iterations:", new_theta)
    print()
    print("Cost function values over iterations:", J_history)
    print()
    plot(arange(iterations), J_history)
    xlabel('Iterations')
    ylabel('Cost Function')
    show()
    print()
if __name__ == "__main__":
    main("prog3_input1.txt")
    #main("prog3_input2.txt")