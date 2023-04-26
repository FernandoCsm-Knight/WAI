import numpy as np

#################################################
# Random data generation ------------------------
# 
# The data is stored in a list of touples.
# A touple is composed by an array of float64 and
# a float64 number that represents the expected
# output. Both elements in touple are randomly
# generated. 
#################################################

def data(w, d=1):
    arr = []
    for i in range(5000):
        x = np.random.randn(d)
        y = w.dot(x) + np.random.randn()
        arr.append((x, y))
    return arr

std_w = np.random.randn(6)
dim = len(std_w)

arr = data(std_w, dim)
fi = [x for x, y in arr]

#################################################
# Predictor -------------------------------------
#  
# A predictor is a function that maps an element
# in a specific output using a weight vector.
# To calculate the output a scalar product or dot
# operation is computed and its sign is returned.
# 
#################################################

def predictor(w, fi):
    num = w.dot(fi)
    if num > 0:
        return 1
    elif num < 0:
        return -1
    else:
        return 0

#################################################
# Margin ----------------------------------------
#
# The margin meansures how correct is the 
# prediction stablished with the weight vector w.
#
#################################################

def margin(w, x, y):
    return w.dot(x) * y;

#################################################
# Loss ------------------------------------------
#
# The loss measures how much the prediction will
# loose in precision. To compute that we use the
# least squares principle for approximation.
#
#################################################

def sqrLoss(w, x, y):
    return (w.dot(x) - y)**2

#################################################
# Derivative of square loss ---------------------
# 
# The derivative of the loss indicates if the 
# function is increasing or decreasing, and 
# concurrently indicates how to adjusts the
# weight vector.
#
#################################################

def dsqrLoss(w, x, y):
    return 2*(w.dot(x) - y) * x

#################################################
# Train -----------------------------------------
# 
# To machine learning works, two functions are 
# essentialy needed. Firs one is the function to
# computate the average loss over all the 
# training cases. The other one is to calculate
# the average gradiente based on the derivative 
# of the square loss.
#
#################################################

def trainLoss(w, dat):
    return sum(sqrLoss(w, x, y) for x, y in dat) / len(dat)

def trainGradient(w, dat):
    return sum(dsqrLoss(w, x, y) for x, y in dat) / len(dat)

#################################################
# Machine Learning Process
#################################################

def gradientDescent(lossFn, stepGradient, dat, ndim=1, step=0.01):
    w = np.zeros(ndim)
    
    for i in range(1500):
        loss = lossFn(w, dat)
        gradient = stepGradient(w, dat)
        
        w = w - step * gradient
        if i % 100 == 0: print(f'Weight: {w}, loss: {loss}')

print(f'Gabarito da saída: {std_w}\n')
gradientDescent(trainLoss, trainGradient, arr, dim)
