import numpy as np

def GG2ee(yy, T):
    yy = -4186*yy
    T_prime = 8.314*(T + 273.15)
    temp = np.exp(yy/T_prime)
    return (100-200/(temp+1))

def linear_regression(x, y):
    x = np.array(x)
    y = np.array(y)
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x ** 2)
    sumxy = sum(x * y)
    A = np.mat([[N, sumx], [sumx, sumx2]])
    b = np.array([sumy, sumxy])
    return np.linalg.solve(A, b)