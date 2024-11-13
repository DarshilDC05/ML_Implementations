import numpy as np
import matplotlib.pyplot as plt
import math

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0
    z_i = np.dot(X, w) + b
    f_wb = sigmoid(z_i)
    for i in range(m):
        cost += y[i]*math.log(f_wb[i]) + (1-y[i])*math.log(1-f_wb[i])
    cost = -cost/(m)
    # print(type(cost))
    # print(cost.shape)
    return cost

def compute_derivatives(X, y, w, b):
    m = X.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        dj_dw += (f_wb_i - y[i])*X[i]
        dj_db += f_wb_i - y[i]
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_dw, dj_db; 

def gradient_descent(X, y, w_in, b_in, iterations, alpha, compute_derivatives, compute_cost):
    j_history = []
    p_history = []
    w = w_in
    b = b_in
    for i in range(iterations + 1):
        dj_dw, dj_db = compute_derivatives(X, y, w, b)
        w = w - (alpha*dj_dw)
        b = b - (alpha*dj_db)

        if(i <= 100000):
            j_history.append(compute_cost(X, y, w, b))
            p_history.append([w, b])

        if(i % math.ceil(iterations/10) == 0):
            print(f"Iteration " , i , ". Cost = " , j_history[-1] , " [w, b] = " , p_history[-1])

    return w, b, j_history, p_history





x_train = np.array([1,2,3,4,5,6])
y_train = np.array([0,0,0,1,1,1])

w_in = 0
b_in = 0

alpha = 0.01
iterations = 100000

w, b, j_history , p_history = gradient_descent(x_train, y_train, w_in, b_in, iterations, alpha, compute_derivatives, compute_cost)

a = float(input("Enter the input : "))
ans = w*a + b

if ans>0.5:
    print(1)
else:
    print(0)

x_axis = np.arange(0.01, 7.01, 0.01)
y_axis = np.zeros(700)


y_axis = sigmoid(np.dot(x_axis, w) + b)

plt.scatter(x_train, y_train)
plt.plot(x_axis, y_axis)
plt.show()