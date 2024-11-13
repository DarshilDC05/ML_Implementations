import math, copy
import numpy as np
import matplotlib.pyplot as plt


def compute_cost(x, y, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    cost = 0
    for i in range(m):
        f_wb[i] = w*x[i] + b
        cost += (f_wb[i] - y[i])**2
    total_cost = cost/(2*m)
    return total_cost

def compute_derivatives(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w*x[i] + b
        dj_dw += (f_wb[i] - y[i])*x[i]
        dj_db += f_wb[i] - y[i]
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, iterations, compute_cost, compute_derivative):
    j_history = []
    p_history = []
    w = w_in
    b = b_in

    for i in range(iterations + 1):
        dj_dw, dj_db = compute_derivative(x, y, w, b)
        w = w-(alpha*dj_dw)
        b = b-(alpha*dj_db)

        if i<=10000:
            j_history.append(compute_cost(x, y, w, b))
            p_history.append([w, b])

        if i%math.ceil(iterations/10) == 0:
            print(f"Itr : {i}\t Cost : {j_history[-1]:0.3e}\tdj_dw:{dj_dw:0.3e}    dj_db:{dj_db:0.3e}\t w:{w:0.3e}   b:{b:0.3e}")

    return w, b, j_history, p_history

def regression(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w*x[i] + b
    
    return f_wb


x_train = np.array([1.0, 2.0, 3.0, 2.5, 6, 8.4, 1.8, 4.0, 5.0])
y_train = np.array([300.0, 500.0, 750.0, 700.0, 1300.0, 1500, 400.0, 800.0, 1100.0])

w_init = 0
b_init = 0
iterations = 10000
tmp_alpha = 1.0e-2

w_final, b_final, j_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_derivatives)
f_wb = regression(x_train, w_final, b_final)

print(f"(w, b) found by gradient descent is ({w_final: 8.4f}, {b_final: 8.4f})")
# print(f"x = {x_train.shape}   y = {y_train.shape} f_wb = {f_wb.shape}")

plt.scatter(x_train, y_train, marker='x', c='r', label="Actual values")
plt.plot(x_train, f_wb, label="Our Prediction")
plt.title("Housing Prices vs Area")
plt.ylabel("Price in 1000$")
plt.xlabel("Area in 1000sq. ft")
plt.grid()
plt.legend()
plt.show()
