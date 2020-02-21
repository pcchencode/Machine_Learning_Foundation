## Date: 2017-12-27
## Purpose: Machine Learning Fundation hw3_7
## Author: Po-Chu Chen

import numpy as np
import matplotlib.pylab as plt

def generate(N, flips_rate):
    x1 = np.random.uniform(-1,1,N)
    x2 = np.random.uniform(-1,1,N)
    f = np.sign(x1**2+x2**2-0.6)
    y = np.where(np.random.rand(N) < flips_rate, -f, f).T
    return x1, x2, y

def generate_X(N, flips_rate):
    x1, x2, y = generate(N, flips_rate)  
    X = np.vstack((np.ones_like(x1), x1, x2)).T
    return X, y

def generate_Z(N, flips_rate):
    x1, x2, y = generate(N, flips_rate)
    Z = np.vstack((np.ones_like(x1), x1, x2,x1*x2,x1**2,x2**2)).T
    return Z, y

def error(X, y, w):
    y_predict = np.sign(X.dot(w))
    return np.sum(y!=y_predict) / y.size

N = 1000
flips_rate = 0.1
T=1000


all_e_in = []
for i in range(T):
    X, y = generate_X(N, flips_rate)
    w = np.linalg.pinv(X).dot(y)
    e_in = error(X ,y ,w)
    all_e_in.append(e_in)



all_e_out = []
all_w3 = []
for i in range(T):
    X, y = generate_Z(N, flips_rate)
    w = np.linalg.pinv(X).dot(y)
    Xt, yt = generate_Z(N, flips_rate) 
    e_out = error(Xt ,yt ,w)
    all_e_out.append(e_out)
    all_w3.append(w[3])


print('#7: e_out =', np.mean(all_e_out))
plt.figure()
plt.hist(all_e_out)
plt.show()
