## Date: 2017-12-27
## Purpose: Machine Learning Fundation hw3_8&9
## Author: Po-Chu Chen

import numpy as np
import matplotlib.pylab as plt

# Load the dadaset
data = np.loadtxt('hw3_train.dat.txt')
X = data[:,:-1]
y = data[:,-1]
data_test = np.loadtxt('hw3_test.dat.txt')
X_test = data_test[:,:-1]
y_test = data_test[:,-1]


def sigmoid(s):
    return 1.0 / (1 + np.exp(-s))

def gd_lr(X ,y, lr, T):    
    N, d = X.shape    
    w = np.zeros(d)
    all_e_in_gd = []
    all_e_out_gd = []
    for i in range(T):
        delta = np.mean((sigmoid(-y*X.dot(w))*(-y)).reshape(N,1)*X, axis = 0)
        w = w - lr*delta
        e_in_gd = np.sum(np.sign(X.dot(w))!=y) / 1000 #1000 here is the size of X
        e_out_gd = np.sum(np.sign(X_test.dot(w))!=y_test) / 3000 #3000 here is the size of X_test
        all_e_in_gd.append(e_in_gd)
        all_e_out_gd.append(e_out_gd)
    return all_e_in_gd, all_e_out_gd

def sgd_lr(X ,y, lr, T):    
    N, d = X.shape    
    w = np.zeros(d)
    all_e_in_sgd = []
    all_e_out_sgd = []
    for i in range(T):
        n = i % N
        Xn = X[n]
        yn = y[n]
        w = w + lr*sigmoid(-yn*Xn.dot(w))*(yn*Xn)
        e_in_sgd = np.sum(np.sign(X.dot(w))!=y) / 1000 #1000 here is the size of X
        e_out_sgd = np.sum(np.sign(X_test.dot(w))!=y_test) / 3000 #3000 here is the size of X_test
        all_e_in_sgd.append(e_in_sgd)
        all_e_out_sgd.append(e_out_sgd)
    return all_e_in_sgd, all_e_out_sgd


# lr=0.01
Ein_sgd1 , Eout_sgd1 = sgd_lr(X, y, 0.01, 2000)
Ein_gd1 , Eout_gd1 = gd_lr(X, y, 0.01, 2000)
# lr=0.001
Ein_sgd2 , Eout_sgd2 = sgd_lr(X, y, 0.001, 2000)
Ein_gd2 , Eout_gd2 = gd_lr(X, y, 0.001, 2000)

## Q8. One figure for lr = 0.01 and the other for lr = 0.001
## Each figure should contain Ein for both GD and SGD on the same 
plt.figure()
plt.plot(Ein_gd1, 'rs' , label='GD')
plt.plot(Ein_sgd1, 'b--', label='SGD')
plt.title('Learning Rate = 0.01')
plt.ylabel("Ein")
plt.xlabel("t")
plt.legend()
plt.show()

plt.plot(Ein_gd2, 'rs', label='GD')
plt.plot(Ein_sgd2, 'b--', label='SGD')
plt.title('Learning Rate = 0.001')
plt.ylabel("Ein")
plt.xlabel("t")
plt.legend()
plt.show()

## Q9. One figure for lr = 0.01 and the other for lr = 0.001
## Each figure should contain Eout for both GD and SGD on the same 
plt.figure()
plt.plot(Eout_gd1, 'rs', label='GD')
plt.plot(Eout_sgd1, 'b--', label='SGD')
plt.title('Learning Rate = 0.01')
plt.ylabel("Eout")
plt.xlabel("t")
plt.show()

plt.figure()
plt.plot(Eout_gd2, 'rs', label='GD')
plt.plot(Eout_sgd2, 'b--', label='SGD')
plt.title('Learning Rate = 0.001')
plt.ylabel("Eout")
plt.xlabel("t")
plt.show()


