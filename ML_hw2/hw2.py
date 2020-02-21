## Date: 2017-12-03
## Purpose: Machine Learning Fundation hw2
## Author: Po-Chu Chen

import numpy as np
import random
import matplotlib.pyplot as plt

def gen_data(size, flips_rate):
    x = np.random.uniform(-1,1,size)
    y = np.sign(np.random.uniform(0,1,size) - flips_rate) * np.sign(x)
    return x, y 
    
def e_out(lamda, mu):
    return lamda*mu+(1-lamda)*(1-mu)
    
def decision_stump(x , y):
    size = x.size   
    sorted_x = np.sort(x)
    pocket_err = 1 # initial err.rate
    for i in range(size-2):
        s = 1
        theta = (sorted_x[i] + sorted_x[i+1]) / 2 
        y_predict = np.sign(x-theta)
        err = np.sum(y!=y_predict) / size
        if err > 0.5:
            err = 1 - err
            s = -1 # the ideal-mini target 
        if err < pocket_err or (err==pocket_err and random.random()>0.5):
            pocket_s = s
            pocket_theta = theta
            pocket_err = err
    return pocket_s, pocket_theta, pocket_err
    
flips_rate = 0.2
lamda = 1-flips_rate
size = 20
T = 1000

all_e_in=[]
all_e_out=[]
for i in range(T):
    x, y = gen_data(size,flips_rate)
    s,theta,E_in = decision_stump(x, y)
    mu = 0.5 + 0.5 * s * (abs(theta) - 1)
    E_out = e_out(lamda, mu)
    all_e_in.append(E_in)
    all_e_out.append(E_out)
    
print('Average_E_in=', np.average(all_e_in))
plt.figure()
plt.hist(all_e_in)
plt.show()

print('Average_E_out=', np.average(all_e_out))
plt.figure()
plt.hist(all_e_out)
plt.show()


x = np.arange(0, 1000, 1)
plt.figure()
plt.scatter(x, all_e_in, label="E_in")
plt.scatter(x, all_e_out, label="E_out")
plt.xlabel("Experiments N")
plt.ylabel("Ein\Eout")
plt.legend(loc=2)
plt.show()

plt.figure()
plt.scatter(all_e_in, all_e_out)
plt.xlabel("E_in")
plt.ylabel("E_out")
plt.show()

