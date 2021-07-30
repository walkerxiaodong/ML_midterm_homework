import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cv

w = np.zeros(2)
A = np.array([[3, 0.5], [0.5, 1]])
mu = np.array([1, 2])
lr = 1/np.max(np.linalg.eig(2*A)[0])

#define the function
def func(lamda):
    w = np.array([3.0, -1.0])
    w_list = [np.copy(w)]
    t = 0
    lipsitz = lr*lamda
    while 1:
        grad = np.zeros_like(w)
        w_temp = np.copy(w)
        grad = 2*np.dot(A, (w - mu).T)
        w -= lr*grad
        for i in range(len(w)):
            if w[i] > lipsitz:
                w[i] -= lipsitz
            elif w[i] < -lipsitz:
                w[i] += lipsitz
            else:
                w[i] = 0
        w_list.append(np.copy(w))
        if np.all(w == w_temp) or t == 200:
            print(w)
            break
        t += 1        
    w_list = np.array(w_list)
    w_path_list = np.copy(w_list[:-1])
    w_list = w_list - w_list[-1]
    w_list = [np.linalg.norm(i, ord=2) for i in w_list]    
    return np.copy(w_path_list)
#Implement 
p2 = func(2)
p4 = func(4)
p6 = func(6)
plt.plot(p2[0:-1, 0], p2[0:-1, 1], label = 'λ = 2')
plt.plot(p4[0:-1, 0], p4[0:-1, 1], label = 'λ = 4')
plt.plot(p6[0:-1, 0], p6[0:-1, 1], label = 'λ = 6')
plt.legend()
plt.show()