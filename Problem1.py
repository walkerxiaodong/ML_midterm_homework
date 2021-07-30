import matplotlib.pyplot as plt
import numpy as np


#prepare the data4
n = 200
x = 3 * (np.random.rand(n, 4) - 0.5)
y = (2 * x[:, 1] - 1 * x[:, 2] + 0.5 + 0.5 * np.random.randn(n)) > 0
y = 2 * y -1

##1.batch steepest gradient method
w = np.zeros(len(x[0]))
BSG = []
t = 0
lamba = 0.1
lr = 0.01
while 1:
    jwt,grad = 0,0
    w_temp = np.copy(w)
    for i in range(len(y)):
        exponent = np.exp(-y[i]*np.dot(w.T, x[i]))
        temp = exponent/(1 + exponent)
        grad -= y[i]*np.dot(x[i], temp)
        jwt += np.log(1 + exponent)
    grad += 2*lamba*w
    w -= lr*grad
    jwt += lamba*np.dot(w.T, w)
    BSG.append(jwt)
    if (t >1 and np.abs(BSG[-2] - BSG[-1]) < 1e-5):
        break
    t += 1

print(t)
print(w)
BSG = BSG - BSG[-1]
# bsg = [i for i in range(len(BSG))]
# plt.semilogy(bsg, BSG, label='Batch Steepest Gradient')
# plt.xlabel("Iteration")
# plt.ylabel("J(w)-J(w_hat)")
# plt.legend()
# plt.show()



##2.Newton method
w = np.zeros(len(x[0]))
Newton = []
t = 0
while 1:
    jwt, grad, hess = 0,0,0
    w_temp = np.copy(w)
    for i in range(len(y)):
        exponent = np.exp(-y[i]*np.dot(w.T, x[i]))
        temp = exponent/(1 + exponent)
        grad -= y[i]*np.dot(x[i], temp)
        hess += temp*(1 - temp)*np.dot(x[i], x[i].T)
        jwt += np.log(1 + exponent)
    grad += 2*lamba*w
    hess += 2
    w -= grad/hess
    jwt += lamba*np.dot(w.T, w)
    Newton.append(jwt)
    if (t >1 and np.abs(Newton[-2] - Newton[-1]) < 1e-5):
        break
    t = t + 1

print(t)
print(w)
Newton = Newton - Newton[-1]
# N = [i for i in range(len(Newton))]
# plt.semilogy(N, Newton, label='Newton Method')
# plt.xlabel("Iteration")
# plt.ylabel("J(w)-J(w_hat)")
# plt.legend()
# plt.show()

##3 Compare two method

bsg = [i for i in range(len(BSG))]
N = [i for i in range(len(Newton))]
plt.semilogy(bsg, BSG, label='Batch Steepest Gradient')
plt.semilogy(N, Newton, label='Newton Method')
plt.xlabel("Iteration")
plt.ylabel("J(w)-J(w_hat)")
plt.legend()
plt.show()


# # prepare date5
# n = 200
# x = 3 * (np.random.rand(n, 4) - 0.5)
# W = np.array([[ 2,  -1, 0.5,],
#               [-3,   2,   1,],
#               [ 1,   2,   3]])
# y = np.argmax(np.dot(np.hstack([x[:,:2], np.ones((n, 1))]), W.T)
#                         + 0.5 * np.random.randn(n, 3), axis=1)




# #4.1 multiclass using BSG
# BSG = []
# t = 0
# w = np.zeros([3,4])
# lamba = 0.1
# lr = 0.01
# while 1:
#     jwt = 0
#     grad = np.zeros([3,4])
#     temp = np.zeros(np.max(y) + 1)
#     w_temp = np.copy(w)
#     for i in range(len(y)):
#         p = 0
#         for j in range(len(temp)):
#             temp[j] = np.exp(np.dot(w[j].T, x[i]))
#             p += np.exp(np.dot(w[j].T, x[i]))
#         temp /= p
#         for j in range(len(temp)):
#             if y[i] == j:
#                 grad[j] -= (1 - temp[j])*x[i]
#             else:
#                 grad[j] += temp[j]*x[i]
#         jwt += -np.dot(w[y[i]].T, x[i]) + np.log(p)
#     grad += 2*lamba*w
#     w -= lr*grad
#     jwt += np.linalg.norm(w, ord=2)**2
#     BSG.append(jwt)
#     if (t > 1 and np.abs(BSG[-2] - BSG[-1]) < 1e-5):
#         print(t)
#         print(w)
#         break
#     t += 1

# BSG = BSG - BSG[-1]
# x_bsg_m = [i for i in range(len(BSG))]
# plt.semilogy(x_bsg_m, BSG, label='BSG multiclass')
# plt.legend()
# plt.show()

# #4.2multiclass using Newton
# 
# Newton = []
# w = np.zeros([3,4])
# t = 0
# while 1:
#     jwt = 0
#     grad = np.zeros(w)
#     temp = np.zeros(np.max(y) + 1)
#     hess = np.zeros([len(temp), len(temp)])
#     w_temp = np.copy(w)
#     for i in range(len(y)):
#         p = 0
#         for j in range(len(temp)):
#             temp[j] = np.exp(np.dot(w[j].T, x[i]))
#             p += np.exp(np.dot(w[j].T, x[i]))
#         temp /= p
#         for j in range(len(temp)):
#             if y[i] == j:
#                 grad[j] -= (1 - temp[j])*x[i]
#             else:
#                 grad[j] += temp[j]*x[i]
#             for k in range(len(temp)):
#                 hess[j][k] = (temp[j]*temp[k] - temp[k])*np.dot(x[i], x[i].T)
#         jwt += -np.dot(w[y[i]].T, x[i]) + np.log(p)
#     grad += 2*lamba*w
#     hess += 2*np.identity(len(hess))
#     w -= lr*np.dot(np.linalg.inv(hess), grad)
#     jwt += np.linalg.norm(w, ord=2)**2
#     Newton.append(jwt)
#     if (t > 1 and np.abs(Newton[-2] - Newton[-1]) < 1e-5):
#         print(t)
#         print(w)
#         break
#     t += 1

# Newton = Newton - Newton[-1]
# x_N_m = [i for i in range(len(Newton))]
# plt.semilogy(x_N_m, Newton, label='Newton multiclass')
# plt.legend()
# plt.show()

# x_bsg_m = [i for i in range(len(BSG))]
# x_N_m = [i for i in range(len(Newton))]
# plt.semilogy(x_bsg_m, BSG, label='batch steepest gradient')
# plt.semilogy(x_N_m, Newton, label='Newton')
# plt.legend()
# plt.show()