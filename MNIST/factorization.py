import torch
import time

import pyswarms as ps
import numpy as np

def random_split(size,k):
    m=size[0]
    n=size[1]

    if m>=n:
        W_pos = np.random.randint(0,2, [m,k])
        W_small = np.random.randint(-1,1, [k,n])
    else:
        W_pos = np.random.randint(0,2, [k,n])
        W_small = np.random.randint(-1,1, [m,k])
        
    for i in range(np.shape(W_small)[0]):
            for j in range(np.shape(W_small)[1]):
                if W_small[i][j]==0:
                    W_small[i][j]=1
    return W_pos, W_small

    
def frobenius(W, W_pos, W_small):
    m=size[0]
    n=size[1]
    
    if m>=n:
        return np.linalg.norm(W-np.dot(W_pos,W_small))
    else:
        return np.linalg.norm(W-np.dot(W_small,W_pos))

#cost for each particle
def cost(x):
    m=size[0]
    n=size[1]

    if m>=n:    
        W_pos = x[0:m*k].reshape(m,k)
        W_small = x[m*k:m*k + k*n].reshape(k,n)
        result = np.dot(W_pos,W_small)
    else:
        W_small = x[0:m*k].reshape(m,k)
        W_pos = x[m*k:m*k + k*n].reshape(k,n)
        result = np.dot(W_small,W_pos)
    return np.linalg.norm(W-result) - 0*np.linalg.norm(result - np.zeros((m,n)))

#cost for all particles
def f(x):
    n_particles = x.shape[0]
    j = [cost(x[i]) for i in range(n_particles)]
    return np.array(j)

#evaluation
def pso(size, k):
    m=size[0]
    n=size[1]

    dimensions = m*k + k*n
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options = options, bounds = constraints)
    loss, pos = optimizer.optimize(f, iters=100)
    if m>=n:
        W_pos = pos[0:m*k].reshape(m,k)
        W_small = pos[m*k:(m*k + k*n)].reshape(k,n)
    else:
        W_small = pos[0:m*k].reshape(m,k)
        W_pos = pos[m*k:(m*k + k*n)].reshape(k,n)
    return loss, W_pos, W_small

def constraints(x, *args):
    m=size[0]
    n=size[1]
    if m>=n:
        W_pos = x[0:m*k].reshape(m,k)
        W_small = x[m*k:(m*k + k*n)].reshape(k,n)
    else:
        W_small = x[0:m*k].reshape(m,k)
        W_pos = x[m*k:(m*k + k*n)].reshape(k,n)
    return [-W_pos]


K = 316
N = 100

experiments = 1
size = (K,N)

k = 25


for i in range(experiments):
    t1 = time.time()
    loss, W_pos, W_small = pso(size, k)
    t2 = time.time()
    
print(loss)
