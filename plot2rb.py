'''
Generate the plot that shows how the method converges with mesh size. (FEM included).
'''

import numpy as np
import matplotlib.pyplot as plt
from gridlod.world import World
from math import log
import time
from methods import reference_sol, rb_method, standard_fem, standard_method

# fine mesh set up
fine = 128
NFine = np.array([fine, fine])
NpFine = np.prod(NFine + 1)
boundaryConditions = np.array([[0, 0], [0, 0]])
world = World(NFine, NFine // NFine, boundaryConditions)

# parameters
tau = 0.01
tot_time_steps = 100
no_rb_vecs = 5
n = 2

# ms coefficient A
A = np.kron(np.random.rand(fine // n, fine // n) * 0.9 + 0.1, np.ones((n, n)))
aFine = A.flatten()

# ms coefficient B
B = np.kron(np.random.rand(fine // n, fine // n) * 0.9 + 0.1, np.ones((n, n)))
bFine = B.flatten()

# localization and mesh width parameters
NList = [2, 4, 8, 16]

error = []
error2 = []
errorFEM = []
y = []
start = time.time()
for N in NList:

    y.append(1. / N)
    k = log(N, 2)
    f = np.ones(NpFine)

    sol, init_val = rb_method(world, N, fine, tau, tot_time_steps, k, aFine, bFine, f, no_rb_vecs)
    sol2, init_val2 = standard_method(world, N, fine, tau, tot_time_steps, k, aFine, bFine, f)
    ref_sol = reference_sol(world, fine, tau, tot_time_steps, init_val, aFine, bFine, f)
    fem_sol = standard_fem(world, N, fine, tau, tot_time_steps, aFine, bFine, f)
    
    error.append(np.sqrt(np.dot(np.gradient(ref_sol - sol), np.gradient(ref_sol - sol))) / np.sqrt(np.dot(np.gradient(ref_sol), np.gradient(ref_sol))))
    error2.append(np.sqrt(np.dot(np.gradient(ref_sol - sol2), np.gradient(ref_sol - sol2))) / np.sqrt(np.dot(np.gradient(ref_sol), np.gradient(ref_sol))))
    errorFEM.append(np.sqrt(np.dot(np.gradient(ref_sol - fem_sol), np.gradient(ref_sol - fem_sol))) / np.sqrt(np.dot(np.gradient(ref_sol), np.gradient(ref_sol))))

end = time.time()
print("Elapsed time: %d minutes." % (int((end - start) / 60)))

plt.figure('Error', figsize=(16, 9)) 
plt.rc('text', usetex=True) 
plt.rc('font', family='serif') 
plt.tick_params(labelsize=24) 
plt.loglog(NList, error, '--s', basex=2, basey=2, label='LODRB') 
plt.loglog(NList, error2, '--s', basex=2, basey=2, label='LOD') 
plt.loglog(NList, errorFEM, '--*', basex=2, basey=2, label='FEM') 
plt.loglog(NList, y, '--k', basex=2, basey=2, label=r'$\mathcal{O}(H)$') 
plt.grid(True, which="both") 
plt.xlabel('$1/H$', fontsize=24) 
plt.title(r'Error at $T=%.1f$' % (tot_time_steps * tau), fontsize=32) 
plt.legend(fontsize=24) 
plt.show()    