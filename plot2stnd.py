'''
Generate the plot that shows how the method converges with mesh size. (FEM included).
'''

import numpy as np
import matplotlib.pyplot as plt
from gridlod.world import World
from math import log
import time
from methods import reference_sol, standard_method, standard_fem, standard_lod

# fine mesh set up
fine = 256
NFine = np.array([fine, fine])
NpFine = np.prod(NFine + 1)
boundaryConditions = np.array([[0, 0], [0, 0]])
world = World(NFine, NFine // NFine, boundaryConditions)

# parameters
tau = 0.02
tot_time_steps = 50
n = 2

# ms coefficient A
A = np.kron(np.random.rand(fine // n, fine // n) * 999.9 + 0.1, np.ones((n, n)))
aFine = A.flatten()

# ms coefficient B
B = np.kron(np.random.rand(fine // n, fine // n) * 999.9 + 0.1, np.ones((n, n)))
bFine = B.flatten()

# localization and mesh width parameters
NList = [2, 4, 8, 16, 32]

error = []
errorFEM = []
errorLODA = []
errorLODB = []
y = []
start = time.time()
for N in NList:

    y.append(1. / N)
    k = log(N, 2)
    f = np.ones(NpFine)

    sol, init_val = standard_method(world, N, fine, tau, tot_time_steps, k, aFine, bFine, f)
    ref_sol = reference_sol(world, fine, tau, tot_time_steps, init_val, aFine, bFine, f)
    fem_sol = standard_fem(world, N, fine, tau, tot_time_steps, aFine, bFine, f)
    loda_sol = standard_lod(world, N, fine, tau, tot_time_steps, k, aFine, bFine, f, damp_corr=True)
    lodb_sol = standard_lod(world, N, fine, tau, tot_time_steps, k, aFine, bFine, f)
    
    error.append(np.sqrt(np.dot(np.gradient(ref_sol - sol), np.gradient(ref_sol - sol))) / np.sqrt(np.dot(np.gradient(ref_sol), np.gradient(ref_sol))))
    errorFEM.append(np.sqrt(np.dot(np.gradient(ref_sol - fem_sol), np.gradient(ref_sol - fem_sol))) / np.sqrt(np.dot(np.gradient(ref_sol), np.gradient(ref_sol))))
    errorLODA.append(np.sqrt(np.dot(np.gradient(ref_sol - loda_sol), np.gradient(ref_sol - loda_sol))) / np.sqrt(np.dot(np.gradient(ref_sol), np.gradient(ref_sol))))
    errorLODB.append(np.sqrt(np.dot(np.gradient(ref_sol - lodb_sol), np.gradient(ref_sol - lodb_sol))) / np.sqrt(np.dot(np.gradient(ref_sol), np.gradient(ref_sol))))

end = time.time()
print("Elapsed time: %d minutes." % (int((end - start) / 60)))  

plt.figure('Error', figsize=(16, 9))  
plt.rc('text', usetex=True)  
plt.rc('font', family='serif')
plt.tick_params(labelsize=24)  
plt.loglog(NList, error, '-s', basex=2, basey=10, label='LOD')  
plt.loglog(NList, errorFEM, '-o', basex=2, basey=10, label='FEM')  
plt.loglog(NList, errorLODA, '-s', basex=2, basey=10, label='LODA')  
plt.loglog(NList, errorLODB, '-*', basex=2, basey=10, label='LODB')  
plt.loglog(NList, y, '--k', basex=2, basey=10, label=r'$\mathcal{O}(H)$')   
plt.grid(True, which="both")  
plt.xlabel('$1/H$', fontsize=24)  
plt.legend(fontsize=24)  
plt.show()   

#error = [0.5582970114920637, 0.22012587691291693, 0.07964270248035327, 0.027948856768672266, 0.009267165625562538]
#errorFEM = [0.6166204221764932, 0.399727220044101, 0.31970290085717507, 0.2888499702154521, 0.27337388318668027]
#errorLODA = [0.6304317673071184, 0.42040382339916804, 0.34722066365899645, 0.309541285205036, 0.2854547218091534]
#errorLODB = [0.6419493785496603, 0.421815382214023, 0.36335765757969607, 0.3431595113573698, 0.32305533702727585]