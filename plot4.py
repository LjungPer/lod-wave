import numpy as np
import matplotlib.pyplot as plt
from gridlod.world import World
from math import log
from methods import reference_sol, rb_method_sparse_stop

# fine mesh set up
fine = 256
NFine = np.array([fine, fine])
NpFine = np.prod(NFine + 1)
boundaryConditions = np.array([[0, 0], [0, 0]])
world = World(NFine, NFine // NFine, boundaryConditions)

# parameters
tau = 0.02
Nlist = [64]
tot_time_steps = 50
n = 2
no_rb_vecs = tot_time_steps

np.random.seed(0)

# ms coefficient A
A = np.kron(np.random.rand(fine // n, fine // n) * 999.9 + 0.1, np.ones((n, n)))
aFine = A.flatten()

# ms coefficient B
B = np.kron(np.random.rand(fine // n, fine // n) * 999.9 + 0.1, np.ones((n, n)))
bFine = B.flatten()

error = []
f = np.ones(NpFine)
for N in Nlist:
    k = log(N, 2)
    rb_sol, init_val = rb_method_sparse_stop(world, N, fine, tau, tot_time_steps, k, aFine, bFine, f, no_rb_vecs, 1e-10)
    ref_sol = reference_sol(world, fine, tau, tot_time_steps, init_val, aFine, bFine, f)
    error.append(np.sqrt(np.dot(np.gradient(rb_sol - ref_sol), np.gradient(rb_sol - ref_sol))) / np.sqrt(np.dot(np.gradient(ref_sol), np.gradient(ref_sol))))

print(error)
plt.figure('Error comparison', figsize=(16, 9)) 
plt.rc('text', usetex=True) 
plt.rc('font', family='serif') 
plt.tick_params(labelsize=24) 
plt.loglog(Nlist, error, '-s', basex=2, basey=10) 
plt.grid(True, which="both") 
plt.xlabel('$1/H$', fontsize=24)
plt.show()      


#error = [0.559626357311834, 0.21639836867103124, 0.07971552002933713, 0.028146837176322752, 0.009333305319712054]