'''
Generate the third plot, i.e. compare error between standard method and the svd-rb one.
- Fixed TOL, k, H
- Vary over no_rb_vectors (maybe in another way but do this way)
- Plot error
'''
import numpy as np
import matplotlib.pyplot as plt
from gridlod.world import World
from math import log
import time
from methods import rb_svd_method, reference_sol, rb_method, standard_method, standard_fem

# fine mesh set up
fine = 128
NFine = np.array([fine, fine])
NpFine = np.prod(NFine + 1)
boundaryConditions = np.array([[0, 0], [0, 0]])
world = World(NFine, NFine // NFine, boundaryConditions)

# parameters
tau = 0.01
no_rb_vecs = 3
tot_time_steps = 10
TOL = 1e-5
n = 2

# ms coefficient A
A = np.kron(np.random.rand(fine // n, fine // n) * 0.9 + 0.1, np.ones((n, n)))
aFine = A.flatten()

# ms coefficient B
B = np.kron(np.random.rand(fine // n, fine // n) * 0.9 + 0.1, np.ones((n, n)))
bFine = B.flatten()

# localization and mesh width parameters
NList = [2, 4]

error = []
errorFEM = []
y = []
start = time.time()
for N in NList:

    y.append(1. / N)
    k = log(N, 2)
    f = np.ones(NpFine)

    #sol, init_val = rb_svd_method(world, N, fine, tau, tot_time_steps, k, aFine, bFine, f, TOL)
    sol, init_val = rb_method(world, N, fine, tau, tot_time_steps, k, aFine, bFine, f, no_rb_vecs)
    #sol, init_val = standard_method(world, N, fine, tau, tot_time_steps, k, aFine, bFine, f)
    ref_sol = reference_sol(world, fine, tau, tot_time_steps, init_val, aFine, bFine, f)
    fem_sol = standard_fem(world, N, fine, tau, tot_time_steps, aFine, bFine, f)
    # evaluate H^1-error for time step N
    error.append(np.sqrt(np.dot(np.gradient(ref_sol - sol), np.gradient(ref_sol - sol))) / np.sqrt(np.dot(np.gradient(ref_sol), np.gradient(ref_sol))))
    errorFEM.append(np.sqrt(np.dot(np.gradient(ref_sol - fem_sol), np.gradient(ref_sol - fem_sol))) / np.sqrt(np.dot(np.gradient(ref_sol), np.gradient(ref_sol))))

end = time.time()
print("Elapsed time: %d minutes." % (int((end - start) / 60)))

# plot errors
plt.figure('Error comparison', figsize=(16, 9))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.subplots_adjust(left=0.05, bottom=0.12, right=0.99, top=0.92, wspace=0.1, hspace=0.2)
plt.tick_params(labelsize=24)
plt.loglog(NList, error, '--s', basex=2, basey=2, label='LOD')
plt.loglog(NList, errorFEM, '--*', basex=2, basey=2, label='FEM')
plt.loglog(NList, y, '--k', basex=2, basey=2, label=r'$\mathcal{O}(H)$')
plt.grid(True, which="both")
plt.xlabel('$1/H$', fontsize=30)
plt.title(r'The error $\|u^n-u^n_{\mathrm{ms}, k}\|_{H^1}$ at $T=%.1f$' % (tot_time_steps * tau), fontsize=44)
plt.legend(fontsize=24)
plt.show()
