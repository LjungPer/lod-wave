'''
Generate the first plot. Compare localized to non-localized method as k increases.
'''
import numpy as np
import matplotlib.pyplot as plt
from gridlod.world import World
from methods import standard_method

# fine mesh set up
fine = 128
NFine = np.array([fine, fine])
NpFine = np.prod(NFine + 1)
boundaryConditions = np.array([[0, 0], [0, 0]])
world = World(NFine, NFine // NFine, boundaryConditions)

# parameters
tau = 0.02
N = 16
tot_time_steps = 50
n = 2

# ms coefficient A
A = np.kron(np.random.rand(fine // n, fine // n) * 999.9 + 0.1, np.ones((n, n)))
aFine = A.flatten()

# ms coefficient B
B = np.kron(np.random.rand(fine // n, fine // n) * 999.9 + 0.1, np.ones((n, n)))
bFine = B.flatten()

# source function
f = np.ones(NpFine)

# localization and mesh width parameters
kList = range(2,7)

error = [] 
sol, init_val = standard_method(world, N, fine, tau, tot_time_steps, np.inf, aFine, bFine, f)
for k in kList:

    locsol, init_val = standard_method(world, N, fine, tau, tot_time_steps, k, aFine, bFine, f)
    error.append(np.sqrt(np.dot(np.gradient(sol - locsol), np.gradient(sol - locsol))) / np.sqrt(np.dot(np.gradient(sol), np.gradient(sol))))



# plot errors
plt.figure('Error', figsize=(16, 9))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tick_params(labelsize=24)
plt.semilogy(kList, error, '-o')
plt.grid(True, which="both")
plt.xlabel('$1/H$', fontsize=24)
plt.legend(fontsize=24)

plt.show()



