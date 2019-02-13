import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from gridlod import util, fem, coef, interp
from gridlod.world import World
import lod_wave

'''
Settings
'''

# fine mesh parameters
fine = 1024
NFine = np.array([fine])
NpFine = np.prod(NFine + 1)
boundaryConditions = np.array([[0, 0]])
world = World(np.array([256]), NFine // np.array([256]), boundaryConditions)
NWorldFine = world.NWorldCoarse * world.NCoarseElement

# fine grid elements and nodes
xt = util.tCoordinates(NFine).flatten()
xp = util.pCoordinates(NFine).flatten()

# time step parameters
tau = 0.001
numTimeSteps = 10

# ms coefficients
epsA = 2 ** (-4)
epsB = 2 ** (-6)
aFine = (2 - np.sin(2 * np.pi * xt / epsA)) ** (-1)
bFine = (2 - np.cos(2 * np.pi * xt / epsB)) ** (-1)

k_0 = np.inf
k_1 = k_0
N = 16

# coarse mesh parameters
NWorldCoarse = np.array([N])
NCoarseElement = NFine // NWorldCoarse
world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

# grid nodes
xpCoarse = util.pCoordinates(NWorldCoarse).flatten()
NpCoarse = np.prod(NWorldCoarse + 1)

'''
Compute multiscale basis
'''


# patch generator and coefficients
def IPatchGenerator(i, N):
    return interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement, boundaryConditions)


b_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, bFine)
a_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aFine / tau)

# compute basis correctors
lod = lod_wave.LodWave(b_coef, world, k_0, IPatchGenerator, a_coef)
lod.compute_basis_correctors()

# compute ms basis
basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
basis_correctors = lod.assembleBasisCorrectors()
ms_basis = basis - basis_correctors

'''
Compute finescale system

fs_solutions[i] = {w^i_x}_x
'''

prev_fs_sol = ms_basis
fs_solutions = []
for i in range(numTimeSteps):

    # solve system
    lod = lod_wave.LodWave(b_coef, world, k_1, IPatchGenerator, a_coef, prev_fs_sol, ms_basis)
    lod.solve_fs_system(localized=False)

    # store sparse solution
    prev_fs_sol = sparse.csc_matrix(np.array(np.column_stack(lod.fs_list)))
    fs_solutions.append(prev_fs_sol[:, N / 2])

# convert finescale corrections to correct shape
wn = np.array([np.squeeze(np.asarray(fs_solutions[n].todense())) for n in range(numTimeSteps)])
u, s, vh = np.linalg.svd(wn)


# plot
plt.figure('Singular values', figsize=(16, 9))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.subplots_adjust(left=0.08, bottom=0.09, right=0.99, top=0.92, wspace=0.1, hspace=0.2)
plt.tick_params(labelsize=24)
plt.semilogy(list(range(numTimeSteps)), s, '*')
plt.title('Singular values', fontsize=32)
plt.xlabel('$n$', fontsize=24)
plt.ylabel('$\sigma$', fontsize=24)
plt.grid(True, which="both", ls="--")
plt.show()
