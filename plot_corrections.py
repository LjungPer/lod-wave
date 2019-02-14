import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from gridlod import util, fem, coef, interp
from gridlod.world import World
import lod_wave

# fine mesh parameters
fine = 1024
NFine = np.array([fine])
NpFine = np.prod(NFine + 1)
boundaryConditions = np.array([[0, 0]])
world = World(NFine, NFine // NFine, boundaryConditions)
NWorldFine = world.NWorldCoarse * world.NCoarseElement

# fine grid elements and nodes
xt = util.tCoordinates(NFine).flatten()
xp = util.pCoordinates(NFine).flatten()

# time step parameters
tau = 0.1
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

prev_fs_sol = ms_basis
fs_solutions = []
for i in range(numTimeSteps):

    # solve system
    lod = lod_wave.LodWave(b_coef, world, k_1, IPatchGenerator, a_coef, prev_fs_sol)
    lod.solve_fs_system()

    # store sparse solution
    prev_fs_sol = sparse.csc_matrix(np.array(np.column_stack(lod.fs_list)))
    fs_solutions.append(prev_fs_sol)

nList = list(range(numTimeSteps))
x = N / 2
plt.figure('Solution corrections', figsize=(16, 9))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.subplots_adjust(left=0.07, bottom=0.07, right=0.99, top=0.92, wspace=0.1, hspace=0.2)
plt.tick_params(labelsize=24)
for n in nList:
    plt.plot(xp, fs_solutions[n][:, x].todense(), label='$n=%d$' % (n + 1), linewidth=1.5)
plt.xlim(0.25, 0.75)
plt.title('Solution correction $w^n_x$ for different choices of $n$', fontsize=44)
plt.grid(True, which="both", ls="--")
plt.legend(frameon=True, loc=1, fontsize=24)
plt.show()
