'''
Provar för tillfället att t.ex. beräkna 50 lösningar, och använda dem som V^RB. Även att beräkna 50 och använda 25 av dem
som V^RB, osv. Det tycks fungera ok, men vissa saker är konstiga, t.ex. att man får bättre av att ta 1:25 istället för 0:25.

Saker som verkligen inte fungerar: Använda SVD enligt POD-grejen...

Saker som bör göras: Använda greedy för att ortogonalisera V^RB, kanske det fungerar bättre då? (JUST NU EPIC FAIL)
'''


import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from gridlod import util, fem, coef, interp, linalg
from gridlod.world import World
import lod_wave
import lod_node
from pymor.algorithms import gram_schmidt
from pymor.vectorarrays.numpy import NumpyVectorArray, VectorSpaceInterface


def gram_schmidt_rb(snapshots):
        A = NumpyVectorArray(np.array(snapshots), VectorSpaceInterface)
        A_ortho = gram_schmidt.gram_schmidt(A)
        return A_ortho.to_numpy()

def projH1(u, v):
        return np.dot(np.gradient(u), np.gradient(v)) / np.dot(np.gradient(u), np.gradient(u)) * u

def projL2(u, v):
        return np.dot(u, v) / np.dot(u, u) * u

def orthonormalize_vector(basis, vector):
        for i in range(len(basis)):
                vector = vector - projH1(basis[i], vector)
        vector = vector / np.sqrt(np.dot(vector, vector))
        return vector

def mod_gram_schmidt(vectors):
        basis = []
        for v in vectors:
                basis.append(orthonormalize_vector(basis, v))
        return basis




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
tau = 0.01
numTimeSteps = 5
extra = 95
node = 4

# ms coefficients
epsA = 2 ** (-4)
epsB = 2 ** (-6)
aFine = (2 - np.sin(2 * np.pi * xt / epsA)) ** (-1)
bFine = (2 - np.cos(2 * np.pi * xt / epsB)) ** (-1)

N = 8
k = np.inf

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
lod = lod_wave.LodWave(b_coef, world, k, IPatchGenerator, a_coef)
lod.compute_basis_correctors()

# compute ms basis
basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
basis_correctors = lod.assembleBasisCorrectors()
ms_basis = basis - basis_correctors

prev_fs_sol = ms_basis
fs_solutions = []
for i in range(numTimeSteps+extra):

    # solve system
    lod = lod_wave.LodWave(b_coef, world, k, IPatchGenerator, a_coef, prev_fs_sol, ms_basis)
    lod.solve_fs_system()

    # store sparse solution
    prev_fs_sol = sparse.csc_matrix(np.array(np.column_stack(lod.fs_list)))
    fs_solutions.append(prev_fs_sol.toarray()[:, node])

## Try to recreate fs_solutions for that node only
prev_fs_sol_new = ms_basis.toarray()[:, node]
fs_solutions_new = []
for i in range(numTimeSteps):
    ecT = lod_node.nodeCorrector(world, k, 0)

    b_patch = b_coef.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
    a_patch = a_coef.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)

    IPatch = IPatchGenerator(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)

    fs_list = ecT.compute_node_correction(b_patch, a_patch, IPatch, prev_fs_sol_new, node)
    # store sparse solution
    #prev_fs_sol_new = sparse.csc_matrix(np.array(np.column_stack(lod.fs_list)))
    prev_fs_sol_new = fs_list[0]
    fs_solutions_new.append(prev_fs_sol_new)

wn = np.array([np.squeeze(np.asarray(fs_solutions_new[n])) for n in range(numTimeSteps)])
u, s, vh = np.linalg.svd(wn.T)
#V = sparse.csc_matrix(u[:, 1:numTimeSteps])
#V = sparse.csc_matrix(wn[0:numTimeSteps, :])
#V = sparse.csc_matrix(mod_gram_schmidt(fs_solutions_new))
V = sparse.csc_matrix(gram_schmidt_rb(fs_solutions_new))

ecT = lod_node.nodeCorrector(world, k, 0)
b_patch = b_coef.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
a_patch = a_coef.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
IPatch = IPatchGenerator(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)

fs_solutions_newnew = [ms_basis.toarray()[:, node]]
for i in range(extra):
    fs_solutions_new.append(V.T * ecT.compute_rb_node_correction(b_patch, a_patch, IPatch, fs_solutions_new[-1], V))


nList = range(numTimeSteps,numTimeSteps+extra)
plt.figure('Solution corrections', figsize=(16, 9))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.subplots_adjust(left=0.07, bottom=0.07, right=0.99, top=0.92, wspace=0.1, hspace=0.2)
plt.tick_params(labelsize=24)
for n in nList:
        plt.plot(xp, fs_solutions[n], 'b', linewidth=2)
        plt.plot(xp, fs_solutions_new[n], 'r', linewidth=1.5)
plt.xlim(0.25, 0.75)
plt.title('Comparison of exact and approximate solution correctors', fontsize=32)
plt.grid(True, which="both", ls="--")
plt.legend(frameon=True, loc=1, fontsize=24)
plt.show()

error = []
for i in range(numTimeSteps,numTimeSteps+extra):
        error = abs(fs_solutions[i] - fs_solutions_new[i])
print(max(error))
print(max(fs_solutions[numTimeSteps]))

