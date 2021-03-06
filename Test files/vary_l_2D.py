import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from gridlod import util, fem, coef, interp, linalg, pg
from gridlod.world import World
import lod_wave
from visualize import drawCoefficient
import gc
from math import log


'''
Settings
'''

# fine mesh parameters
fine = 64
NFine = np.array([fine, fine])
NpFine = np.prod(NFine + 1)
boundaryConditions = np.array([[0, 0], [0, 0]])
world = World(NFine, NFine / NFine, boundaryConditions)
NWorldFine = world.NWorldCoarse * world.NCoarseElement

# fine grid elements and nodes
xt = util.tCoordinates(NFine).flatten()
xp = util.pCoordinates(NFine).flatten()

# time step parameters
tau = 0.1
numTimeSteps = 150
n = 2

# ms coefficient B
B = np.kron(np.random.rand(fine/n, fine/n)*0.9 + 0.1, np.ones((n, n)))
bFine = B.flatten()
plt.figure("OriginalCoefficient")
drawCoefficient(NWorldFine, bFine)
plt.title('$B(x,y)$', fontsize=24)
plt.show()

# ms coefficient A
A = np.kron(np.random.rand(fine/n, fine/n)*0.9 + 0.1, np.ones((n, n)))
aFine = A.flatten()
plt.figure("OriginalCoefficient")
drawCoefficient(NWorldFine, aFine)
plt.title('$A(x,y)$', fontsize=24)
plt.show()


# localization and mesh width parameters
N = 16
k = log(N, 2)
lList = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150]

# coarse mesh parameters
NWorldCoarse = np.array([N, N])
NCoarseElement = NFine / NWorldCoarse
world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

# grid nodes
xpCoarse = util.pCoordinates(NWorldCoarse).flatten()
NpCoarse = np.prod(NWorldCoarse + 1)

# patch generator and coefficients
IPatchGenerator = lambda i, N: interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse,
                                                              NCoarseElement, boundaryConditions)
b_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, bFine)
a_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aFine / tau)

error = []

for l in lList:

    # compute basis correctors
    lod = lod_wave.LodWave(b_coef, world, k, IPatchGenerator, a_coef)
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
    for i in range(l):
        print('Calculating correction at l = %d, i = %d' % (l, i))

        # solve non-localized system
        lod = lod_wave.LodWave(b_coef, world, k+1, IPatchGenerator, a_coef, prev_fs_sol, ms_basis)
        lod.solve_fs_system_parallel()

        # store sparse solution
        prev_fs_sol = sparse.csc_matrix(np.array(np.column_stack(lod.fs_list)))
        fs_solutions.append(prev_fs_sol)

    '''
    Compute v^n and w^n
    '''

    # initial value
    Uo = np.zeros(NpCoarse)

    # coarse v^(-1) and v^0
    V = [Uo]
    V.append(Uo)

    # reference solution
    UFine = [ms_basis * Uo]
    UFine.append(ms_basis * Uo)

    # compute ms matrices
    S = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, aFine)
    K = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, bFine)
    M = fem.assemblePatchMatrix(NWorldFine, world.MLocFine)

    free = util.interiorpIndexMap(NWorldCoarse)

    SmsFree = (ms_basis.T * S * ms_basis)[free][:, free]
    KmsFree = (ms_basis.T * K * ms_basis)[free][:, free]
    MmsFree = (ms_basis.T * M * ms_basis)[free][:, free]

    # load vector
    f = np.ones(NpFine)
    LmsFree = (ms_basis.T * M * f)[free]

    RmsFreeList = []
    for i in range(numTimeSteps):
        n = i + 1

        # linear system
        A = (1. / (tau ** 2)) * MmsFree + (1. / tau) * SmsFree + KmsFree
        b = LmsFree + (1. / tau) * SmsFree * V[n][free] + (2. / (tau ** 2)) * MmsFree * V[n][free] - (1. / (
        tau ** 2)) * MmsFree * V[n - 1][free]

        # store ms matrix R^{ms',h}_{H,i,k}
        if i < l:
            RmsFull = ms_basis.T * S * fs_solutions[i]
            RmsFree = RmsFull[free][:, free]
            RmsFreeList.append(RmsFree)

        # add sum to linear system
        if i is not 0:
            if i < l:
                for j in range(i):
                    b += (1. / tau) * RmsFreeList[j] * V[n - 1 - j][free]
            else:
                for j in range(l):
                    b += (1. / tau) * RmsFreeList[j] * V[n - 1 - j][free]

        # solve system
        VFree = linalg.linSolve(A, b)
        VFull = np.zeros(NpCoarse)
        VFull[free] = VFree

        # append solution for current time step
        V.append(VFull)

    gc.collect()

    VFine = ms_basis * V[-1]

    WFine = 0
    for j in range(0, l):
        WFine += fs_solutions[j] * V[n - j]


    '''
    Compute reference solution
    '''

    # fine free indices
    boundaryMap = boundaryConditions == 0
    fixedFine = util.boundarypIndexMap(NWorldFine, boundaryMap)
    freeFine = np.setdiff1d(np.arange(NpFine), fixedFine)

    # fine matrices
    SFree = S[freeFine][:, freeFine]
    KFree = K[freeFine][:, freeFine]
    MFree = M[freeFine][:, freeFine]
    LFineFree = (M * f)[freeFine]

    del S, K, M, f
    gc.collect()

    for i in range(numTimeSteps):

        # reference system
        A = (1. / (tau ** 2)) * MFree + (1. / tau) * SFree + KFree
        b = LFineFree + (1. / tau) * SFree * UFine[1][freeFine] + (2. / (tau ** 2)) * MFree * UFine[1][freeFine] - \
            (1. / (tau ** 2)) * MFree * UFine[0][freeFine]

        # solve system
        UFineFree = linalg.linSolve(A, b)
        UFineFull = np.zeros(NpFine)
        UFineFull[freeFine] = UFineFree

        # append solution
        UFine[0] = UFine[1]
        UFine[1] = UFineFull

    # evaluate H^1-error for time step N
    error.append(np.sqrt(np.dot(np.gradient(UFine[-1] - VFine - WFine), np.gradient(UFine[-1] - VFine - WFine))))

    del SFree, KFree, MFree, LFineFree, A, b, UFineFree, UFineFull, VFine, WFine, freeFine, fixedFine, fs_solutions
    gc.collect()


# plot errors
plt.figure('Error comparison', figsize=(16,9))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.subplots_adjust(left=0.05, bottom=0.12, right=0.99, top=0.92, wspace=0.1, hspace=0.2)
plt.tick_params(labelsize=24)
plt.semilogy(lList, error, '--s')
plt.grid(True, which="both")
plt.xlabel('$l$', fontsize=30)
plt.title(r'The error $\|u^n-u^n_{\mathrm{ms}, k, l}\|_{H^1}$ at $T=%.1f$' % (numTimeSteps * tau), fontsize=44)
plt.legend(fontsize=24)

plt.show()