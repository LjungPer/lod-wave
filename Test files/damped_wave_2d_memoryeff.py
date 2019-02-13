import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from gridlod import util, fem, coef, interp, linalg, pg
from gridlod.world import World
import lod_wave
from visualize import drawCoefficient
from math import log
import gc
import time

'''
Settings
'''

# fine mesh parameters
fine = 64
NFine = np.array([fine, fine])
NpFine = np.prod(NFine + 1)
boundaryConditions = np.array([[0, 0], [0, 0]])
world = World(NFine, NFine // NFine, boundaryConditions)
NWorldFine = world.NWorldCoarse * world.NCoarseElement

# fine grid elements and nodes
xt = util.tCoordinates(NFine).flatten()
xp = util.pCoordinates(NFine).flatten()

# time step parameters
tau = 0.05
numTimeSteps = 8
n = 2

# ms coefficient B
B = np.kron(np.random.rand(fine // n, fine // n), np.ones((n, n)))
bFine = B.flatten()
#plt.figure("OriginalCoefficient")
#drawCoefficient(NWorldFine, bFine)
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#plt.title(r'Multiscale coefficient $B(x,y)$', fontsize=24)
#plt.show()

# ms coefficient A
A = np.kron(np.random.rand(fine // n, fine // n), np.ones((n, n)))
aFine = A.flatten()
#plt.figure("OriginalCoefficient")
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#drawCoefficient(NWorldFine, aFine)
#plt.title(r'Multiscale coefficient $A(x,y)$', fontsize=24)
#plt.show()

# localization and mesh width parameters
NList = [2, 4, 8, 16]
error = []
y = []
start = time.time()
for N in NList:

    print('N = %d' %N)
    k_0 = log(N, 2)
    y.append(1. / N)
    # coarse mesh parameters
    NWorldCoarse = np.array([N, N])
    NCoarseElement = NFine // NWorldCoarse
    world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

    # grid nodes
    xpCoarse = util.pCoordinates(NWorldCoarse).flatten()
    NpCoarse = np.prod(NWorldCoarse + 1)

    '''
    Compute multiscale basis
    '''

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
    del basis, basis_correctors
    gc.collect()

    prev_fs_sol = ms_basis
    fs_solutions = []
    for i in range(numTimeSteps):
        print('Calculating correction at N = %d, i = %d' % (N, i))

        # solve non-localized system
        lod = lod_wave.LodWave(b_coef, world, k_0 + 1, IPatchGenerator, a_coef, prev_fs_sol)
        lod.solve_fs_system(localized=True)

        # store sparse solution
        prev_fs_sol = sparse.csc_matrix(np.array(np.column_stack(lod.fs_list)))
        fs_solutions.append(prev_fs_sol)


    '''
    Initialize data
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

    prev_fs_sol = ms_basis

    '''
    Compute v^n
    '''

    RmsFreeList = []
    for i in range(numTimeSteps):
        print('Calc v^n N=%d, i=%d' %(N, i))

        n = i + 1

        # linear system
        A = (1. / (tau ** 2)) * MmsFree + (1. / tau) * SmsFree + KmsFree
        b = LmsFree + (1. / tau) * SmsFree * V[n][free] + (2. / (tau ** 2)) * MmsFree * V[n][free] \
            - (1. / (tau ** 2)) * MmsFree * V[n - 1][free]

        # solve localized system
        #lod = lod_wave.LodWave(b_coef, world, k_0 + 1, IPatchGenerator, a_coef, prev_fs_sol)
        #lod.solve_fs_system(localized=True)

        # store sparse solution
        #prev_fs_sol = sparse.csc_matrix(np.array(np.column_stack(lod.fs_list)))

        # store ms matrix R^{ms',h}_{H,i,k}
        RmsFree = (ms_basis.T * S * fs_solutions[i])[free][:, free]
        RmsFreeList.append(RmsFree)

        # add sum to linear system
        if i != 0:
            for j in range(i):
                b += (1. / tau) * RmsFreeList[j] * V[n - 1 - j][free]

        # solve system
        VFree = linalg.linSolve(A, b)
        VFull = np.zeros(NpCoarse)
        VFull[free] = VFree

        # append solution for current time step
        V.append(VFull)

    del RmsFreeList, RmsFree, VFull, VFree, A, b, SmsFree, KmsFree, MmsFree, free
    gc.collect()

    VFine = ms_basis * V[-1]

    '''
    Compute w^n
    '''

    WFine = 0
    prev_fs_sol = ms_basis
    for j in range(0, numTimeSteps):
        print('Calc w^n N=%d, i=%d' % (N, j))

        # solve localized system
        #lod = lod_wave.LodWave(b_coef, world, k_0 + 1, IPatchGenerator, a_coef, prev_fs_sol)
        #lod.solve_fs_system(localized=True)
        # store sparse solution
        #prev_fs_sol = sparse.csc_matrix(np.array(np.column_stack(lod.fs_list)))

        WFine += fs_solutions[j] * V[n - j]
        gc.collect()


    '''
    Compute reference solution
    '''

    # fine free indices
    boundaryMap = boundaryConditions == 0
    fixedFine = util.boundarypIndexMap(NWorldFine, boundaryMap)
    freeFine = np.setdiff1d(np.arange(NpFine), fixedFine)

    SFree = S[freeFine][:, freeFine]
    KFree = K[freeFine][:, freeFine]
    MFree = M[freeFine][:, freeFine]
    LFineFree = (M * f)[freeFine]

    del S, K, M, f
    gc.collect()

    for i in range(numTimeSteps):
        print('Calc uref N=%d, i=%d' % (N, i))

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
    del SFree, KFree, MFree, LFineFree, A, b, UFineFree, UFineFull, VFine, WFine, freeFine, fixedFine
    gc.collect()

end = time.time()
print("Elapsed time: %d minutes." % (int((end - start) / 60)))

# plot errors
plt.figure('Error comparison', figsize=(16, 9))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.subplots_adjust(left=0.05, bottom=0.12, right=0.99, top=0.92, wspace=0.1, hspace=0.2)
plt.tick_params(labelsize=24)
plt.loglog(NList, error, '--s', basex=2, basey=2, label='LOD')
plt.loglog(NList, y, '--k', basex=2, basey=2, label=r'$\mathcal{O}(H)$')
plt.grid(True, which="both")
plt.xlabel('$1/H$', fontsize=30)
plt.title(r'The error $\|u^n-u^n_{\mathrm{ms}, k}\|_{H^1}$ at $T=%.1f$' % (numTimeSteps * tau), fontsize=44)
plt.legend(fontsize=24)

plt.show()