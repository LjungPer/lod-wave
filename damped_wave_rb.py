'''
Väljer nu alla snapshots så fort ett singulärvärde är under TOL. Problematiskt att SVD tar tid som fanken?
'''

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from gridlod import util, fem, coef, interp, linalg
from gridlod.world import World
import lod_wave
import lod_node
from math import log
import time
from pymor.algorithms import gram_schmidt
from pymor.vectorarrays.numpy import NumpyVectorArray, VectorSpaceInterface


def gram_schmidt_rb(snapshots):
        A = NumpyVectorArray(np.array(snapshots), VectorSpaceInterface)
        A_ortho = gram_schmidt.gram_schmidt(A, rtol=1e-6)
        return A_ortho.to_numpy()

def compute_2d_node(NWorldCoarse, node_index):
    N = NWorldCoarse[0]
    b = 0
    while node_index > N:
        b += 1
        node_index -= N + 1
    a = node_index

    return np.array([a, b])

# fine mesh parameters
fine = 128
NFine = np.array([fine, fine])
NpFine = np.prod(NFine + 1)
boundaryConditions = np.array([[0, 0], [0, 0]])
world = World(NFine, NFine // NFine, boundaryConditions)
NWorldFine = world.NWorldCoarse * world.NCoarseElement

# fine grid elements and nodes
xt = util.tCoordinates(NFine).flatten()
xp = util.pCoordinates(NFine).flatten()

# time step parameters
tau = 0.01
numTimeSteps = 10
extra = 5
tot_time_steps = 100
TOL = 1e-8
n = 2

np.random.seed(1)

# ms coefficient B
B = np.kron(np.random.rand(fine // n, fine // n) * 0.9 + 0.1, np.ones((n, n)))
bFine = B.flatten()

# ms coefficient A
A = np.kron(np.random.rand(fine // n, fine // n) * 0.9 + 0.1, np.ones((n, n)))
aFine = A.flatten()

# localization and mesh width parameters
NList = [2,4,8]
rb_vecs = [8]
slist = []
error_rb = []
Vlist2 = []
for rb in rb_vecs:
    error = []
    y = []
    start = time.time()
    for N in NList:

        print('N = %d' % N)
        y.append(1. / N)
        k = log(N, 2)

        # coarse mesh parameters
        NWorldCoarse = np.array([N, N])
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

        ### HERE WE NEED TO IMPLEMENT THE RB THING ###
        fs_solutions_new = [np.zeros_like(ms_basis.toarray()) for i in range(tot_time_steps)]
        for node in range(NpCoarse):
            prev_fs_sol_new = ms_basis.toarray()[:, node]
            snapshots = []
            for time_step in range(rb):
                print('Calculating correction at N = %d, node = %d/%d, time_step = %d' % (N, node, NpCoarse, time_step))
                node_index_arr = compute_2d_node(world.NWorldCoarse, node)
                ecT = lod_node.nodeCorrector(world, k, node_index_arr)

                b_patch = b_coef.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
                a_patch = a_coef.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)

                IPatch = IPatchGenerator(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)

                fs_list = ecT.compute_localized_node_correction(b_patch, a_patch, IPatch, prev_fs_sol_new, node)
                prev_fs_sol_new = fs_list
                #fs_solutions_new.append(prev_fs_sol_new)
                fs_solutions_new[time_step][:, node] = prev_fs_sol_new
                snapshots.append(prev_fs_sol_new)
                #wn = np.array([np.squeeze(np.asarray(snapshots[n])) for n in range(time_step+1)])
                #if len(wn) == rb_vecs:
                #    u, s, vh = np.linalg.svd(wn.T)
                #    print(s)
                #    if np.any([s < TOL]):
                #        print('Snapshots computed, %d selected' %len(snapshots))
                #        break


            V = sparse.csc_matrix(gram_schmidt_rb(snapshots))
            #V = sparse.csc_matrix(snapshots)

            for i in range(time_step+1, tot_time_steps):
                fs_solutions_new[i][:, node] = V.T * ecT.compute_rb_node_correction(b_patch, a_patch, IPatch, fs_solutions_new[i-1][:, node], V)


        ### BEFORE HERE ###

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
        for i in range(tot_time_steps):
            n = i + 1

            # linear system
            A = (1. / (tau ** 2)) * MmsFree + (1. / tau) * SmsFree + KmsFree
            b = LmsFree + (1. / tau) * SmsFree * V[n][free] + (2. / (tau ** 2)) * MmsFree * V[n][free] \
                - (1. / (tau ** 2)) * MmsFree * V[n - 1][free]

            # store ms matrix R^{ms',h}_{H,i,k}
            RmsFull = ms_basis.T * S * sparse.csc_matrix(fs_solutions_new[i])
            RmsFree = RmsFull[free][:, free]
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
        VFine = ms_basis * V[-1]

        WFine = 0
        for j in range(tot_time_steps):
            WFine += sparse.csc_matrix(fs_solutions_new[j]) * V[n - j]

        # fine free indices
        boundaryMap = boundaryConditions == 0
        fixedFine = util.boundarypIndexMap(NWorldFine, boundaryMap)
        freeFine = np.setdiff1d(np.arange(NpFine), fixedFine)

        # fine matrices
        SFree = S[freeFine][:, freeFine]
        KFree = K[freeFine][:, freeFine]
        MFree = M[freeFine][:, freeFine]
        LFineFree = (M * f)[freeFine]

        for i in range(tot_time_steps):
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
    end = time.time()
    print("Elapsed time: %d minutes." % (int((end - start) / 60)))
    error_rb.append(error)

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


# plot errors
plt.figure('Error comparison', figsize=(16, 9))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.subplots_adjust(left=0.05, bottom=0.12, right=0.99, top=0.92, wspace=0.1, hspace=0.2)
plt.tick_params(labelsize=24)
for i in range(len(error_rb)):
    plt.loglog(NList, error_rb[i], '--s', basex=2, basey=2, label='LOD')
#plt.loglog(NList, y, '--k', basex=2, basey=2, label=r'$\mathcal{O}(H)$')
plt.grid(True, which="both")
plt.xlabel('$1/H$', fontsize=30)
plt.title(r'The error $\|u^n-u^n_{\mathrm{ms}, k}\|_{H^1}$ at $T=%.1f$' % (numTimeSteps * tau), fontsize=44)
plt.legend(fontsize=24)
plt.show()
