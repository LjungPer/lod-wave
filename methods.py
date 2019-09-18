
import numpy as np
import scipy.sparse as sparse
from gridlod import util, fem, coef, interp, linalg, pg
from gridlod.world import World
import lod_wave
import lod_node
from pymor.algorithms import gram_schmidt
from pymor.vectorarrays.numpy import NumpyVectorArray, VectorSpaceInterface


def reference_sol(world, fine, tau, tot_time_steps, init_val, aFine, bFine, f):

    NFine = np.array([fine, fine])
    NpFine = np.prod(NFine + 1)
    bc = world.boundaryConditions
    NWorldFine = world.NWorldCoarse * world.NCoarseElement

    # set initial value
    U = [init_val]
    U.append(init_val)

    # assemble matrices
    S = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, aFine)
    K = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, bFine)
    M = fem.assemblePatchMatrix(NWorldFine, world.MLocFine)

    # find free indices
    boundary_map = bc == 0
    fixed = util.boundarypIndexMap(NWorldFine, boundary_map)
    free = np.setdiff1d(np.arange(NpFine), fixed)

    # create free matrices
    Sf = S[free][:, free]
    Kf = K[free][:, free]
    Mf = M[free][:, free]
    Lf = (M * f)[free]

    for i in range(tot_time_steps):

        # reference system
        A = (1. / (tau ** 2)) * Mf + (1. / tau) * Sf + Kf
        b = Lf + (1. / tau) * Sf * U[1][free] + (2. / (tau ** 2)) * Mf * U[1][free] - \
            (1. / (tau ** 2)) * Mf * U[0][free]

        # solve system
        UFineFree = linalg.linSolve(A, b)
        UFineFull = np.zeros(NpFine)
        UFineFull[free] = UFineFree

        # append solution
        U[0] = U[1]
        U[1] = UFineFull

    return U[-1]


def standard_method(world, N, fine, tau, tot_time_steps, k, aFine, bFine, f):
    
    # coarse mesh parameters
    NWorldCoarse = np.array([N, N])
    NFine = np.array([fine, fine])
    NpFine = np.prod(NFine + 1)
    NCoarseElement = NFine // NWorldCoarse
    NWorldFine = world.NWorldCoarse * world.NCoarseElement
    bc = world.boundaryConditions
    world = World(NWorldCoarse, NCoarseElement, bc)

    NpCoarse = np.prod(NWorldCoarse + 1)

    def IPatchGenerator(i, N):
        return interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement, bc)

    b_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, bFine)
    a_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aFine / tau)

    # compute basis correctors
    lod = lod_wave.LodWave(b_coef, world, k, IPatchGenerator, a_coef)
    lod.compute_basis_correctors()

    # compute ms basis
    basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
    basis_correctors = lod.assembleBasisCorrectors()
    ms_basis = basis - basis_correctors

    # compute finescale solution correctors
    prev_fs_sol = ms_basis
    fs_solutions_new = []
    for i in range(tot_time_steps):
        print('Calculating correction at N = %d, i = %d' % (N, i))

        # solve localized system
        lod = lod_wave.LodWave(b_coef, world, k, IPatchGenerator, a_coef, prev_fs_sol)
        lod.solve_fs_system(localized=True)

        # store sparse solution
        prev_fs_sol = sparse.csc_matrix(np.array(np.column_stack(lod.fs_list)))
        fs_solutions_new.append(prev_fs_sol)

    # initial value
    Uo = np.zeros(NpCoarse)

    # coarse v^(-1) and v^0
    V = [Uo]
    V.append(Uo)

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

    return VFine + WFine, ms_basis * Uo



def gram_schmidt_rb(snapshots, TOL):
    A = NumpyVectorArray(np.array(snapshots), VectorSpaceInterface)
    A_ortho = gram_schmidt.gram_schmidt(A, rtol=TOL)
    return A_ortho.to_numpy()


def compute_2d_node(NWorldCoarse, node_index):
    N = NWorldCoarse[0]
    b = 0
    while node_index > N:
        b += 1
        node_index -= N + 1
    a = node_index

    return np.array([a, b])


def rb_svd_method(world, N, fine, tau, tot_time_steps, k, aFine, bFine, f, TOL):

    # coarse mesh parameters
    NWorldCoarse = np.array([N, N])
    NFine = np.array([fine, fine])
    NpFine = np.prod(NFine + 1)
    NCoarseElement = NFine // NWorldCoarse
    NWorldFine = world.NWorldCoarse * world.NCoarseElement
    bc = world.boundaryConditions
    world = World(NWorldCoarse, NCoarseElement, bc) # need this?

    NpCoarse = np.prod(NWorldCoarse + 1)

    def IPatchGenerator(i, N):
        return interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement, bc)

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
        for time_step in range(tot_time_steps):
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
            wn = np.array([np.squeeze(np.asarray(snapshots[n])) for n in range(time_step + 1)])
            if len(wn) > 1:
                u, s, vh = np.linalg.svd(wn.T)
                print(s)
                if np.any([s < TOL]):
                    print('Snapshots computed, %d selected' % len(snapshots))
                    break

        V = sparse.csc_matrix(gram_schmidt_rb(snapshots))

        for i in range(time_step + 1, tot_time_steps):
            fs_solutions_new[i][:, node] = V.T * ecT.compute_rb_node_correction(b_patch, a_patch, IPatch, fs_solutions_new[i-1][:, node], V)


    ### BEFORE HERE ###

    # initial value
    Uo = np.zeros(NpCoarse)

    # coarse v^(-1) and v^0
    V = [Uo]
    V.append(Uo)

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

    return VFine + WFine, ms_basis * Uo


def rb_method(world, N, fine, tau, tot_time_steps, k, aFine, bFine, f, no_rb_vecs, TOL=None):

    # coarse mesh parameters
    NWorldCoarse = np.array([N, N])
    NFine = np.array([fine, fine])
    NpFine = np.prod(NFine + 1)
    NCoarseElement = NFine // NWorldCoarse
    NWorldFine = world.NWorldCoarse * world.NCoarseElement
    bc = world.boundaryConditions
    world = World(NWorldCoarse, NCoarseElement, bc)

    if TOL is None:
        TOL = 1e-6

    NpCoarse = np.prod(NWorldCoarse + 1)

    def IPatchGenerator(i, N):
        return interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement, bc)

    b_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, bFine)
    a_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aFine / tau)

    # compute basis correctors
    lod = lod_wave.LodWave(b_coef, world, k, IPatchGenerator, a_coef)
    lod.compute_basis_correctors()

    # compute ms basis
    basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
    basis_correctors = lod.assembleBasisCorrectors()
    ms_basis = basis - basis_correctors

    fs_solutions_new = [np.zeros_like(ms_basis.toarray()) for i in range(tot_time_steps)]
    for node in range(NpCoarse):
        prev_fs_sol_new = ms_basis.toarray()[:, node]
        snapshots = []
        for time_step in range(no_rb_vecs):
            print('Calculating correction at N = %d, node = %d/%d, time_step = %d' % (N, node, NpCoarse, time_step))
            node_index_arr = compute_2d_node(world.NWorldCoarse, node)
            ecT = lod_node.nodeCorrector(world, k, node_index_arr)

            b_patch = b_coef.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
            a_patch = a_coef.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)

            IPatch = IPatchGenerator(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)

            fs_list = ecT.compute_localized_node_correction(b_patch, a_patch, IPatch, prev_fs_sol_new, node)
            prev_fs_sol_new = fs_list
            fs_solutions_new[time_step][:, node] = prev_fs_sol_new
            snapshots.append(prev_fs_sol_new)

        V = sparse.csc_matrix(gram_schmidt_rb(snapshots, TOL))

        for i in range(time_step + 1, tot_time_steps):
            fs_solutions_new[i][:, node] = V.T * ecT.compute_rb_node_correction(b_patch, a_patch, IPatch, fs_solutions_new[i-1][:, node], V)


    # initial value
    Uo = np.zeros(NpCoarse)

    # coarse v^(-1) and v^0
    V = [Uo]
    V.append(Uo)

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

    return VFine + WFine, ms_basis * Uo

def standard_fem(world, N, fine, tau, tot_time_steps, aFine, bFine, f):
    
    # mesh parameters
    NWorldCoarse = np.array([N, N])
    NFine = np.array([fine, fine])
    NCoarseElement = NFine // NWorldCoarse
    NWorldFine = world.NWorldCoarse * world.NCoarseElement
    bc = world.boundaryConditions
    world = World(NWorldCoarse, NCoarseElement, bc)

    NpCoarse = np.prod(NWorldCoarse + 1)    
    basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
    Uo = np.zeros(NpCoarse)
    

    # coarse v^(-1) and v^0
    V = [Uo]
    V.append(Uo)

    # compute ms matrices
    S = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, aFine)
    K = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, bFine)
    M = fem.assemblePatchMatrix(NWorldFine, world.MLocFine)

    free = util.interiorpIndexMap(NWorldCoarse)

    SmsFree = (basis.T * S * basis)[free][:, free]
    KmsFree = (basis.T * K * basis)[free][:, free]
    MmsFree = (basis.T * M * basis)[free][:, free]
    LmsFree = (basis.T * M * f)[free]

    for i in range(tot_time_steps):
        n = i + 1

        # linear system
        A = (1. / (tau ** 2)) * MmsFree + (1. / tau) * SmsFree + KmsFree
        b = LmsFree + (1. / tau) * SmsFree * V[n][free] + (2. / (tau ** 2)) * MmsFree * V[n][free] \
            - (1. / (tau ** 2)) * MmsFree * V[n - 1][free]

        # solve system
        VFree = linalg.linSolve(A, b)
        VFull = np.zeros(NpCoarse)
        VFull[free] = VFree

        # append solution for current time step
        V.append(VFull)
    return basis * V[-1]


def standard_lod(world, N, fine, tau, tot_time_steps, k, aFine, bFine, f, damp_corr=False, both_coef=False):
    
    # mesh parameters
    NWorldCoarse = np.array([N, N])
    NFine = np.array([fine, fine])
    NCoarseElement = NFine // NWorldCoarse
    NWorldFine = world.NWorldCoarse * world.NCoarseElement
    bc = world.boundaryConditions
    world = World(NWorldCoarse, NCoarseElement, bc)

    def IPatchGenerator(i, N):
        return interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement, bc)

    b_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, bFine)
    a_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aFine / tau)

    NpCoarse = np.prod(NWorldCoarse + 1)    
    basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)

    # compute basis correctors
    if damp_corr:
        b_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, np.zeros_like(bFine))
        a_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aFine / tau)
    elif both_coef:
        b_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, bFine)
        a_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aFine)
    else:
        a_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, np.zeros_like(aFine))
        b_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, bFine)
    
    lod = lod_wave.LodWave(b_coef, world, k, IPatchGenerator, a_coef)
    lod.compute_basis_correctors()

    # compute ms basis
    basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
    basis_correctors = lod.assembleBasisCorrectors()
    mod_basis = basis - basis_correctors

    Uo = np.zeros(NpCoarse)

    # coarse v^(-1) and v^0
    V = [Uo]
    V.append(Uo)

    # compute ms matrices
    S = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, aFine)
    K = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, bFine)
    M = fem.assemblePatchMatrix(NWorldFine, world.MLocFine)

    free = util.interiorpIndexMap(NWorldCoarse)

    SmsFree = (mod_basis.T * S * mod_basis)[free][:, free]
    KmsFree = (mod_basis.T * K * mod_basis)[free][:, free]
    MmsFree = (mod_basis.T * M * mod_basis)[free][:, free]
    LmsFree = (mod_basis.T * M * f)[free]

    for i in range(tot_time_steps):
        n = i + 1

        # linear system
        A = (1. / (tau ** 2)) * MmsFree + (1. / tau) * SmsFree + KmsFree
        b = LmsFree + (1. / tau) * SmsFree * V[n][free] + (2. / (tau ** 2)) * MmsFree * V[n][free] \
            - (1. / (tau ** 2)) * MmsFree * V[n - 1][free]

        # solve system
        VFree = linalg.linSolve(A, b)
        VFull = np.zeros(NpCoarse)
        VFull[free] = VFree

        # append solution for current time step
        V.append(VFull)
    return mod_basis * V[-1]



def rb_method_sparse(world, N, fine, tau, tot_time_steps, k, aFine, bFine, f, no_rb_vecs, TOL=None):

    # coarse mesh parameters
    NWorldCoarse = np.array([N, N])
    NFine = np.array([fine, fine])
    NpFine = np.prod(NFine + 1)
    NCoarseElement = NFine // NWorldCoarse
    NWorldFine = world.NWorldCoarse * world.NCoarseElement
    bc = world.boundaryConditions
    world = World(NWorldCoarse, NCoarseElement, bc)

    if TOL is None:
        TOL = 1e-6

    NpCoarse = np.prod(NWorldCoarse + 1)

    def IPatchGenerator(i, N):
        return interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement, bc)

    b_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, bFine)
    a_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aFine / tau)

    # compute basis correctors
    lod = lod_wave.LodWave(b_coef, world, k, IPatchGenerator, a_coef)
    lod.compute_basis_correctors()

    # compute ms basis
    basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
    basis_correctors = lod.assembleBasisCorrectors()
    ms_basis = basis - basis_correctors

    fs_solutions_new = [sparse.csc_matrix(np.zeros_like(ms_basis.toarray())) for i in range(tot_time_steps)]
    for node in range(NpCoarse):
        prev_fs_sol_new = ms_basis[:, node]
        snapshots = []
        for time_step in range(no_rb_vecs):
            print('Calculating correction at N = %d, node = %d/%d, time_step = %d' % (N, node, NpCoarse, time_step))
            node_index_arr = compute_2d_node(world.NWorldCoarse, node)
            ecT = lod_node.nodeCorrector(world, k, node_index_arr)

            b_patch = b_coef.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
            a_patch = a_coef.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)

            IPatch = IPatchGenerator(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)

            fs_list = ecT.compute_localized_node_correction_test(b_patch, a_patch, IPatch, prev_fs_sol_new, node)
            prev_fs_sol_new = sparse.csr_matrix(fs_list).T
            fs_solutions_new[time_step][:, node] = prev_fs_sol_new
            snapshots.append(fs_list)

        V = sparse.csc_matrix(gram_schmidt_rb(snapshots, TOL))

        for i in range(time_step + 1, tot_time_steps):
            fs_solutions_new[i][:, node] = sparse.csc_matrix(V.T * ecT.compute_rb_node_correction_test(b_patch, a_patch, IPatch, fs_solutions_new[i-1][:, node], V)).T


    # initial value
    Uo = np.zeros(NpCoarse)

    # coarse v^(-1) and v^0
    V = [Uo]
    V.append(Uo)

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

    return VFine + WFine, ms_basis * Uo


def rb_method_sparse_stop(world, N, fine, tau, tot_time_steps, k, aFine, bFine, f, no_rb_vecs, TOL=None):

    # coarse mesh parameters
    NWorldCoarse = np.array([N, N])
    NFine = np.array([fine, fine])
    NpFine = np.prod(NFine + 1)
    NCoarseElement = NFine // NWorldCoarse
    NWorldFine = world.NWorldCoarse * world.NCoarseElement
    bc = world.boundaryConditions
    world = World(NWorldCoarse, NCoarseElement, bc)

    if TOL is None:
        TOL = 1e-6

    NpCoarse = np.prod(NWorldCoarse + 1)

    def IPatchGenerator(i, N):
        return interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement, bc)

    b_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, bFine)
    a_coef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aFine / tau)

    # compute basis correctors
    lod = lod_wave.LodWave(b_coef, world, k, IPatchGenerator, a_coef)
    lod.compute_basis_correctors()

    # compute ms basis
    basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
    basis_correctors = lod.assembleBasisCorrectors()
    ms_basis = basis - basis_correctors

    fs_solutions_new = [sparse.csc_matrix(np.zeros_like(ms_basis.toarray())) for i in range(tot_time_steps)]
    for node in range(NpCoarse):
        prev_fs_sol_new = ms_basis[:, node]
        snapshots = []
        
        for time_step in range(no_rb_vecs):
            print('Calculating correction at N = %d, node = %d/%d, time_step = %d' % (N, node, NpCoarse, time_step))
            node_index_arr = compute_2d_node(world.NWorldCoarse, node)
            ecT = lod_node.nodeCorrector(world, k, node_index_arr)

            b_patch = b_coef.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
            a_patch = a_coef.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)

            IPatch = IPatchGenerator(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)

            fs_list = ecT.compute_localized_node_correction_test(b_patch, a_patch, IPatch, prev_fs_sol_new, node)
            prev_fs_sol_new = sparse.csr_matrix(fs_list).T
            fs_solutions_new[time_step][:, node] = prev_fs_sol_new
            snapshots.append(fs_list)

            V = sparse.csc_matrix(gram_schmidt_rb(snapshots, TOL))
            if V.get_shape()[0] == time_step:
                break

        for i in range(time_step + 1, tot_time_steps):
            fs_solutions_new[i][:, node] = sparse.csc_matrix(V.T * ecT.compute_rb_node_correction_test(b_patch, a_patch, IPatch, fs_solutions_new[i-1][:, node], V)).T


    # initial value
    Uo = np.zeros(NpCoarse)

    # coarse v^(-1) and v^0
    V = [Uo]
    V.append(Uo)

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

    return VFine + WFine, ms_basis * Uo