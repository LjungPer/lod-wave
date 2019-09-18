import numpy as np
import matplotlib.pyplot as plt
from gridlod import util, fem, coef, interp
from gridlod.world import World
import lod_wave
from visualize import d3plotter

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
tau = 0.05
n = 2

# ms coefficient B
B = np.kron(np.random.rand(fine // n, fine // n) * 999.9 + 0.1, np.ones((n, n)))
bFine = B.flatten()

# ms coefficient A
A = np.kron(np.random.rand(fine // n, fine // n) * 9.9 + 0.1, np.ones((n, n)))
aFine = A.flatten()


# localization and mesh width parameters
N = 8
k = np.inf

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

schauen = np.zeros(NpCoarse)
schauen[40] = 1
standard_hat = basis * schauen
corr_hat = basis_correctors * schauen
mod_hat = ms_basis * schauen
zmax = np.max(corr_hat)
zmin = np.min(corr_hat)
#d3plotter(NWorldFine, mod_hat, '3', zmax=zmax, zmin=zmin)
d3plotter(NWorldFine, corr_hat, '2', zmax=zmax, zmin=zmin, Blues=True)
plt.show()