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
from methods import rb_method, standard_method, rb_method_sparse

# fine mesh set up
fine = 256
NFine = np.array([fine, fine])
NpFine = np.prod(NFine + 1)
boundaryConditions = np.array([[0, 0], [0, 0]])
world = World(NFine, NFine // NFine, boundaryConditions)

# parameters
tau = 0.02
N = 32
tot_time_steps = 50
k = log(N, 2)
n = 2

np.random.seed(0)

# ms coefficient A
A = np.kron(np.random.rand(fine // n, fine // n) * 999.9 + 0.1, np.ones((n, n)))
aFine = A.flatten()

# ms coefficient B
B = np.kron(np.random.rand(fine // n, fine // n) * 999.9 + 0.1, np.ones((n, n)))
bFine = B.flatten()

rb_vec_list = range(2,tot_time_steps-1)
error = []
f = np.ones(NpFine) 
sol, init_val = standard_method(world, N, fine, tau, tot_time_steps, k, aFine, bFine, f)
for no_rb_vecs in rb_vec_list:
    rb_sol, init_val = rb_method_sparse(world, N, fine, tau, tot_time_steps, k, aFine, bFine, f, no_rb_vecs, 1e-8)
    error.append(np.sqrt(np.dot(np.gradient(rb_sol - sol), np.gradient(rb_sol - sol))) / np.sqrt(np.dot(np.gradient(sol), np.gradient(sol))))

print(error)
plt.figure('Error comparison', figsize=(16, 9)) 
plt.rc('text', usetex=True) 
plt.rc('font', family='serif') 
plt.tick_params(labelsize=24) 
plt.semilogy(rb_vec_list, error, '--s')
plt.grid(True, which="both") 
plt.xlabel('$M$', fontsize=24) 
plt.xticks([5,10,15,20])
plt.show()      





#errorfix = [1.1011300315189732, 0.6766610484183562, 0.4091231783248846, 0.25272169298171326, 0.15748747362841506, 0.08362349032301757, 0.03837004715581939, 0.018521156983903612, 0.009253350346329353, 0.004484116696781967, 0.001959564578758493, 0.0011463635033888532, 0.0007104083102308696, 0.00042171090316275156, 0.00026908111195390297, 0.00014649708339903048, 0.00010081483850969649, 6.260038709302683e-05, 4.0908374879173706e-05, 2.530023298243963e-05, 1.873089576671518e-05, 1.526490548384742e-05, 2.1849555466336532e-05, 1.9610302863159285e-05, 2.5686860348758342e-05, 3.1665038114750704e-05, 4.085608078874985e-05, 4.551956013527602e-05, 5.3659867220899614e-05, 5.865206454173325e-05, 7.367171952343957e-05, 7.451411459302922e-05, 7.6511080169737e-05, 8.436562755129965e-05, 8.524512884307578e-05, 0.00010939476092161587, 0.00010028133452364508, 8.758819725682779e-05, 7.815475814506417e-05, 8.369472606217654e-05, 6.527634361068212e-05, 5.475288693804039e-05, 3.846414718613574e-05, 2.3347355962676888e-05, 1.2928818931594122e-05, 6.42639229630842e-06, 1.8576537913092869e-06]