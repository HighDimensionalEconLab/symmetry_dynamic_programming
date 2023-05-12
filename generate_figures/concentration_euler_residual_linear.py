import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from utilities import plot_params

params = plot_params((8,3.5))

# Defining the output path
output_dir = "./figures"
plot_name = "concentration_euler_residual_linear"
output_path = output_dir + "/" + plot_name + ".pdf"

beta = 0.95
sigma = 0.005
a_1 = 1.0 #this alpha_1
gamma = 90
delta = 0.05
h_1 = -0.050461355001343744

def std_euler_res(N):
    num= beta*sigma*(a_1-gamma*(1-delta)*h_1) #equation 43
    return np.abs(num)/np.sqrt(N) 

def std_policy(N):
    num = h_1*sigma #equation 40
    return np.abs(num)/np.sqrt(N)

N_space = np.linspace(1,10000,1000)    
std_val_res = std_euler_res(N_space)
std_val_policy = std_policy(N_space)


plt.rcParams.update(params) 

ax_res = plt.subplot(121)
plt.plot(N_space, std_val_res)
plt.title(r"Std. Dev. of $\varepsilon(X;u)$")
ax_res.set_yscale('log')
ax_res.set_xscale('log')
ax_res.xaxis.set_ticks([1, 10, 100, 1000, 10000])
ax_res.yaxis.set_ticks([10e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6,1e-7])
plt.xlabel(r"$N$")

ax_pol = plt.subplot(122, sharey=ax_res, sharex=ax_res)
plt.plot(N_space, std_val_policy)
plt.title(r"Std. Dev. of $u(X')$ Errors")
plt.xlabel(r"$N$")
plt.tight_layout()

plt.savefig(output_path)
