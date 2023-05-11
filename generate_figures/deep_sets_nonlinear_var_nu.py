
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

from utilities import first_satisfying_run, get_plot_params

#suggestion but it kinda is the first satisfying run that retcode = 0 
# test_results_by_tag

params = get_plot_params((6, 3.5), 10, 14)

output_dir = "./figures"
plot_name = "deep-sets-nonlinear-var-nu"
output_path = output_dir + "/" + plot_name + ".pdf"
project = "highdimensionaleconlab/symmetry_dynamic_programming"
                
df_nu_150 = first_satisfying_run(project,"deep_sets_nonlinear_nu_150_one_run")
df_nu_150 = df_nu_150[df_nu_150["ensemble"] == 0]

df_nu_130 = first_satisfying_run(project, "deep_sets_nonlinear_nu_130_one_run")
df_nu_130 = df_nu_130[df_nu_130["ensemble"] == 0]

df_nu_100 = first_satisfying_run(project, "baseline_deep_sets_one_run")
df_nu_100 = df_nu_100[df_nu_100["ensemble"] == 0]

plt.rcParams.update(params)

fig, ax = plt.subplots()
plt.plot(df_nu_150["t"], df_nu_150["u_hat"], label=r"$\nu = 1.5$")
plt.plot(df_nu_130["t"], df_nu_130["u_hat"], label=r"$\nu = 1.3$")
plt.plot(df_nu_100["t"], df_nu_100["u_hat"], label=r"$\nu = 1.0$")


plt.legend(prop={"size": params['font.size']})
plt.title(r"$u(X_t)$ with $\phi($ReLU$)$: Equilibrium Path")
plt.xlabel(r"Time(t)")
plt.tight_layout()

plt.show()
#plt.savefig(output_path)
#plt.clf()