import numpy as np

from crtomo.mpl_setup import *

import crtomo.tdManager as CRman
import crtomo.grid as CRGrid

grid = CRGrid.crt_grid.create_surface_grid(
    nr_electrodes=50,
    spacing=0.5,
    depth=25,
    right=20,
    left=20,
    char_lengths=[0.1, 5, 5, 5],
)
print(grid)

man = CRman.tdMan(grid=grid)
man.configs.gen_schlumberger(M=20, N=21)
K = man.configs.compute_K_factors(spacing=0.5)
print(man.configs.configs)

# pseudo depth after Knödel et al for Schlumberger configurations
pdepth = np.abs(
    np.max(
        man.configs.configs, axis=1
    ) - np.min(
        man.configs.configs, axis=1
    )
) * 0.19

fig, axes = plt.subplots(2, 1, figsize=(15 / 2.54, 10 / 2.54))

ax = axes[0]
for contrast in (2, 5, 10):
    pid_mag, pid_pha = man.add_homogeneous_model(magnitude=1000, phase=0)
    man.clear_measurements()

    man.parman.modify_area(pid_mag, -100, 100, -40, -3, 1000 / contrast)
    # man.parman.modify_area(pid_mag, -100, 100, -40, -10, 500 / contrast)

    ax.plot(pdepth, man.measurements()[:, 0] * K, '.-')

ax.axvline(x=3, color='k', linestyle='dashed')
# ax.axvline(x=10, color='k', linestyle='dashed')
ax.set_xlabel('pseudo depth [m]')
ax.set_ylabel('measurement [$\Omega$]')
ax.set_title('Schlumberger sounding for different layer resistivities')

ax = axes[1]
# grid.plot_grid_to_ax(ax)
man.plotman.plot_elements_to_ax(
    pid_mag,
    ax=ax,
    plot_colorbar=True,
)

fig.tight_layout()
fig.savefig('schlumberger_sounding.png', dpi=300)
