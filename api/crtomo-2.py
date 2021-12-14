import crtomo.debug
import crtomo
grid = crtomo.debug.get_grid(key=20)
td = crtomo.tdMan(grid=grid)
td.configs.add_to_configs([1, 5, 9, 13])
cid_mag, cid_pha = td.add_homogeneous_model(25, 0)
td.register_forward_model(cid_mag, cid_pha)
td.model(sensitivities=True)
fig, axes = td.plot_sensitivity(0)
fig.tight_layout()
fig.savefig('sens_plot.pdf', bbox_inches='tight')