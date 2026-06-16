"""Interpret inversion results using Jupyter Notebooks

"""
from shapely.ops import transform
import matplotlib.pyplot as plt
# import matplotlib
import numpy as np
import shapely
import shapely.plotting
import crtomo

import ipywidgets.widgets as widgets
from ipywidgets import Output
from IPython.display import display
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


def round_coordinates(geom, ndigits=3):

    def _round_coords(x, y, z=None):
        x = round(x, ndigits)
        y = round(y, ndigits)

        if z is not None:
            z = round(z, ndigits)

        return [c for c in (x, y, z) if c is not None]

    return transform(_round_coords, geom)


class notebook_interpreter():
    def __init__(self):
        mesh = crtomo.crt_grid()
        plotmgr = crtomo.pltMan(grid=mesh)
        rmag_seg = np.loadtxt('rmag_seg.dat')
        pid = plotmgr.parman.add_data(rmag_seg)

        toggle_add_poly = widgets.ToggleButton(
            value=True,
            description='Click to finish adding nodes to polygon',
        )

        toggle_tool = widgets.ToggleButtons(
            options=[
                'Polygon',
                'Line',
            ],
            description='What type of geometry to generate:',
            disabled=True,
            button_style='',
        )

        polygon_list = []
        linestring_list = []
        output = Output()

        current_polygon = None
        current_linestring = None
        xlim_orig = None
        ylim_orig = None

        def print_linestring_pycode():

            with output:
                print('linestring pycode')
                index = 0
                for entry in linestring_list:
                    pol = shapely.geometry.LineString(entry)
                    print("line{:02} = shapely.from_wkt('{}')".format(
                        index,
                        round_coordinates(pol).wkt
                    ))
                    index += 1

        def print_polygon_pycode():
            with output:
                index = 0
                for entry in polygon_list:
                    pol = shapely.geometry.Polygon(entry)
                    print("poly{:02} = shapely.from_wkt('{}')".format(
                        index,
                        round_coordinates(pol).wkt
                    ))
                    index += 1

        def print_pycodes(event):
            output.clear_output()
            with output:
                print('import shapely')

            print_linestring_pycode()
            print_polygon_pycode()

        btn_print_pycode = widgets.Button(
            description='Generate Polygons'
        )
        btn_print_pycode.on_click(print_pycodes)
        # plt.close('test')
        with plt.ioff():
            fig, axes = plt.subplots(2, 1, num='test', dpi=100, figsize=(12, 8))

        def set_axes_lims():
            global xlim_orig
            global ylim_orig
            
            plotmgr.plot_elements_to_ax(pid, ax=axes[0])
            xlim_orig = axes[0].get_xlim()
            ylim_orig = axes[0].get_ylim()
            
            xlim = list(axes[0].get_xlim())
            xlen = xlim[1] - xlim[0]
            xlim[0] -= xlen / 4
            xlim[1] += xlen / 4
            axes[0].set_xlim(xlim)

            ylim = list(axes[0].get_ylim())
            ylen = ylim[1] - ylim[0]
            ylim[0] -= ylen / 4
            ylim[1] += ylen / 4
            axes[0].set_ylim(ylim)

            for ax in axes:
                #ax.set_ylim(-20, 0)
                #ax.set_xlim(-5, 45)
                ax.set_aspect('equal')
            axes[1].set_xlim(axes[0].get_xlim())
            axes[1].set_ylim(axes[0].get_ylim())
            
        set_axes_lims()


        def plot_polies():
            base_poly = shapely.geometry.Polygon(
                [
                    [xlim_orig[0], ylim_orig[0]],
                    [xlim_orig[1], ylim_orig[0]],
                    [xlim_orig[1], ylim_orig[1]],
                    [xlim_orig[0], ylim_orig[1]],
                ]
            )
            ax = axes[1]
            index = 0
            for polygon in polygon_list:
                poly = shapely.geometry.Polygon(polygon)

                poly_crop = poly.intersection(base_poly)
                patch = shapely.plotting.patch_from_polygon(poly_crop, color=colors[index])
                ax.add_patch(patch)
                index += 1

        def plot_current_polygon(no_lims=False):
            ax = axes[0]
            
            xl = ax.get_xlim()
            yl = ax.get_ylim()
            ax.clear()
            set_axes_lims()
            ax.set_xlim(xl)
            ax.set_ylim(yl)
            if current_polygon is None:
                return
            poly = np.array(current_polygon)
            
            ax.scatter(
                poly[:, 0],
                poly[:, 1],
                color='k',
            )
            if poly.shape[0] == 1:
                return
            
            for index in range(-1, poly.shape[0] - 1):
                ax.plot(
                    [poly[index, 0], poly[index + 1, 0]],
                    [poly[index, 1], poly[index + 1, 1]],
                )


        def plot_current_linestring(no_lims=False):
            ax = axes[0]
            
            xl = ax.get_xlim()
            yl = ax.get_ylim()
            ax.clear()
            set_axes_lims()
            ax.set_xlim(xl)
            ax.set_ylim(yl)
            if current_linestring is None:
                return
            linestring = np.array(current_linestring)
            
            ax.scatter(
                linestring[:, 0],
                linestring[:, 1],
                color='k',
            )
            if linestring.shape[0] == 1:
                return

            ax.plot(
                linestring[:, 0], linestring[:, 1]
            )
            

        def event_handler(event):
            """Handle a click event

            """
            global current_polygon
            global current_linestring
            with output:
                # print(event)
                if event.button == 3 and event.inaxes is not None:
                    check_in_ax1 = (event.inaxes == axes[0])
                    if check_in_ax1:
                        # we only do something here
                        if toggle_add_poly.value:
                            if toggle_tool.value == 'Polygon':
                                if current_polygon is None:
                                    current_polygon = []
                                current_polygon += [[event.xdata, event.ydata]]
                                if len(current_polygon) > 2:
                                    test_poly = shapely.geometry.Polygon(current_polygon)
                                    if not test_poly.is_simple:
                                        current_polygon = current_polygon[0:-1]
                                    
                                plot_current_polygon(no_lims=True)
                            elif toggle_tool.value == 'Line':
                                if current_linestring is None:
                                   current_linestring = []
                                current_linestring += [[event.xdata, event.ydata]]
                                plot_current_linestring()
                                
        # fig.canvas.mpl_connect('key_press_event', event_handler)
        # fig.canvas.mpl_connect('button_release_event', event_handler)

        fig.canvas.mpl_connect('button_press_event', event_handler)
        fig.tight_layout()

        def on_toggle_change(toggle):
            global polygon_list
            global linestring_list
            global current_polygon
            global current_linestring
            global toggle_add_poly
            global toggle_button
                
            if toggle['name'] != 'value':
                return
            if toggle['new'] == False:
                # finish adding the geometry
                if toggle_tool.value == 'Polygon':
                    if current_polygon is not None:
                        
                        # we need at least 3 points for a polygon
                        if len(current_polygon) >= 3:
                            polygon_list += [current_polygon]
                        current_polygon = None
                        plot_current_polygon()
                        plot_polies()
                elif toggle_tool.value == 'Line':
                    if current_linestring is not None:
                        
                        if len(current_linestring) >= 2:
                            linestring_list += [current_linestring]
                        current_linestring = None
                toggle['owner'].description = 'Start'
                toggle_tool.disabled = False
            else:
                # we want to start a new polygon
                toggle['owner'].description = 'Finish'
                toggle_tool.disabled = True

        toggle_add_poly.observe(on_toggle_change)

        vbox = widgets.VBox([
            widgets.HBox([
                toggle_add_poly,
                toggle_tool,
                btn_print_pycode,
                
            ]),
            output,
        ])


        display(vbox)
        display(fig.canvas)
