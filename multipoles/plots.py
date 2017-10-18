import numpy as np
from matplotlib import colors
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections.polar import PolarAxes
from contextlib import contextmanager


def semiradial_grid(radius, extension=0, number=100):
    limit = radius * (1 + extension)
    ys, xs = np.mgrid[-limit:limit:number * 1j, -limit:limit:number * 1j]
    mask = np.sqrt(xs ** 2 + ys ** 2) <= radius
    return xs, ys, mask

@contextmanager
def circular_plot(ax, radius, colorbar=True):
    divider = make_axes_locatable(ax)
    # Divide axis into two, for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.5) if colorbar else None
    # Set up cartesian axis
    ax_cartesian = ax
    ax_cartesian.set_frame_on(False)
    ax_cartesian.set_aspect('equal')
    ax_cartesian.set_xticks([])
    ax_cartesian.set_yticks([])
    # Polar axis is superimposed on cartesian axis for grid and labels
    ax_polar = inset_axes(ax, width='100%', height='100%', borderpad=0, axes_class=PolarAxes)
    ax_polar.patch.set_alpha(0)
    ax_polar.grid(color='k', alpha=0.2)
    ax_polar.plot([0,2*np.pi],[10,10],'r-')
    ax_polar.set_rmax(radius)
    ax_polar.xaxis.set_ticklabels([''])#[f'$\\theta=0$ \n $r={radius}$'])
    ax_polar.set_rlabel_position(0)
    ax_polar.set_rticks(np.linspace(0,radius,5))
    ax_polar.yaxis.set_ticklabels([''])#*4+[])
    ax_polar.text(0, radius*1.05, f'$\\theta=0$ \n  $r={radius}$', horizontalalignment='left', verticalalignment='center')
    # plot on these!
    yield (ax_cartesian, ax_polar, ax_colorbar) if colorbar else (ax_cartesian, ax_polar)
    # Clip everything to a circle
    paths_to_clip = ax_cartesian.get_children()
    circle_clip = patches.Circle((0, 0), radius, fc='none')
    ax_cartesian.add_patch(circle_clip)
    for p in paths_to_clip:
        p.set_clip_path(circle_clip)
    # Ensure limits are correct
    ax_cartesian.set_xlim(-radius, radius)
    ax_cartesian.set_ylim(-radius, radius)


def plot_complex_field_on_circle(field, ax, radius, vmax=None):
    with circular_plot(ax, radius) as axes:
        ax_cartesian, ax_polar, ax_colorbar = axes
        # Plot contours
        xs, ys, mask = semiradial_grid(radius, extension=0.1, number=201)
        field_dense = field(xs + 1j * ys)
        max_value = vmax or np.max(abs(field_dense[mask]))
        levels = np.linspace(0, max_value, 11)
        contours = ax_cartesian.contourf(xs, ys, abs(field_dense), levels, cmap='Blues', extend='max')
        ax_cartesian.get_figure().colorbar(contours, cax=ax_colorbar)
        # Plot arrows
        xs, ys, mask = semiradial_grid(radius, extension=0.1, number=19)
        field_sparse = field(xs + 1j * ys)
        field_x = np.sqrt(np.abs(field_sparse.imag)) * np.sign(field_sparse.imag)
        field_y = np.sqrt(np.abs(field_sparse.real)) * np.sign(field_sparse.real)
        quiver_args = dict(color='k', width=0.004, alpha=1, minshaft=2, pivot='middle')
        arrows = ax_cartesian.quiver(xs[mask], ys[mask], field_x[mask], field_y[mask], **quiver_args)


def plot_error_field_on_circle(field, ax, radius, levels=None, scale=1, title=None):
    with circular_plot(ax, radius, colorbar=False) as axes:
        ax_cartesian, ax_polar = axes
        xs, ys, mask = semiradial_grid(radius, extension=0.1, number=500)
        field_values = np.abs(field(xs + 1j * ys))*10**scale
        vmax = 10**int(np.log10(np.max(field_values[mask])))
        levels = np.geomspace(vmax/100,vmax,5) if levels is None else levels
        contours = ax_cartesian.contour(xs, ys, field_values, levels, norm=colors.LogNorm())
        ax.clabel(contours, inline=1, fmt='%.2f', fontsize='small')
        # Title
        ax_polar.set_title(title)


##### LEGACY -- TODO
def plot_potential(self, ax, xlim, ylim):
    xs, ys = self._plot_grids(xlim, ylim)
    W = self.complex_potential(xs + 1j * ys)
    ax.imshow(W.imag, origin='lower', extent=xlim + ylim, cmap='RdBu')
    lev_i = W.imag.max() * np.logspace(-2, 0, num=6)
    lev_i = np.append(-lev_i[::-1], lev_i)
    max_r = np.max(np.abs(W.real))
    lev_r = np.linspace(-max_r, max_r, 24)
    ax.contour(xs, ys, W.imag, lev_i, colors='k', linestyles='-')
    ax.contour(xs, ys, np.abs(W.real), lev_r, colors='w', linestyles='-')
    ax.set_aspect('equal')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def _plot_grids(self, xlim, ylim):
    left, right = xlim
    bottom, top = ylim
    xext = (right - left) / 10
    yext = (top - bottom) / 10
    ys, xs = np.mgrid[bottom - yext:top + yext:100j, left - xext:right + xext:100j]
    return xs, ys
