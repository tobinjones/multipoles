from abc import ABC, abstractmethod
from .helpers import transparent_numpy, plot_log_errors
from . import plots
import numpy as np

import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections.polar import PolarAxes


class Multipoles(ABC):
    def __init__(self, complex_multipoles, d0=0):
        self._c = np.asarray(complex_multipoles)
        self._d0 = d0

    @property
    def N(self):
        """
        Number of multipole components

        """
        return len(self._c)

    @property
    def n(self):
        """
        Multipole indexes (in the european convention)

        :returns: n=[1, 2, ..., N]
        """
        return np.arange(1, self.N + 1)

    @property
    def c(self):
        """
        Complex multipole coefficients

        :returns: list of [c_1, c_2, ..., c_N]
        """
        return self._c

    @property
    def skew(self):
        """
        Skew multipole coefficients
        """
        return self._c.imag

    @property
    def normal(self):
        """
        Normal multipole coefficients
        """
        return self._c.real

    @property
    def magnitude(self):
        """
        Magnitude of multipole coefficients
        """
        return np.abs(self._c)

    @property
    def phase(self):
        """
        Phase of multipole coefficients
        """
        return np.angle(self._c)

    @property
    def c_n(self):
        """
        Multipoles enumerated by their index

        :returns: list of (n, c_n) tuples where n=1,2,...
        """
        return zip(self.n, self.c)

    @abstractmethod
    def complex_field(self, z):
        """
        Complex field reconstructed from multipole components

        :param z: complex variable z=x+iy
        :returns: complex field F(x+iy) = F_y(x,y) + i·F_x(x,y)
        """
        raise NotImplementedError

    @abstractmethod
    def complex_potential(self, z):
        """
        Complex potential reconstructed from multipole components

        :param z: complex variable z=x+iy
        :returns: complex field W(x+iy) = A_z(x,y) + i·V(x,y)
        """
        pass

    @transparent_numpy()
    def cartesian_field(self, x, y):
        """
        Components of field reconstructed from multipole components

        :param x: x
        :param y: y
        :returns: tuple(F_x, F_y)
        """
        # Definition of complex argument
        z = x + 1j * y
        # Calculate complex field at z
        F = self.complex_field(z)
        # Return components of field
        return F.imag, F.real

    @transparent_numpy()
    def cartesian_scalar_potential(self, x, y):
        """
        Scalar potential reconstructed from multipole components

        :param x: x
        :param y: y
        :returns: V(x,y)
        """
        # Definition of complex argument
        z = x + 1j * y
        # Calculate complex potential at z
        W = self.complex_potential(z)
        # Return real component
        return W.imag

    def plot_field(self, ax, radius=None, bmax=None):
        # Default radius of plot to reference radius
        radius = radius or self.ref_radius
        plots.plot_complex_field_on_circle(self.complex_field, ax, radius)

    def plot_potential(self, ax, radius=None):
        raise NotImplementedError()

    def _multipole_table(self, headers=[]):
        format_exp = lambda a: '${0:.3f}\\times 10^{{{1}}}$'.format(*(lambda m,e: (float(m), f'{int(e):d}' if int(e)<0 else f'{int(e):d}\\;\\;' ))(*'{0:e}'.format(a).split('e')))
        row_tmpl = '<tr><td>{n}</td><td>{normal}</td><td>{skew}</td></tr>'
        rep = ['<table><thead style="background-color: #efefef">']
        rep += ['<tr><th colspan="3">{0}</th></tr>'.format(h) for h in headers]
        rep += ['<tr><th>$n$</th><th>Normal $\mathscr{R}\{c_n\}$</th>']
        rep += ['<th>Skew $\mathscr{I}\{c_n\}$</th></tr></thead>']
        rep += ['<tbody>']
        rep += [row_tmpl.format(n=n, normal=format_exp(cn.real), skew=format_exp(cn.imag)) for n, cn in self.c_n]
        rep += ['</tbody>']
        rep += ['</table>']
        return '\n'.join(rep)