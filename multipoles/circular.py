import numpy as np
from .abc import Multipoles
from .helpers import transparent_numpy
from . import plots
from itertools import zip_longest
from scipy.special import binom

class CircularMultipoles(Multipoles):
    def __init__(self, complex_coefficients, ref_radius, d0=0):
        """Circular multipoles"""
        self._r = ref_radius
        super().__init__(complex_coefficients, d0)

    def __repr__(self):
        return f'CircularMultipoles({self._c}, ref_radius={self._r})'

    def _repr_html_(self):
        head = ['<h4>Circular Multipole Field</h4>',
                f'<em>Reference radius: {self._r}</em>']
        return self._multipole_table(headers=head)

    @property
    def ref_radius(self):
        return self._r

    @transparent_numpy()
    def complex_field(self, z):
        """
        Complex field reconstructed from multipole components

        :param z: complex variable z=x+iy
        :returns: complex field F(x+iy) = F_y(x,y) + i·F_x(x,y)
        """
        zs = np.expand_dims(z, -1)
        terms = self.c * (zs / self._r) ** (self.n - 1)
        field = np.sum(terms, -1)
        return field

    @transparent_numpy()
    def polar_field(self, r, phi):
        """
        polar_field field reconstructed from multipole components

        :param r: radius
        :param phi: angle
        :returns: Field components (F_r, F_phi)
        """
        # convert to complex
        zs = r*np.exp(1j*phi)
        field = self.complex_field(zs)
        #rotate field by phi to convert to radial/azimuthal
        field_rotated = field*np.exp(1j*phi)
        return field_rotated.imag, field_rotated.real

    @transparent_numpy()
    def cartesian_field(self, x, y):
        """
        polar_field field reconstructed from multipole components

        :param x: x
        :param y: y
        :returns: Field components (F_x, F_y)
        """
        field = self.complex_field(x+1j*y)
        return field.imag, field.real

    @transparent_numpy()
    def complex_potential(self, z):
        """
        Complex potential reconstructed from multipole components

        :param z: complex variable z=x+iy
        :returns: complex potential W(x+iy) = A_z(x,y) + i·V(x,y)
        """
        zs = np.expand_dims(z, -1)
        terms = - self.c * (self._r / self.n) * (zs / self._r)**self.n
        potential = np.sum(terms, -1) + self._d0
        return potential

    @property
    def field_at_origin(self):
        return self.complex_field(0)

    @transparent_numpy()
    def complex_field_error_2(self, z):
        f = self.complex_field(z)
        f0 = self.field_at_origin
        return (f-f0)/f0

    def complex_field_error(self, n):
        fundamental = self.c[n - 1].real
        coeffs = self.c / fundamental
        coeffs[:n] = 0
        @transparent_numpy(skip=0)
        def field(z):
            z = np.expand_dims(z, -1)
            return np.sum(coeffs*(z/self.ref_radius)**(self.n-n), -1)
        return field

    def centre(self, n):
        d = -self.ref_radius * self.c[n-2] / ((n-1)*self.c[n-1])
        return d.real, d.imag

    def resampled(self, new_radius):
        """returns a copy of this multipole object, at a different reference radius"""
        new_coefficients = self.c * (new_radius / self.ref_radius)**(self.n - 1)
        return CircularMultipoles(new_coefficients, new_radius)

    def translated(self, dx=0, dy=0):
        delta = -(dx + 1j * dy)/self.ref_radius
        c_new = np.zeros_like(self.c, dtype=np.complex128)
        for n in range(1, self.N + 1):
            ks = np.arange(n, self.N + 1)
            c_new[n - 1] = np.sum(self.c[n - 1:] * binom(ks - 1, n - 1) * delta ** (ks - n))
        return CircularMultipoles(c_new, self.ref_radius)

    def normalized(self, n):
        return self / self.normal[n-1]

    def plot_field_error(self, ax, n, scale=0, radius=None):
        radius = radius or self.ref_radius
        plots.plot_error_field_on_circle(self.complex_field_error(n), ax, radius, scale=scale,
                                         title=f"Iso-error plot, $\Delta B / B_{n}$ $(\\times 10^{scale})$")

    def plot_field_error_2(self, ax, scale=0, radius=None):
        radius = radius or self.ref_radius
        plots.plot_error_field_on_circle(self.complex_field_error_2, ax, radius, scale=scale, title=f"Iso-error plot")

    @property
    def gradient(self):
        d0 = self.c[0]
        ns = self.n[:-1]
        coefficients = ns * self.c[1:] / self.ref_radius
        return CircularMultipoles(coefficients, self.ref_radius, d0=d0 )

    def __add__(self, other):
        if not isinstance(other, CircularMultipoles):
            raise TypeError
        # Scale to equal radii
        if not self.ref_radius == other.ref_radius:
            other = other.resampled(self.ref_radius)
        # Add coefficients
        new_coeffs = [a+b for a,b in zip_longest(self.c, other.c, fillvalue=0)]
        return CircularMultipoles(new_coeffs, self.ref_radius)

    def __sub__(self, other):
        negated_other = CircularMultipoles(-other.c, other.ref_radius)
        return self + negated_other

    def __eq__(self, other):
        if not isinstance(other, CircularMultipoles):
            raise False
        # Scale to equal radii
        if not self.ref_radius == other.ref_radius:
            other = other.resampled(self.ref_radius)
        # Check all coefficients are equal
        return all(a==b for a,b in zip_longest(self.c, other.c, fillvalue=0))

    def __mul__(self, other):
        return CircularMultipoles(self.c*other, self.ref_radius)
    __rmul__ = __mul__

    def __truediv__(self, other):
        return CircularMultipoles(self.c/other, self.ref_radius)

    def __ne__(self, other):
        return not self.__eq__(other)


def from_complex_field(field, ref_radius, N=21, num=361):
    """
    Creates a CircularMultipole object from a complex field, such as B=By+iBx
    """
    # Values of phi for numerical integration
    phi = np.linspace(0, 2 * np.pi, num)
    # Evaluate field at points
    z = ref_radius * (np.cos(phi) + 1j * np.sin(phi))
    F = np.array(list(map(field, z)))
    # Find n coefficients
    n = np.expand_dims(np.arange(1, N + 1), -1)
    c = (1 / (2 * np.pi)) * np.trapz(F * np.exp(-1j * (n - 1) * phi), phi)
    return CircularMultipoles(c, ref_radius)


def from_scalar_potential(potential, ref_radius, N=21, num=361):
    # Values of phi for numerical integration
    phi = np.linspace(0, 2 * np.pi, num)
    # Evaluate potential at points
    x = ref_radius * np.cos(phi)
    y = ref_radius * np.sin(phi)
    V = potential(x, y)
    # Find n coefficients
    n_h = np.arange(0, N + 1)
    n_v = np.expand_dims(n_h, -1)
    d = (1 / (2 * np.pi)) * np.trapz(V * np.exp(-1j * n_v * phi), phi)
    c = -1j * 2 * d * n_h / ref_radius
    return CircularMultipoles(c[1:], ref_radius, d0=d[0])

def from_polar_coefficients(magnitudes, phases, ref_radius):
    complex_coefficients = magnitudes * np.exp(1j * phases)
    return CircularMultipoles(complex_coefficients, ref_radius)

def from_normal_and_skew(normal, skew, ref_radius):
    normal = np.asarray(normal, dtype=np.complex128)
    skew = np.asarray(skew, dtype=np.complex128)
    return CircularMultipoles(normal+1j*skew, ref_radius)

def pure_field(n, strength=1):
    coeffs = np.zeros(n, dtype=np.complex128)
    coeffs[n - 1] = strength
    return CircularMultipoles(coeffs, ref_radius=1)

pure_dipole = pure_field(1, 1)
pure_quadrupole = pure_field(2, 1)
pure_sextupole = pure_field(3, 1)
pure_octopole = pure_field(4, 1)
skew_dipole = pure_field(1, 1j)
skew_quadrupole = pure_field(2, 1j)
skew_sextupole = pure_field(3, 1j)
skew_octopole = pure_field(4, 1j)