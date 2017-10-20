import numpy as np

from .abc import Multipoles
from .helpers import transparent_numpy


def from_real_field(field, a, b, N=21, num=361):
    """
    Creates a EllipticMultipoles object from a real field

    using:
        F/F₀ = c₀/2 + Σ_n=1..N-1 [cₙ·cosh(nη+inΨ)/cosh(nη₀)]

    ATTN:
        + semi-major axis a is on x axis

    """
    f = np.sqrt(a ** 2 - b ** 2)  # 'focal distance' or 'linear eccentricity' ATTN != 'eccentricity'
    eta0 = np.arctanh(b / a)
    # Values of psi for numerical integration
    psi = np.linspace(-np.pi, np.pi, num)
    # Evaluate potential at points
    x = f * np.cosh(eta0) * np.cos(psi)
    y = f * np.sinh(eta0) * np.sin(psi)
    #        F = np.array(list(map(field, x, y)))   # simion doesn't like i×(x[n],y[n]) requests
    Fx, Fy = field(x, y)
    F = Fy + 1j * Fx  # create complex field
    # Find n coefficients
    # FIXME c0, relationship to potential, potential (n-1) shift
    # depends on whether we allow potential treatment in elliptical case
    n = np.expand_dims(np.arange(0, N), -1)
    c = (1 / np.pi) * np.trapz(F * np.cos(n * psi), psi)
    return EllipticMultipoles(c, a, b)


def from_complex_field(field, a, b, N=21, num=361):
    """
    Creates a EllipticMultipoles object from a real field

    using:
        F/F₀ = c₀/2 + Σ_n=1..N-1 [cₙ·cosh(nη+inΨ)/cosh(nη₀)]

    ATTN:
        + semi-major axis a is on x axis

    """
    f = np.sqrt(a ** 2 - b ** 2)  # 'focal distance' or 'linear eccentricity' ATTN != 'eccentricity'
    eta0 = np.arctanh(b / a)
    # Values of psi for numerical integration
    psi = np.linspace(-np.pi, np.pi, num)
    # Evaluate potential at points
    z = f * np.cosh(eta0 + 1j * psi)
    #        F = np.array(list(map(field, z)))   # simion doesn't like i×(x[n],y[n]) requests
    F = field(z)
    # Find n coefficients
    # FIXME c0, relationship to potential, potential (n-1) shift
    # depends on whether we allow potential treatment in elliptical case
    n = np.expand_dims(np.arange(0, N), -1)
    c = (1 / np.pi) * np.trapz(F * np.cos(n * psi), psi)
    #        c = (1/np.pi/2)*np.trapz( F*np.cos(n*psi), psi ) # ATTN test, should be wrong
    #        c[0] /= 2.0 # ATTN test, should be wrong
    return EllipticMultipoles(c, a, b)


def from_scalar_potential(potential, a, b, *, N=21, num=361):
    raise Exception('treatment of potential N/A for elliptical multipole expansion')

    # TODO This whole thing again
    f = np.sqrt(a ** 2 - b ** 2)  # 'focal distance' or 'linear eccentricity' ATTN != 'eccentricity'
    eta0 = np.arctanh(b / a)
    # Values of psi for numerical integration
    psi = np.linspace(-np.pi, np.pi, num)
    # Evaluate potential at points
    x = f * np.cosh(eta0) * np.cos(psi)
    y = f * np.sinh(eta0) * np.sin(psi)
    V = np.array(list(map(potential, x, y)))
    # Find n coefficients
    n = np.expand_dims(np.arange(1, N + 1), -1)
    an = (1 / np.pi) * np.trapz(V * np.cos(n * psi), psi)
    bn = (1 / np.pi) * np.trapz(V * np.sin(n * psi), psi)
    d0 = (1 / np.pi) * np.trapz(V, psi)
    multipoles = bn + 1j * an
    return EllipticMultipoles(multipoles, a, b, d0=d0)


class EllipticMultipoles(Multipoles):
    def __init__(self, complex_multipoles, a, b, *, d0=0):
        """
        :param a: semi-major axis, in our y-direction
        :param b: semi-minor axis, in our x-direction
        """
        self._a = a
        self._b = b
        super().__init__(complex_multipoles, d0)

    def __repr__(self):
        rep = 'EllipticMultipoles({0._c}, a={0._a}, b={0._b}, d0={0._d0})'
        return rep.format(self)

    def _repr_html_(self):
        rep = ('<h4>Elliptic multipole field</h4>'
               f'<em>Major-semi axis a: {self._a}, Minor-semi axis b: {self._b}</em>')
        return rep + self._multipole_table()

    @property
    def eccentricity(self):
        # TODO Move this into a 'coordinate system' object
        return np.sqrt(self._a ** 2 - self._b ** 2)

    @property
    def eta0(self):
        return np.arctanh(self._b / self._a)

    @transparent_numpy()
    def complex_field(self, z):
        """
        Complex field reconstructed from multipole components

        using:
            F/F₀ = c₀/2 + Σ_n=1..N-1 [cₙ·cosh(nη+inΨ)/cosh(nη₀)]

        ATTN:
            + semi-major axis a is on x axis

        :param z: complex variable z=x+iy
        :returns: complex field F(x+iy) = F_y(x,y) + i·F_x(x,y)
        """
        f = np.sqrt(self._a ** 2 - self._b ** 2)  # 'focal distance' or 'linear eccentricity' ATTN != 'eccentricity'
        eta0 = np.arctanh(self._b / self._a)
        w = np.arccosh(z / f)
        ws = np.expand_dims(w, -1)

        # self.n is 1-based, but here we need 0-based
        n = self.n - 1
        terms = self.c * np.cosh(n * ws) / np.cosh(n * eta0)
        terms = n * n * terms  # FUDGE FACTOR!!!!
        #        terms[0] = terms[0] / 2.0 # FUDGE FACTOR!!!!
        terms[0] = terms[0] / 2.0
        field = np.sum(terms, -1)
        return field

    @transparent_numpy()
    def complex_potential(self, z):
        """
        Complex potential reconstructed from multipole components

        :param z: complex variable z=x+iy
        :returns: complex field W(x+iy) = A_z(x,y) + i·V(x,y)
        """
        raise Exception('treatment of potential N/A for elliptical multipole expansion')

        # TODO: Redo this whole method
        w = np.arccosh(z / self.eccentricity)
        eta, psi = w.real, w.imag
        n = self.n
        a_terms = self.skew * np.cosh(n * eta) / np.cosh(n * self.eta0) * np.cos(n * psi)
        b_terms = self.normal * np.sinh(n * eta) / np.sinh(n * self.eta0) * np.sin(n * psi)
        pot = self._d0 / 2 + np.sum(a_terms + b_terms, -1)
        return np.atleast_1d(1j * pot)
