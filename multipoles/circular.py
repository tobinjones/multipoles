from itertools import zip_longest
from types import SimpleNamespace

import numpy as np
from scipy.special import binom

from .abc import Multipoles
from .helpers import transparent_numpy
from . import plots

class CircularMultipoles(Multipoles):
    def __init__(self, complex_coefficients, ref_radius, d0=0):
        self._r = ref_radius
        super().__init__(complex_coefficients, d0)

    def __repr__(self):
        return f'CircularMultipoles({self._c}, ref_radius={self._r})'

    def _repr_html_(self):
        head = ['<h4>Circular Multipole Field</h4>',
                f'<em>Reference radius: {self._r}</em>']
        return self._multipole_table(headers=head)

    #
    # Properties of the field
    #

    @property
    def reference_radius(self):
        return self._r
    
    def centre(self, n):
        d = -self.reference_radius * self.coefficients[n - 2] / ((n - 1) * self.coefficients[n - 1])
        return d.real, d.imag
    
    def angle(self, n):
        angle = -(np.angle(self.coefficients[n-1]))/n
        return angle



    #
    # Fields
    #

    @transparent_numpy()
    def field(self, z):
        """
        Complex field reconstructed from multipole components

        :param z: complex variable z=x+iy
        :returns: complex field F(x+iy) = F_y(x,y) + i·F_x(x,y)
        """
        zs = np.expand_dims(z, -1)
        terms = self.coefficients * (zs / self._r) ** (self.indices - 1)
        field = np.sum(terms, -1)
        return field

    @transparent_numpy()
    def potential(self, z):
        """
        Complex potential reconstructed from multipole components

        :param z: complex variable z=x+iy
        :returns: complex potential W(x+iy) = A_z(x,y) + i·V(x,y)
        """
        zs = np.expand_dims(z, -1)
        terms = - self.coefficients * (self._r / self.indices) * (zs / self._r) ** self.indices
        potential = np.sum(terms, -1) + self._d0
        return potential

    def error_relative_to_harmonic(self, n):
        """
        Returns field of ΔF/Fn as function of z

        Where Fn(z) is the nth-harmonic component of F(z), and ΔF(z) = F(z)-Fn(z). See Tanabe p222 for details.

        :param n: multipole field
        :return: error field E(z)
        """
        fundamental = self.coefficients[n - 1].real
        coeffs = self.coefficients / fundamental
 
        @transparent_numpy(skip=0)
        def field(z):
            z = np.expand_dims(z, -1)
            return np.sum(coeffs * (z / self.reference_radius) ** (self.indices - n), -1)

        return field

    #
    # Derived CircularMultipoles
    #
    
    def resampled(self, new_radius):
        """
        returns a copy of this multipole object, at a different reference radius
        """
        new_coefficients = self.coefficients * (new_radius / self.reference_radius) ** (self.indices - 1)
        return CircularMultipoles(new_coefficients, new_radius)
    
    at_reference_radius = resampled
    
    def translated(self, dx=0, dy=0):
        delta = -(dx + 1j * dy) / self.reference_radius
        c_new = np.zeros_like(self.coefficients, dtype=np.complex128)
        for n in range(1, self.order + 1):
            ks = np.arange(n, self.order + 1)
            c_new[n - 1] = np.sum(self.coefficients[n - 1:] * binom(ks - 1, n - 1) * delta ** (ks - n))
        return CircularMultipoles(c_new, self.reference_radius)
    
    def centralized(self, n):
        x0, y0 = self.centre(n)
        return self.translated(-x0, -y0)
    
    def rotated(self, theta):
        c_new = [c*np.exp(-1j*(n+1)*theta) for n, c in enumerate(self.coefficients)]
        return CircularMultipoles(c_new, self.reference_radius)
    
    def normalized(self, n):
        return self / self.normal_coefficients[n - 1]

    def with_sensitivities(self, sensitivities):
        """
        returns a copy of this multipole object, calibrated with sensitivities
        """
        new_coefficients = [c/s for c, s in zip(self.coefficients, sensitivities)]
        return CircularMultipoles(new_coefficients, self.reference_radius)
    
    @property
    def gradient(self):
        """
        Complex differential of the multipole

        :return: new multipole
        """
        d0 = self.coefficients[0]
        ns = self.indices[:-1]
        coefficients = ns * self.coefficients[1:] / self.reference_radius
        return CircularMultipoles(coefficients, self.reference_radius, d0=d0)
  
  
    #
    # Assembly Errors
    #
    def with_quad_pole_errors(self, radius, dr=0, da=0, dp_deg=0):
      """
      Calculates the harmonic errors arising from assembly errors of one pole.
      See page 75 of Tanabe for more information (this is all from there).
      Inputs are...
      aperture: the quad aperture (metres)
      dr: the radial offset (metres)
      da: the linear azimuthal offset (metres)
      dp: the angular azimuthal offset (degrees)
      """

      dp = dp_deg*np.pi/180
      
      # the table columns, copied in.
      n = np.arange(1,17)
      xr = np.array([-425e-1,-516e-1,-288e-1,676e-2,108e-1,445e-2,-104e-2,128e-2,125e-2,637e-3,-244e-3,266e-3,227e-3,126e-3,-55e-4,576e-4])/100
      xa = np.array([746e-2,214e-1,288e-1,231e-1,108e-1,287e-2,104e-2,156e-2,125e-2,581e-3,244e-3,279e-3,227e-3,123e-3,555e-4,582e-4])/100
      xp = np.array([176e-1,500e-1,660e-1,500e-1,191e-1,0,-306e-2,0,753e-3,0,-362e-3,0,928e-4,0,666e-4,0])/100

      # errors (the epsilons)
      er = dr/radius
      ea = da/radius
      ep = dp
      
      # normalized field errors 
      r_error = 1j*er*xr*n 
      a_error = ea*xa*n
      p_error = ep*xp
      normalized_error = CircularMultipoles((r_error+a_error+p_error)*np.exp(-1j*n*np.pi/4),  radius)
      fundamental = self.resampled(radius).normal_coefficients[1]
      
      # new field
      return self + normalized_error*fundamental
    
    #
    # Indexing allows extraction of only certain multipoles
    #
    def __getitem__(self, key):
        if isinstance(key,slice):
            raise NotImplementedError("Slicing is not supported")
        if key.real and key.imag:
            raise IndexError("Must be purely real or purely imaginary")
        if (key.real <= 0) and (key.imag <= 0):
            raise IndexError(f"{key} is out of range")
        index = int(abs(key))
        value = self.coefficients[index-1]
        component = value.real if key.real else 1j*value.imag
        new_coeffs = np.zeros(index, dtype=complex)
        new_coeffs[index-1] = component
        return CircularMultipoles(new_coeffs, self.reference_radius)
            
    #
    # Mathematical Operators
    #

    def __add__(self, other):
        """
        Multipole fields are superposable, so adding them just adds their coefficients
        The new field will have the same reference radius as the left-hand field

        :param other: multipole field
        :return: new multipole field
        """
        if not isinstance(other, CircularMultipoles):
            raise TypeError
        # Scale to equal radii
        if not self.reference_radius == other.reference_radius:
            other = other.resampled(self.reference_radius)
        # Add coefficients
        new_coeffs = [a + b for a, b in zip_longest(self.coefficients, other.coefficients, fillvalue=0)]
        return CircularMultipoles(new_coeffs, self.reference_radius)

    def __sub__(self, other):
        """
        Subtracting a multipole field is the same as adding its negation

        :param other:
        :return: new multipole field
        """
        negated_other = CircularMultipoles(-other.coefficients, other.reference_radius)
        return self + negated_other

    def __eq__(self, other):
        if not isinstance(other, CircularMultipoles):
            raise False
        # Scale to equal radii
        if not self.reference_radius == other.reference_radius:
            other = other.resampled(self.reference_radius)
        # Check all coefficients are equal
        return all(a == b for a, b in zip_longest(self.coefficients, other.coefficients, fillvalue=0))

    def __mul__(self, other):
        """
        A multipole field can be multiplied by a scalar factor

        :param other: scalar number
        :return: scaled multipole field
        """
        return CircularMultipoles(self.coefficients * other, self.reference_radius)

    # Multiplication of scalar and field is commutative
    # number*field == field*number
    __rmul__ = __mul__

    def __truediv__(self, other):
        """
        Dividing a field by a scalar is the same as multiplying by the inverse

        :param other: scalar number
        :return: scaled multipole field
        """
        return CircularMultipoles(self.coefficients / other, self.reference_radius)

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
    return from_complex_field_at_angles(F, phi, ref_radius, N)

def from_complex_field_at_angles(field_values, phi, ref_radius, N=21):
    # Find n coefficients
    n = np.expand_dims(np.arange(1, N + 1), -1)
    c = (1 / (2 * np.pi)) * np.trapz(field_values * np.exp(-1j * (n - 1) * phi), phi)
    return CircularMultipoles(c, ref_radius)    

def from_scalar_potential(potential, ref_radius, N=21, num=361):
    # Values of phi for numerical integration
    phi = np.linspace(0, 2 * np.pi, num)
    # Evaluate potential at points
    x = ref_radius * np.cos(phi)
    y = ref_radius * np.sin(phi)
    V = potential(x, y)
    return from_scalar_potential_at_angles(V, phi, ref_radius, N)

def from_scalar_potential_at_angles(potentials, phi, ref_radius, N=21):
    """ Calculate circular multipoles from a set of (potential, angle) pairs. 
    Phi is in radians
    """
    total_angle = (np.max(phi) - np.min(phi))
    # Find n coefficients
    n_h = np.arange(0, N + 1)
    n_v = np.expand_dims(n_h, -1)
    d = (1 / total_angle) * np.trapz(potentials * np.exp(-1j * n_v * phi), phi)
    c = -1j * 2 * d * n_h / ref_radius
    return CircularMultipoles(c[1:], ref_radius, d0=d[0])

def from_polar_coefficients(magnitudes, phases, ref_radius):
    complex_coefficients = magnitudes * np.exp(1j * phases)
    return CircularMultipoles(complex_coefficients, ref_radius)


def from_normal_and_skew(normal, skew, ref_radius):
    normal = np.asarray(normal, dtype=np.complex128)
    skew = np.asarray(skew, dtype=np.complex128)
    return CircularMultipoles(normal + 1j * skew, ref_radius)


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
