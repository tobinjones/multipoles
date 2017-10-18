import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
import numpy as np
from typing import Iterable
from functools import wraps, partial
import inspect

def transparent_numpy(skip=1):
    """
    Decorator lets a vectorized method/function to optionally take scalar arguments

    That means if any of the args are numpy arrays, the result will also be a
    numpy array, but if all the args are scalars, the result will be a scalar.
    Inside the function, the args will ALWAYS BE ARRAYS, so scalars will be
    wrapped into a 1-element array before, and transparently unwrapped after.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args):
            """Takes method, returns wrapped version"""
            # partially apply skipped arguments to function
            func_actual = partial(func, *args[:skip])
            args = args[skip:]
            # if all are already numpy arrays, do nothing
            if all(isinstance(a, np.ndarray) for a in args):
                return func_actual(*args)
            # cast args into numpy arrays
            nargs = [np.atleast_1d(a) for a in args]
            # function gets called with arrays, always
            res = func_actual(*nargs)
            # cast back to simple python types if all args were scalar
            if not any(isinstance(a, Iterable) for a in args):
                res = tuple(r.item() for r in res)
                if len(res) == 1:
                    res = res[0]
            return res

        return wrapper
    return decorator

_cmap_errors = copy.copy(mpl.pyplot.cm.gray)
_cmap_errors.set_over('orangered', .5)
_cmap_errors.set_under('skyblue', .3)
_cmap_errors.set_bad('g', 1.0)

def plot_log_errors(XX, YY, errors, ax, cax=None, exp_max=4):
    # Colormaps
    cmap_plus = mpl.pyplot.cm.Reds_r
    cmap_minus = mpl.pyplot.cm.Blues
    mask_cmap = mpl.colors.ListedColormap([(1, 1, 1, 0), (1, 1, 1, 1)])

    # Plots
    with np.errstate(invalid='ignore'):
        logs = (-np.log10(np.abs(errors)))
        signed = logs * np.sign(errors)
        mask = np.logical_or(signed > exp_max, signed < -exp_max)
    cont1 = ax.contourf(XX, YY, signed, range(-exp_max, 1), cmap=cmap_minus, )
    cont2 = ax.contourf(XX, YY, signed, range(exp_max + 1), cmap=cmap_plus)
    cont3 = ax.contourf(XX, YY, mask, 1, cmap=mask_cmap)
    ax.grid(True, color='k', alpha=.25)

    # colorbar
    if not cax: return

    normplus = mpl.colors.Normalize(vmin=0, vmax=exp_max)
    normminus = mpl.colors.Normalize(vmin=-exp_max, vmax=0)

    mapplus = plt.cm.ScalarMappable(norm=normplus, cmap=cmap_plus)
    mapminus = plt.cm.ScalarMappable(norm=normminus, cmap=cmap_minus)

    cmap = mpl.colors.ListedColormap([mapminus.to_rgba(x + 0.5) for x in range(-1, -exp_max - 1, -1)] +
                                     ['white'] +
                                     [mapplus.to_rgba(x - 0.5) for x in range(exp_max, 0, -1)])

    bounds = np.arange(-exp_max - 1, exp_max + 2)-0.5
    print(bounds)
    #bounds = [-4.5, -3.5, -2.5, -1.5, -.5, .5, 1.5, 2.5, 3.5, 4.5]

    labels = [r'$-1 \times 10^{' + str(x) + '}$' for x in range(-1, -exp_max - 1, -1)] + \
             [0] + [r'$+1 \times 10^{' + str(x) + '}$' for x in range(-exp_max, 0)]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                   boundaries=bounds,
                                   ticks=bounds,  # optional
                                   extend='both',
                                   spacing='proportional')
    cb.set_ticklabels(labels)
    return [cont1, cont2, cont3]