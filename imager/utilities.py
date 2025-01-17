import matplotlib.pyplot as plt
import matplotlib.ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import axes_size
import math
from scipy.linalg import lapack
from .imager import *
from .source import *
import pathlib
import time
import matplotlib as mpl
import cycler

def hex_to_rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip("#")
    lv = len(value)
    return [int(value[i : i + lv // 3], 16) / 255 for i in range(0, lv, lv // 3)]


tab10 = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
]

tab10_rgb = [hex_to_rgb(h) for h in tab10]
mpl.rcParams["axes.prop_cycle"] = cycler.cycler("color", tab10_rgb)

# matplotlib RC parameters
plt.rcParams.update({"font.size": 30})

from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.interpolate._bsplines import make_interp_spline

def _check_points(points):
    descending_dimensions = []
    grid = []
    for i, p in enumerate(points):
        # early make points float
        # see https://github.com/scipy/scipy/pull/17230
        p = np.asarray(p, dtype=float)
        if not np.all(p[1:] > p[:-1]):
            if np.all(p[1:] < p[:-1]):
                # input is descending, so make it ascending
                descending_dimensions.append(i)
                p = np.flip(p)
            else:
                raise ValueError(
                    "The points in dimension %d must be strictly "
                    "ascending or descending" % i)
        # see https://github.com/scipy/scipy/issues/17716
        p = np.ascontiguousarray(p)
        grid.append(p)
    return tuple(grid), tuple(descending_dimensions)


def _check_dimensionality(points, values):
    if len(points) > values.ndim:
        raise ValueError("There are %d point arrays, but values has %d "
                         "dimensions" % (len(points), values.ndim))
    for i, p in enumerate(points):
        if not np.asarray(p).ndim == 1:
            raise ValueError("The points in dimension %d must be "
                             "1-dimensional" % i)
        if not values.shape[i] == len(p):
            raise ValueError("There are %d points and %d values in "
                             "dimension %d" % (len(p), values.shape[i], i))

class FormatScalarFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, fformat="%1.1f", offset=True, mathText=True):
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(
            self, useOffset=offset, useMathText=mathText
        )

    def _set_format(self):
        self.format = self.fformat
        if self._useMathText:
            # self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)
            self.format = r"$\mathdefault{%s}$" % self.format

def draw_2D_img(
    scenario,
    srcs: list = None,
    saveimg=False,
    fname=None,
    cbar_label=r"Activity (Ci)",
    ax=None,
    return_figure=False,
    draw_img=True,
    draw_contour=False,
    contour_src=None,
    text=None,
    grid=True,
    **kwargs,
):
    if "dpi" in kwargs.keys():
        dpi = kwargs.pop("dpi")
    else:
        dpi = 100

    if "cmap" not in kwargs.keys():
        kwargs["cmap"] = "magma"
    alpha = kwargs.pop("alpha") if "alpha" in kwargs.keys() else 1
    if "figsize" in kwargs.keys():
        figsize = kwargs.pop("figsize")
        assert isinstance(figsize, tuple) and np.size(figsize) == 2
    else:
        x_extent = scenario.extent[0, 0] - scenario.extent[0, 1]
        y_extent = scenario.extent[1, 0] - scenario.extent[1, 1]
        figsize = (15, 15 * y_extent / x_extent)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_xlim((scenario.extent[0, 0], scenario.extent[0, 1]))
    ax.set_ylim((scenario.extent[1, 0], scenario.extent[1, 1]))

    if srcs is None:
        srcs = scenario.srcs

    dist_srcs = [s for s in srcs if isinstance(s, DistributedSource)]

    if len(dist_srcs) > 0:
        dist_cbar = True
    else:
        dist_cbar = False

    if len(dist_srcs) > 0:
        summed_src = sum(dist_srcs)

        src_map = ax.pcolormesh(
            scenario.X_bound[:, :, 0],
            scenario.Y_bound[:, :, 0],
            summed_src.src_dist.sum(axis=2),
            alpha=alpha,
            **kwargs,
        )

    counts_map = ax.quiver(
        scenario.path[:-1, 0],
        scenario.path[:-1, 1],
        scenario.path[1:, 0] - scenario.path[:-1, 0],
        scenario.path[1:, 1] - scenario.path[:-1, 1],
        scenario.counts.sum(axis=1)[:-1],
        cmap="coolwarm",
        width=0.005,
        scale=None,
        headwidth=1.5,
        alpha=0.8 * alpha,
    )
    ax.set_xlabel("x(m)")
    ax.set_ylabel("y(m)")
    if grid is True:
        ax.grid("on", ls="-.", alpha=0.7)
    cax_count = append_axes(ax, size=0.3, pad=0.4)
    # Colorbar format
    fmt = FormatScalarFormatter("%.1f")
    counts_cbar = plt.colorbar(
        counts_map, cax=cax_count, orientation="vertical", format=fmt
    )
    counts_cbar.ax.yaxis.set_offset_position("left")

    from matplotlib.ticker import FormatStrFormatter

    counts_cbar.formatter.set_powerlimits((0, 0))
    counts_cbar.set_label("Counts")

    fmt = FormatScalarFormatter("%.1f")
    if dist_cbar is True:
        cax_act = append_axes(ax, size=0.3, pad=1.6)
        act_cbar = plt.colorbar(
            src_map, cax=cax_act, orientation="vertical", format=fmt
        )
        act_cbar.ax.yaxis.set_offset_position("left")
        act_cbar.formatter.set_powerlimits((0, 0))
        act_cbar.set_label(cbar_label)

    if draw_contour is True:
        assert contour_src is not None
        if len(contour_src.shape) == 3:
            contour_src = contour_src.sum(axis=2)
        mask = contour_src > 0
        lines = contour_rect_slow(mask, scenario.extent, scenario.vox_size)
        print(lines[1])
        for line in lines:
            ax.plot(line[1], line[0], linewidth=3, color="r")
    if text is not None:
        ax.annotate(
            text=text,
            xy=(0.02, 0.95),
            xycoords="axes fraction",
            color="white",
            fontsize=40,
        )

    if saveimg is True:
        assert fname is not None
        fpath = pathlib.Path(fname)
        path_dir = fpath.parent
        file_name = fpath.name

        path_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fname=str(fpath), dpi=dpi, bbox_inches="tight")
    if draw_img is True:
        plt.show()
    if return_figure is True:
        return fig


def contour_rect_slow(im, extent, vox_size):
    """Clear version"""
    pad = np.pad(im, [(1, 1), (1, 1)])  # zero padding
    im0 = np.abs(np.diff(pad, n=1, axis=0))[:, 1:]
    im1 = np.abs(np.diff(pad, n=1, axis=1))[1:, :]
    lines = []
    x_extent = extent[0][1] - extent[0][0]
    y_extent = extent[1][1] - extent[1][0]
    x_n_vox = x_extent / vox_size
    y_n_vox = y_extent / vox_size
    x_half = x_extent / 2
    y_half = y_extent / 2

    for ii, jj in np.ndindex(im0.shape):
        if im0[ii, jj] == 1:
            lines += [
                (
                    [
                        (ii) / y_n_vox * y_extent - y_half,
                        (ii) / y_n_vox * y_extent - y_half,
                    ],
                    [
                        (jj) / x_n_vox * x_extent - x_half,
                        (jj + 1) / x_n_vox * x_extent - x_half,
                    ],
                )
            ]
        if im1[ii, jj] == 1:
            lines += [
                (
                    [
                        (ii) / y_n_vox * y_extent - y_half,
                        (ii + 1) / y_n_vox * y_extent - y_half,
                    ],
                    [
                        (jj) / x_n_vox * x_extent - x_half,
                        (jj) / x_n_vox * x_extent - x_half,
                    ],
                )
            ]

    return lines

def plot_count_data(count_data, int_time=None, saveimg=False, fname=None, grid=True):
    """Draw and save the count data aggregated over detectors

    Args:
        count_data (list): list of dictionaries, each containing the count data and label]
        saveimg (bool): Defaults to False.
        fname (str): File directory where the figure is saved. Defaults to None.
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    x = np.arange(count_data[0]["counts"].shape[0])
    if int_time is not None:
        x = x * int_time
    for i, data in enumerate(count_data):
        counts = data["counts"]
        #label = data["label"]
        kwargs = data["kwargs"]
        uncertainty = data["uncertainty"] if "uncertainty" in data.keys() else None
        mean = data["mean"] if "mean" in data.keys() else None
        dpi = kwargs.pop("dpi") if "dpi" in kwargs.keys() else 100
        ax.plot(counts, alpha=0.8, **kwargs)
        if uncertainty is not None:
            ax.fill_between(
                np.arange(uncertainty["upper"].size),
                uncertainty["upper"].ravel(),
                uncertainty["lower"].ravel(),
                alpha=0.2,
                color=kwargs["color"],
            )
        if mean is not None:
            ax.plot(
                np.arange(mean["mean"].size),
                mean["mean"].ravel(),
                color=kwargs["color"],
                ls="--"
            )
    if int_time is None:
        ax.set_xlabel("Measurement number")
    else:
        ticks_x = matplotlib.ticker.FuncFormatter(
            lambda x, pos: "{0:g}".format(x * int_time)
        )
        ax.xaxis.set_major_formatter(ticks_x)
        ax.set_xlabel("Measurement time (s)")
    ax.set_ylabel("Counts")
    if grid is True:
        ax.grid("on", ls="-.", alpha=0.5)
    ax.legend()

    if saveimg is True:
        assert fname is not None
        fpath = pathlib.Path(fname)
        path_dir = fpath.parent
        file_name = fpath.name

        path_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fname=str(fpath), bbox_inches="tight")

    plt.show()

    pass

def append_axes(ax, size=0.1, pad=0.1, position="right", extend=False):
    fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height

    divider = make_axes_locatable(ax)
    margin_size = axes_size.Fixed(size)
    pad_size = axes_size.Fixed(pad)
    xsizes = [pad_size, margin_size]
    if position in ["top", "bottom"]:
        xsizes = xsizes[::-1]
    yhax = divider.append_axes(position, size=margin_size, pad=pad_size)

    def extend_ratio(ax):
        ax.figure.canvas.draw()
        orig_size = ax.get_position().size
        new_size = 0
        for itax in ax.figure.axes:
            new_size += itax.get_position().size

        return new_size / orig_size

    if position in ["right"]:
        divider.set_horizontal([axes_size.Fixed(width)] + xsizes)
        fig.set_size_inches(
            fig.get_size_inches()[0] * extend_ratio(ax)[0], fig.get_size_inches()[1]
        )
    elif position in ["left"]:
        divider.set_horizontal(xsizes[::-1] + [axes_size.Fixed(width)])
        fig.set_size_inches(
            fig.get_size_inches()[0] * extend_ratio(ax)[0], fig.get_size_inches()[1]
        )
    elif position in ["top"]:
        divider.set_vertical([axes_size.Fixed(height)] + xsizes[::-1])
        fig.set_size_inches(
            fig.get_size_inches()[0], fig.get_size_inches()[1] * extend_ratio(ax)[1]
        )
    elif position in ["bottom"]:
        divider.set_vertical(xsizes + [axes_size.Fixed(height)])
        fig.set_size_inches(
            fig.get_size_inches()[0], fig.get_size_inches()[1] * extend_ratio(ax)[1]
        )

    return yhax


class Units:
    def __init__(self):
        global si
        si = {
            -18: {"multiplier": 10**18, "prefix": "a"},
            -17: {"multiplier": 10**18, "prefix": "a"},
            -16: {"multiplier": 10**18, "prefix": "a"},
            -15: {"multiplier": 10**15, "prefix": "f"},
            -14: {"multiplier": 10**15, "prefix": "f"},
            -13: {"multiplier": 10**15, "prefix": "f"},
            -12: {"multiplier": 10**12, "prefix": "p"},
            -11: {"multiplier": 10**12, "prefix": "p"},
            -10: {"multiplier": 10**12, "prefix": "p"},
            -9: {"multiplier": 10**9, "prefix": "n"},
            -8: {"multiplier": 10**9, "prefix": "n"},
            -7: {"multiplier": 10**9, "prefix": "n"},
            -6: {"multiplier": 10**6, "prefix": r"$\mu$"},
            -5: {"multiplier": 10**6, "prefix": r"$\mu$"},
            -4: {"multiplier": 10**6, "prefix": r"$\mu$"},
            -3: {"multiplier": 10**3, "prefix": "m"},
            -2: {"multiplier": 10**2, "prefix": "c"},
            -1: {"multiplier": 10**1, "prefix": "d"},
            0: {"multiplier": 1, "prefix": ""},
            1: {"multiplier": 10**1, "prefix": "da"},
            2: {"multiplier": 10**3, "prefix": "k"},
            3: {"multiplier": 10**3, "prefix": "k"},
            4: {"multiplier": 10**3, "prefix": "k"},
            5: {"multiplier": 10**3, "prefix": "k"},
            6: {"multiplier": 10**6, "prefix": "M"},
            7: {"multiplier": 10**6, "prefix": "M"},
            8: {"multiplier": 10**6, "prefix": "M"},
            9: {"multiplier": 10**9, "prefix": "G"},
            10: {"multiplier": 10**9, "prefix": "G"},
            11: {"multiplier": 10**9, "prefix": "G"},
            12: {"multiplier": 10**12, "prefix": "T"},
            13: {"multiplier": 10**12, "prefix": "T"},
            14: {"multiplier": 10**12, "prefix": "T"},
            15: {"multiplier": 10**15, "prefix": "P"},
            16: {"multiplier": 10**15, "prefix": "P"},
            17: {"multiplier": 10**15, "prefix": "P"},
            18: {"multiplier": 10**18, "prefix": "E"},
        }

    def convert(self, number):
        # Checking if its negative or positive
        if number < 0:
            negative = True
        else:
            negative = False

        # if its negative converting to positive (math.log()....)
        if negative:
            number = number - (number * 2)

        # Taking the exponent
        exponent = int(math.log10(number))

        # Checking if it was negative converting it back to negative
        if negative:
            number = number - (number * 2)

        # If the exponent is smaler than 0 dividing the exponent with -1
        if exponent < 0:
            exponent = exponent - 1
            return [number * si[exponent]["multiplier"], si[exponent]["prefix"]]
        # If the exponent bigger than 0 just return it
        elif exponent > 0:
            return [number / si[exponent]["multiplier"], si[exponent]["prefix"]]
        # If the exponent is 0 than return only the value
        elif exponent == 0:
            return [number, ""]


def upper_triangular_to_symmetric(ut):
    n = ut.shape[0]
    inds = np.tri(n, k=-1, dtype=np.bool)
    ut[inds] = ut.T[inds]


def cholesky_to_inv(c):
    inv, info = lapack.dpotri(c)
    if info != 0:
        raise ValueError("dpotri failed on input {}".format(c))
    upper_triangular_to_symmetric(inv)
    return inv


def pd_inverse(m):
    cholesky, info = lapack.dpotrf(m)
    if info != 0:
        raise ValueError("dpotrf failed on input {}".format(m))
    inv = cholesky_to_inv(cholesky)
    return inv


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            print("Timer is already running, stop the timer first.")
            self.stop()
        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")

