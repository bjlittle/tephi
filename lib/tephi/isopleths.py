"""
Tephigram isopleth support for generating and plotting tephigram lines,
environment profiles and barbs.

"""

from __future__ import (absolute_import, division, print_function)

from abc import ABCMeta, abstractmethod
from collections import namedtuple
import math
import matplotlib.artist
from matplotlib.collections import PathCollection
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from mpl_toolkits.axes_grid.inset_locator import AnchoredSizeLocator
from mpl_toolkits.axisartist import Subplot
import numpy as np
from shapely.geometry import LineString
from scipy.interpolate import interp1d

import tephi.constants as constants
from tephi.constants import default
import tephi.transforms as transforms


# Wind barb speed (knots) ranges used since 1 January 1955.
_BARB_BINS = np.arange(20) * 5 + 3
_BARB_DTYPE = np.dtype(dict(names=('speed', 'angle', 'pressure', 'barb'),
                            formats=('f4', 'f4', 'f4', np.object)))

# Isopleth defaults.
_DRY_ADIABAT_STEPS = 50
_HUMIDITY_MIXING_RATIO_STEPS = 50
_ISOBAR_STEPS = 50
_ISOTHERM_STEPS = 50
_SATURATION_ADIABAT_PRESSURE_DELTA = -5.0

BOUNDS = namedtuple('BOUNDS', 'lower upper')
POINTS = namedtuple('POINTS', 'temperature theta pressure')


class BarbArtist(matplotlib.artist.Artist):
    def __init__(self, barbs, **kwargs):
        super(BarbArtist, self).__init__()
        self._gutter = kwargs.pop('gutter', default.get('barbs_gutter'))
        self._kwargs = dict(length=default.get('barbs_length'),
                            zorder=default.get('barbs_zorder', 10))
        self._kwargs.update(kwargs)
        self.set_zorder(self._kwargs['zorder'])
        self._path_kwargs = dict(color=None,
                                 linewidth=default.get('barbs_linewidth'),
                                 zorder=self._kwargs['zorder'])
        alias_by_kwarg = dict(color=['barbcolor', 'color',
                                     'edgecolor', 'facecolor'],
                              linewidth=['lw', 'linewidth'],
                              linestyle=['ls', 'linestyle'])
        for kwarg, alias in alias_by_kwarg.iteritems():
            common = set(alias).intersection(kwargs)
            if common:
                self._path_kwargs[kwarg] = kwargs[sorted(common)[0]]
        barbs = np.asarray(barbs)
        if barbs.ndim != 2 or barbs.shape[-1] != 3:
            msg = 'The barbs require to be a sequence of wind speed, ' \
                  'wind direction and pressure value triples.'
            raise ValueError(msg)
        self.barbs = np.empty(barbs.shape[0], dtype=_BARB_DTYPE)
        for i, barb in enumerate(barbs):
            self.barbs[i] = tuple(barb) + (None,)

    @staticmethod
    def _uv(magnitude, angle):
        """
        Convert magnitude and angle measured in degrees to u and v components,
        where u is -x and v is -y.

        """
        angle = angle % 360
        u = v = 0
        # Snap the magnitude of the barb vector to fall into one of the
        # _BARB_BINS ensuring it's a multiple of five. Five is the increment
        # step size for decorating with barb with flags.
        magnitude = np.searchsorted(_BARB_BINS, magnitude, side='right') * 5
        modulus = angle % 90
        if modulus:
            quadrant = int(angle / 90)
            radians = math.radians(modulus)
            y = math.cos(radians) * magnitude
            x = math.sin(radians) * magnitude
            if quadrant == 0:
                u, v = -x, -y
            elif quadrant == 1:
                u, v = -y, x
            elif quadrant == 2:
                u, v = x, y
            else:
                u, v = y, -x
        else:
            angle = int(angle)
            if angle == 0:
                v = -magnitude
            elif angle == 90:
                u = -magnitude
            elif angle == 180:
                v = magnitude
            else:
                u = magnitude
        return u, v

    def _make_barb(self, temperature, theta, speed, angle):
        """Add the barb to the plot at the specified location."""
        transform = self.get_axes().tephi['transform']
        u, v = self._uv(speed, angle)
        if 0 < speed < _BARB_BINS[0]:
            # Plot the missing barbless 1-2 knots line.
            length = self._kwargs['length']
            pivot_points = dict(tip=0.0, middle=-length / 2.)
            pivot = self._kwargs.get('pivot', 'tip')
            offset = pivot_points[pivot]
            verts = [(0.0, offset), (0.0, length + offset)]
            verts = mtrans.Affine2D().rotate(math.radians(-angle)).transform(verts)
            codes = [Path.MOVETO, Path.LINETO]
            path = Path(verts, codes)
            size = length ** 2 / 4
            xy = np.array([[temperature, theta]])
            barb = PathCollection([path], (size,), offsets=xy,
                                  transOffset=transform,
                                  **self._path_kwargs)
            barb.set_transform(mtrans.IdentityTransform())
        else:
            barb = self.axes.barbs(temperature, theta, u, v,
                                   transform=transform, **self._kwargs)
            self.axes.collections.remove(barb)
        return barb

    @matplotlib.artist.allow_rasterization
    def draw(self, renderer):
        if not self.get_visible():
            return
        axes = self.get_axes()
        x0, x1 = axes.get_xlim()
        y0, y1 = axes.get_ylim()
        y = np.linspace(y0, y1)[::-1]
        x = np.asarray([x1 - ((x1 - x0) * self._gutter)] * y.size)
        temperature, theta = transforms.convert_xy2Tt(x, y)
        pressure, _ = transforms.convert_Tt2pT(temperature, theta)
        min_pressure, max_pressure = np.min(pressure), np.max(pressure)
        func = interp1d(pressure, temperature)
        for i, (speed, angle, pressure, barb) in enumerate(self.barbs):
            if min_pressure < pressure < max_pressure:
                temperature, theta = transforms.convert_pT2Tt(pressure,
                                                              func(pressure))
                if barb is None:
                    barb = self._make_barb(temperature, theta, speed, angle)
                    self.barbs[i]['barb'] = barb
                else:
                    barb.set_offsets(np.array([[temperature, theta]]))
                barb.draw(renderer)


class Isopleth(object):
    __metaclass__ = ABCMeta

    def __init__(self, axes):
        self.axes = axes
        self._transform = axes.tephi['transform']
        self.points = self._generate_points()
        self.geometry = LineString(np.vstack((self.points.temperature,
                                              self.points.theta)).T)
        self.line = None
        self.label = None
        self._kwargs = dict(line={}, text={})
        Tmin, Tmax = (np.argmin(self.points.temperature),
                      np.argmax(self.points.temperature))
        tmin, tmax = (np.argmin(self.points.theta),
                      np.argmax(self.points.theta))
        pmin, pmax = (np.argmin(self.points.pressure),
                      np.argmax(self.points.pressure))
        self.index = POINTS(BOUNDS(Tmin, Tmax),
                            BOUNDS(tmin, tmax),
                            BOUNDS(pmin, pmax))
        self.extent = POINTS(BOUNDS(self.points.temperature[Tmin],
                                    self.points.temperature[Tmax]),
                             BOUNDS(self.points.theta[tmin],
                                    self.points.theta[tmax]),
                             BOUNDS(self.points.pressure[pmin],
                                    self.points.pressure[pmax]))

    @abstractmethod
    def _generate_points(self):
        pass

    def draw(self, renderer, **kwargs):
        if self.line is None:
            if 'zorder' not in kwargs:
                kwargs['zorder'] = default.get('isopleth_zorder')
            draw_kwargs = dict(self._kwargs['line'])
            draw_kwargs.update(kwargs)
            self.line = plt.Line2D(self.points.temperature, self.points.theta,
                                   transform=self._transform, **draw_kwargs)
            self.line.set_clip_box(self.axes.bbox)
        self.line.draw(renderer)
        return self.line

    def plot(self, **kwargs):
        """
        Plot the points of the isopleth.

        Kwargs:
            See :func:`matplotlib.pyplot.plot`.

        Returns:
            The isopleth :class:`matplotlib.lines.Line2D`

        """
        if self.line is not None:
            if self.line in self.axes.lines:
                self.axes.lines.remove(self.line)
        if 'zorder' not in kwargs:
            kwargs['zorder'] = default.get('isopleth_zorder')
        if 'picker' not in kwargs:
            kwargs['picker'] = default.get('isopleth_picker')
        plot_kwargs = dict(self._kwargs['line'])
        plot_kwargs.update(kwargs)
        self.line, = Subplot.plot(self.axes, self.points.temperature,
                                  self.points.theta, transform=self._transform,
                                  **plot_kwargs)
        return self.line

    def text(self, temperature, theta, text, **kwargs):
        if 'zorder' not in kwargs:
            kwargs['zorder'] = default.get('isopleth_zorder', 10) + 1
        text_kwargs = dict(self._kwargs['text'])
        text_kwargs.update(kwargs)
        if self.label is not None and self.label in self.axes.texts:
            self.axes.lines.remove(self.label)
        self.label = self.axes.text(temperature, theta, str(text),
                                    transform=self._transform, **text_kwargs)
        self.label.set_bbox(dict(boxstyle='Round,pad=0.3', facecolor='white',
                                 edgecolor='white', alpha=0.5, clip_on=True,
                                 clip_box=self.axes.bbox))
        return self.label

    def refresh(self, temperature, theta, renderer=None, **kwargs):
        if self.label is None:
            self.text(temperature, theta, self.data, **kwargs)
            if renderer is not None:
                self.axes.texts.remove(self.label)
        else:
            self.label.set_position((temperature, theta))
        if renderer is not None:
            self.label.draw(renderer)


class DryAdiabat(Isopleth):
    def __init__(self, axes, theta, min_pressure, max_pressure):
        self.data = theta
        self.bounds = BOUNDS(min_pressure, max_pressure)
        self._steps = _DRY_ADIABAT_STEPS
        super(DryAdiabat, self).__init__(axes)

    def _generate_points(self):
        pressure = np.linspace(self.bounds.lower, self.bounds.upper,
                               self._steps)
        theta = np.asarray([self.data] * self._steps)
        _, temperature = transforms.convert_pt2pT(pressure, theta)
        return POINTS(temperature, theta, pressure)


class Hodograph(object):
    def __init__(self, parent_axes, width=None, height=None, loc=None):
        self.parent_axes = parent_axes
        self.width = default.get('hodograph_width')
        if width is not None:
            self.width = width
        self.height = default.get('hodograph_height')
        if height is not None:
            self.height = height
        self.loc = default.get('hodograph_loc')
        if loc is not None:
            self.loc = loc
        self.axes = None
        self._items = {}

    def locator(self, width, height, loc):
        self.width = width
        self.height = height
        self.loc = loc
        if self.axes is None:
            self._create_axes()
        else:
            locator = AnchordSizeLocator(self.parent_axes.bbox, width,
                                         height, loc)
            self.axes.set_axes_locator(locator)

    def _create_axes(self):
        if self.axes is None:
            self.axes = plt.axes((0, 0, 1, 1), projection='polar',
                                 theta_direction='clockwise',
                                 theta_offset=np.pi / 2.)
            locator = AnchoredSizeLocator(self.parent_axes.bbox, self.width,
                                          self.height, self.loc)
            self.axes.set_axes_locator(locator)
            self.axes.set_aspect('equal')
            self.axes.set_navigate(False)
            self.axes.tick_params(labelbottom='off')
            self.axes.yaxis.get_major_locator().base.set_params(prune='upper')
            self.parent_axes.figure.add_axes(self.axes)
            angle = default.get('hodograph_angle')
            kwargs = default.get('hodograph_ticklabels')
            self.ticklabels(angle, **kwargs)

    def ticklabels(self, angle, **kwargs):
        self.axes.set_rlabel_position(angle)
        if 'zorder' not in kwargs:
            kwargs['zorder'] = default.get('isopleth_zorder', 10) + 2
        for i, tick in enumerate(self.axes.yaxis.get_ticklabels()):
            if i % 2 == 0:
                tick.update(kwargs)
            else:
                tick.set_visible(False)

    def plot(self, profile, line=None, marker=None):
        if self.axes is None:
            self._create_axes()
        pid = id(profile)
        item = self._items.get(pid)
        if item is not None:
            hline, hmarker = item
            self.axes.lines.remove(hline)
            self.axes.collections.remove(hmarker)

        line_kwargs = default.get('hodograph_line')
        if line is not None:
            line_kwargs.update(line)
        if 'color' not in line_kwargs:
            line_kwargs['color'] = profile.line.get_color()
        if 'linestyle' not in line_kwargs:
            line_kwargs['linestyle'] = profile.line.get_linestyle()
        if 'zorder' not in line_kwargs:
            line_kwargs['zorder'] = default.get('isopleth_zorder')
        marker_kwargs = default.get('hodograph_marker')
        if marker is not None:
            marker_kwargs.update(marker)
        if 'zorder' not in marker_kwargs:
            marker_kwargs['zorder'] = default.get('isopleth_zorder', 10) + 1

        barbs = profile.get_barbs()
        if barbs is not None:
            barbs = np.sort(barbs, order='pressure')[::-1]
            speed, angle = barbs['speed'], (barbs['angle'] + 180) % 360
            hline, = self.axes.plot(angle * np.pi / 180., speed, **line_kwargs)
            hmarker = self.axes.scatter(angle[0], speed[0], **marker_kwargs)
            self._items[pid] = (hline, hmarker)


class HumidityMixingRatio(Isopleth):
    def __init__(self, axes, mixing_ratio, min_pressure, max_pressure):
        self.data = mixing_ratio
        self.bounds = BOUNDS(min_pressure, max_pressure)
        self._step = _HUMIDITY_MIXING_RATIO_STEPS
        super(HumidityMixingRatio, self).__init__(axes)

    def _generate_points(self):
        pressure = np.linspace(self.bounds.lower, self.bounds.upper, self._step)
        temperature = transforms.convert_pw2T(pressure, self.data)
        _, theta = transforms.convert_pT2Tt(pressure, temperature)
        return POINTS(temperature, theta, pressure)


class Isobar(Isopleth):
    def __init__(self, axes, pressure, min_theta, max_theta):
        self.data = pressure
        self.bounds = BOUNDS(min_theta, max_theta)
        self._steps = _ISOBAR_STEPS
        super(Isobar, self).__init__(axes)
        self._kwargs['line'] = default.get('isobar_line')
        self._kwargs['text'] = default.get('isobar_text')

    def _generate_points(self):
        pressure = np.asarray([self.data] * self._steps)
        theta = np.linspace(self.bounds.lower, self.bounds.upper, self._steps)
        _, temperature = transforms.convert_pt2pT(pressure, theta)
        return POINTS(temperature, theta, pressure)


class Isotherm(Isopleth):
    def __init__(self, axes, temperature, min_pressure, max_pressure):
        self.data = temperature
        self.bounds = BOUNDS(min_pressure, max_pressure)
        self._steps = _ISOTHERM_STEPS
        super(Isotherm, self).__init__(axes)

    def _generate_points(self):
        pressure = np.linspace(self.bounds.lower, self.bounds.upper, self._steps)
        temperature = np.asarray([self.data] * self._steps)
        _, theta = transforms.convert_pT2Tt(pressure, temperature)
        return POINTS(temperature, theta, pressure)


class Profile(Isopleth):
    def __init__(self, axes, data):
        """
        Create a profile from the sequence of pressure and temperature points.

        Args:

        * axes:
            The tephigram axes on which to plot the profile.

        * data:
            Sequence of pressure and temperature points defining
            the profile.

        """
        self.data = np.asarray(data)
        super(Profile, self).__init__(axes)
        self._barbs = None
        self._highlight = None

    def has_highlight(self):
        return self._highlight is not None

    def highlight(self, state=None):
        if state is None:
            state = not self.has_highlight()
        if state:
            if self._highlight is None:
                linewidth = self.line.get_linewidth() * 7
                zorder = default.get('isopleth_zorder', 10) - 1
                kwargs = dict(linewidth=linewidth, color='grey', alpha=0.3,
                              transform=self._transform, zorder=zorder)
                self._highlight, = Subplot.plot(self.axes,
                                                self.points.temperature,
                                                self.points.theta,
                                                **kwargs)
        else:
            if self._highlight is not None:
                self.axes.lines.remove(self._highlight)
                self._highlight = None

    def _generate_points(self):
        if self.data.ndim != 2 or self.data.shape[-1] != 2:
            msg = 'The profile data requires to be a sequence ' \
                  'of pressure, temperature value pairs.'
            raise ValueError(msg)

        pressure = self.data[:, 0]
        temperature = self.data[:, 1]
        _, theta = transforms.convert_pT2Tt(pressure, temperature)
        return POINTS(temperature, theta, pressure)

    def barbs(self, barbs, hodograph=None, **kwargs):
        """
        Plot the sequence of barbs associated with this profile.

        Args:

        * barbs:
            Sequence of speed, direction and pressure value triples for
            each barb. Where speed is measured in units of knots, direction
            in units of degrees (clockwise from north), and pressure must
            be in units of mb or hPa.

        Kwargs:
        
        * hodograph:
            Flag whether to plot a hodograph for the profile barbs.

        * kwargs:
            See :func:`matplotlib.pyplot.barbs`

        """
        colors = ['color', 'barbcolor', 'edgecolor', 'facecolor']
        if not set(colors).intersection(kwargs):
            kwargs['color'] = self.line.get_color()
        self._barbs = BarbArtist(barbs, **kwargs)
        self.axes.add_artist(self._barbs)
        if bool(hodograph):
            self.hodograph()

    def hodograph(self, line=None, marker=None):
        self.axes.hodograph.plot(self, line=line, marker=marker)

    def get_barbs(self):
        return self._barbs.barbs


class WetAdiabat(Isopleth):
    def __init__(self, axes, theta_e, min_temperature, max_pressure):
        self.data = theta_e
        self.bounds = BOUNDS(min_temperature, max_pressure)
        self._delta_pressure = _SATURATION_ADIABAT_PRESSURE_DELTA
        super(WetAdiabat, self).__init__(axes)

    def _gradient(self, pressure, temperature, dp):
        stop = False

        kelvin = temperature + constants.KELVIN
        lsbc = (constants.L / constants.Rv) * ((1.0 / constants.KELVIN) - (1.0 / kelvin))
        rw = 6.11 * np.exp(lsbc) * (constants.E / pressure)
        lrwbt = (constants.L * rw) / (constants.Rd * kelvin)
        numerator = ((constants.Rd * kelvin) / (constants.Cp * pressure)) * (1.0 + lrwbt)
        denominator = 1.0 + (lrwbt * ((constants.E * constants.L) / (constants.Cp * kelvin)))
        grad = numerator / denominator
        dt = dp * grad

        if (temperature + dt) < self.bounds.lower:
            dt = self.bounds.lower - temperature
            dp = dt / grad
            stop = True

        return dp, dt, stop

    def _generate_points(self):
        temperature = [self.data]
        pressure = [self.bounds.upper]
        stop = False
        dp = self._delta_pressure

        while not stop:
            dp, dT, stop = self._gradient(pressure[-1], temperature[-1], dp)
            pressure.append(pressure[-1] + dp)
            temperature.append(temperature[-1] + dT)

        _, theta = transforms.convert_pT2Tt(pressure, temperature)
        return POINTS(temperature, theta, pressure)


class ProfileList(list):
    def __new__(cls, profiles=None):
        profile_list = list.__new__(cls, profiles)
        if not all(isinstance(profile, Profile) for profile in profile_list):
            msg = 'All items in the list must be a Profile instance.'
            raise TypeError(msg)
        return profile_list

    def highlighted(self):
        profiles = [profile for profile in self if profile.has_highlight()]
        return profiles

    def picker(self, artist):
        result = None
        for profile in self:
            if profile.line == artist:
                result = profile
                break
        if result is None:
            raise ValueError('Picker cannot find the profile.')
        return result
