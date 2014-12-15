"""Tephigram transform and isopleth constants."""

from __future__ import (absolute_import, division, print_function)

# The specific heat capacity of dry air at a constant pressure,
# in units of J kg-1 K-1.
# TBC: This was originally set to 1.01e3
Cp = 1004.0

# Dimensionless ratio: Rd / Cp.
K = 0.286

# Conversion offset between degree Celsius and Kelvin.
KELVIN = 273.15

# The specific latent heat of vapourisation of water at 0 degC,
# in units of J kg-1.
L = 2.501e6

MA = 300.0

# The specific gas constant for dry air, in units of J kg-1 K-1.
Rd = 287.0

# The specific gas constant for water vapour, in units of J kg-1 K-1.
Rv = 461.0

# Dimensionless ratio: Rd / Rv.
E = 0.622

# Base surface pressure.
P_BASE = 1000.0


default = {
    'barbs_gutter': 0.1,
    'barbs_length': 7,
    'barbs_linewidth': 1.5,
    'barbs_zorder': 10,
    'hodograph_angle': 135,
    'hodograph_height': '20%',
    'hodograph_line': dict(linewidth=1, clip_on=True),
    'hodograph_loc': 2,
    'hodograph_marker': dict(color='red', s=40, marker='o', linewidth=0),
    'hodograph_ticklabels': dict(size=8, clip_on=True),
    'hodograph_width': '20%',
    'isobar_line': dict(color='blue', linewidth=0.5, clip_on=True),
    'isobar_min_theta': 0,
    'isobar_max_theta': 250,
    'isobar_nbins': None,
    'isobar_text': dict(size=8, color='blue', clip_on=True, va='bottom', ha='right'),
    'isobar_ticks': [1050, 1000, 950, 900, 850, 800, 700, 600, 500, 400,
                     300, 250, 200, 150, 100, 70, 50, 40, 30, 20, 10],
    'isopleth_picker': 3,
    'isopleth_zorder': 10,
    'legend_loc': 'upper right',
    'legend_zorder': 20,
    'mixing_ratio_line': dict(color='purple', linewidth=0.5, linestyle='--', clip_on=True),
    'mixing_ratio_text': dict(size=8, color='purple', clip_on=True, va='bottom', ha='right'),
    'mixing_ratio_min_pressure': 10,
    'mixing_ratio_max_pressure': P_BASE,
    'mixing_ratio_nbins': 10,
    'mixing_ratio_ticks': [.001, .002, .005, .01, .02, .03, .05, .1, .15, .2,
                           .3, .4, .5, .6, .8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0,
                           5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0,
                           18.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0,
                           48.0, 52.0, 56.0, 60.0, 68.0, 80.0],
    'mode_loc': 3,
    'mode_frameon': False,
    'mode_size': 9,
    'wet_adiabat_line': dict(color='orange', linewidth=0.5, clip_on=True),
    'wet_adiabat_min_temperature': -50,
    'wet_adiabat_max_pressure': P_BASE,
    'wet_adiabat_nbins': 10,
    'wet_adiabat_text': dict(size=8, color='orange', clip_on=True, va='top', ha='left'),
    'wet_adiabat_ticks': range(1, 61),
    }
