# (C) British Crown Copyright 2014 - 2017, Met Office
#
# This file is part of tephi.
#
# Tephi is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Tephi is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tephi.  If not, see <http://www.gnu.org/licenses/>.
"""Tephigram transform and isopleth constants."""
from __future__ import absolute_import, division, print_function


# The specific heat capacity of dry air at a constant pressure,
# in units of J kg-1 K-1.
Cp = 1004.0

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

# Dimensionless ratio: Rd / Cp
K = Rd / Cp

# Dimensionless ratio: Rd / Rv.
E = Rd / Rv

# Base surface pressure, in units of hPa.
P_BASE = 1000.0
