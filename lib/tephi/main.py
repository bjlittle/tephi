import matplotlib.pyplot as plt
import numpy as np

from tephi import TephiAxes


#    plt.ion()
ax = TephiAxes(anchor=[(0, -10), (0, 100)])
#    ax = TephiAxes()

data = [[1006, 26.4], [924, 20.3], [900, 19.8], [850, 14.5], [800, 12.9],
        [755, 8.3], [710, -5], [700, -5.1], [600, -11.2], [500, -8.3],
        [470, -12.1], [459, -12.5], [400, -32.9], [300, -46], [250, -53]]
profile = ax.plot(data)
barbs = [barb for barb in zip(np.linspace(0, 100, len(data)),
                              np.linspace(0, 360, len(data)),
                              np.asarray(data)[:, 0])]
profile.barbs(barbs, hodograph=True)

#    da = isopleths.DryAdiabat(ax, 50, 700, 1000)
#    da.plot()

#    isotherm = isopleths.Isotherm(ax, 5, 700, 1000)
#    isotherm.plot()

#    hmr = isopleths.HumidityMixingRatio(ax, 10, 700, 1000)
#    hmr.plot()

#    wa = isopleths.WetAdiabat(ax, 0, -40, 1000)
#    wa.plot()

ax.add_isobars()
ax.add_wet_adiabats()
ax.add_humidity_mixing_ratios()

plt.show()
