import matplotlib.pyplot as plt
import numpy as np

from tephi import TephiAxes


ax = TephiAxes(anchor=[(0, -10), (0, 100)])

dew = [[1006, 26.4], [924, 20.3], [900, 19.8], [850, 14.5], [800, 12.9],
       [755, 8.3], [710, -5], [700, -5.1], [600, -11.2], [500, -8.3],
       [470, -12.1], [459, -12.5], [400, -32.9], [300, -46], [250, -53]]
dew_profile = ax.plot(dew, color='blue', linewidth=2, linestyle='--',
                      label='Dew-point')

dry = [[1006.0, 30.0], [924.0, 22.0], [900.0, 21.0], [850.0, 18.0],
       [800.0, 16.0], [755.0, 12.0], [710.0, 12.0], [700.0, 11.0],
       [600.0, 4.0], [500.0, -4.0], [470.0, -7.0], [459.0, -7.0],
       [400.0, -13.0], [300.0, -29.0], [250.0, -38.0]]
dry_profile = ax.plot(dry, color='green', linewidth=2,
                      label='Dry-bulb')

barbs = [barb for barb in zip(np.linspace(0, 100, len(dry)),
                              np.linspace(0, 360, len(dry)),
                              np.asarray(dry)[:, 0])]

dry_profile.barbs(barbs, hodograph=True)

ax.add_isobars()
ax.add_wet_adiabats()
ax.add_humidity_mixing_ratios()

plt.show()
