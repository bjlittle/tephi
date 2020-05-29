import matplotlib.pyplot as plt
import os.path

import tephi

dew_point = os.path.join(tephi.DATA_DIR, 'dews.txt')
dew_data = tephi.loadtxt(dew_point, column_titles=('pressure', 'dewpoint'))
dews = zip(dew_data.pressure, dew_data.dewpoint)

tephi.MIXING_RATIO_LINE.update({"color": "blue", "linestyle": "dotted", "linewidth": 1.0})
tephi.MIXING_RATIO_TEXT.update({"color": "blue"})
tephi.WET_ADIABAT_LINE.update({"color": "red", "linewidth": 1.0}) 
tephi.WET_ADIABAT_TEXT.update({"color": "red"})
tephi.ISOBAR_LINE.update({"color": "lightblue", "linestyle": "dashed", "linewidth": 1.0})
tephi.ISOBAR_TEXT.update({"color": "lightblue"})

tpg = tephi.Tephigram()
tpg.plot(dews)
plt.show()
