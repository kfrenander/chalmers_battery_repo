import numpy as np
import pandas as pd
import scipy.interpolate
from scipy.integrate import odeint
import scipy.optimize as scopt
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.constants
import re
import os
import datetime as dt
import pandas as pd
plt.rcParams['axes.grid'] = True
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 16


F = scipy.constants.value('Faraday constant')
R = scipy.constants.value('molar gas constant')
