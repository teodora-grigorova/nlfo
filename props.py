import numpy as np
from scipy import integrate

import imports

import peakutils
import plotting
import nlfo_math
import nlfo_functions

class Prop(object):

    def __init__(self, filename, sim_type):

        self.type = sim_type

        if self.type == "fnfep":
            prop_data = imports.load_fnfep_prop(filename, (-500, 500), (100, 10000))
        elif self.type == "luna":
            prop_data = imports.load_luna_sim(filename, stats=True)
        elif self.type == "luna_freespace":
            prop_data = imports.load_luna_freespacesim(filename, stats=True)
        else:
            raise AttributeError("The sim type should be given ('fnfep', 'luna', or 'luna_freespace').")

        self.λ = prop_data["λ"]
        self.angles = prop_data["angles"]
        self.prop = prop_data["Iλ"]
        self.stats = prop_data["stats"]





