import h5py
import numpy as np

def load_sim(filename):
    sim = h5py.File(filename, "r")
    field = np.array(sim["Eω"])
    grid = sim["grid"]
    meta = sim["meta"]
    sim_type = sim["simulatio_type"] 
    stats = sim["stats"]
    z = np.array(sim["z"]
    sim.close()
    return field, grid, meta, sim_type, stats, z