import numpy as np

basins_us = np.loadtxt(
    "/home/bernhard/git/Master-Thesis/basin_lists/basin_list_us.txt", dtype="str"
).tolist()
for i, basin in enumerate(basins_us):
    basins_us[i] = "us_" + basin

np.savetxt(
    "/home/bernhard/git/Master-Thesis/basin_lists/basin_list_us_specified.txt",
    basins_us,
    fmt="%s",
)

basins_gb = np.loadtxt(
    "/home/bernhard/git/Master-Thesis/basin_lists/basin_list_gb.txt", dtype="str"
).tolist()
for i, basin in enumerate(basins_gb):
    basins_gb[i] = "gb_" + basin

np.savetxt(
    "/home/bernhard/git/Master-Thesis/basin_lists/basin_list_gb_specified.txt",
    basins_gb,
    fmt="%s",
)
