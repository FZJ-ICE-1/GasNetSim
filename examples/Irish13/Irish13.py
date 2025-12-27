#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2024.
#     Developed by Yifei Lu
#     Last change on 8/22/24, 9:38 AM
#     Last change by yifei
#    *****************************************************************************

# import sys
# import os
# sys.path.append(os.path.dirname(__file__))

from pathlib import Path
from timeit import default_timer as timer
import matplotlib.pyplot as plt

import GasNetSim as gns
from GasNetSim.components.utils.plot_functions import plot_network_pipeline_flow_results

network = gns.create_network_from_folder(Path("."))

# start = timer()
# network.simulation(use_cuda=True, tol=0.0001)
# end = timer()
#
# print(f"Simulation time using CuPy: {end - start}")


network = gns.create_network_from_csv(Path("."))

start = timer()
network.simulation(use_cuda=False, tol=0.0001)
end = timer()

print(f"Simulation time using NumPy: {end - start}")


shapefile_path = Path("./ie_10km.shp")
if shapefile_path.exists():
    print("1. Using shapefile backend with ie_10km.shp")
    fig1, ax1 = plot_network_pipeline_flow_results(
        network, 
        backend="shapefile", 
        shapefile_path=shapefile_path,
        figsize=(10, 12)
    )
    plt.show()
else:
    print("Shapefile not found, trying alternative backends")

print("2. Trying contextily backend with custom styling")
try:
    fig2, ax2 = plot_network_pipeline_flow_results(
        network, 
        backend="contextily",
        pipeline_color="#FF6B6B",
        line_width_scale=2.0,
        min_line_width=1.0,
        margin_factor=0.3,
        figsize=(12, 10)
    )
    plt.show()
except Exception as e:
    print(f"   Contextily backend failed: {e}")

print("3. Trying cartopy backend with thick lines")
try:
    fig3, ax3 = plot_network_pipeline_flow_results(
        network, 
        backend="cartopy",
        pipeline_color="#4ECDC4",
        line_width_scale=3.0,
        min_line_width=2.0,
        margin_factor=0.15,
        figsize=(12, 10)
    )
    plt.show()
except Exception as e:
    print(f"   Cartopy backend failed: {e}")


print("4. Auto backend with default margins and styling")
fig4, ax4 = plot_network_pipeline_flow_results(
    network, 
    backend="auto",
    shapefile_path=shapefile_path if shapefile_path.exists() else None,
    pipeline_color="#FFD93D",
    alpha=0.8,
    line_width_scale=1.5,
    margin_factor=0.2,  # Default 20% margin
    figsize=(10, 8)
)
plt.show()

