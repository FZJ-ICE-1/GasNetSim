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

import GasNetSim as gns
from GasNetSim.components.utils.plot_functions import plot_network_pipeline_flow_results

network = gns.create_network_from_csv(Path("."))

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

plot_network_pipeline_flow_results(network, shapefile_path=Path("./ie_10km.shp"))
