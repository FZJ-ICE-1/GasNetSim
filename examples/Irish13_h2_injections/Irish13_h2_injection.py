# import sys
# import os
# sys.path.append(os.path.dirname(__file__))

from pathlib import Path

import GasNetSim as gns

network = gns.create_network_from_folder(Path("."))
network.simulation()
