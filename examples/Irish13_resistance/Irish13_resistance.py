#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2022.
#     Developed by Yifei Lu
#     Last change on 6/13/22, 4:47 PM
#     Last change by yifei
#    *****************************************************************************
from pathlib import Path

import GasNetSim as gns

network = gns.create_network_from_folder(Path('.'))
network.simulation()
