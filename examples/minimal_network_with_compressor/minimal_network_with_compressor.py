#!/usr/bin/env python
# -*- coding: utf-8 -*-
#******************************************************************************
#  Copyright (c) 2025.
#  Developed by Yifei Lu
#  Compressor validation example
#*****************************************************************************

"""
Compressor Network Validation Example

This example demonstrates the simulation of a gas network with a compressor station.
The network consists of 4 nodes, 3 pipelines, and 1 compressor connecting nodes 2 and 3.

Network topology:
  Node 1 (reference) → Pipeline 1 → Node 2 → Compressor → Node 3 → Pipeline 2 → Node 4
  Node 1 → Pipeline 3 → Node 4
"""

from pathlib import Path
from timeit import default_timer as timer
from scipy.constants import bar

import GasNetSim as gns

network = gns.create_network_from_folder(Path("."))

print(f"\nThis network has:")
print(f"  • {len(network.nodes)} nodes")
print(f"  • {len(network.pipelines)} pipelines")
print(f"  • {len(network.compressors)} compressors")

network.simulation(use_cuda=False, tol=1e-4)

# Print results
print(f"\nNode pressures:")
for idx, node in sorted(network.nodes.items()):
    print(f"  Node {idx}: {node.pressure / bar:6.2f} bar")

print(f"\nPipeline flow rates:")
for idx, pipeline in sorted(network.pipelines.items()):
    flow = pipeline.calc_flow_rate()
    direction = f"{pipeline.inlet_index} → {pipeline.outlet_index}"
    print(f"  Pipeline {idx}: {flow:.2f} sm^3/s")

print(f"\nCompressor status:")
for idx, compressor in sorted(network.compressors.items()):
    flow = compressor.flow_rate or 0
    power = compressor.power_consumption()
    ratio = compressor.compression_ratio
    inlet_p = compressor.inlet.pressure / bar
    outlet_p = compressor.outlet.pressure / bar
    actual_ratio = outlet_p / inlet_p
    direction = f"{compressor.inlet_index} → {compressor.outlet_index}"

    print(f"  Compressor {idx}: {flow:8.2f} sm^3/sm, {power:7.3f} MW, {ratio:.2f} (target), {actual_ratio:.2f} (actual)")
