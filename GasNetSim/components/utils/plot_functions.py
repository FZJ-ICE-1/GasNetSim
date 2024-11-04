#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2024.
#     Developed by Yifei Lu
#     Last change on 8/22/24, 9:39 AM
#     Last change by yifei
#    *****************************************************************************
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
from shapely import wkt
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import distance
import pickle
from matplotlib.collections import LineCollection
from shapely.geometry import LineString
import matplotlib
from matplotlib.collections import PatchCollection


def read_and_convert_shapefile(shapefile_path: Path, crs=4326) -> gpd.GeoDataFrame:
    _shape_gdf = gpd.read_file(shapefile_path)
    print(_shape_gdf.crs)
    if _shape_gdf.crs is None:
        _shape_gdf_in_desired_crs = _shape_gdf.set_crs(
            epsg=crs, inplace=True
        )  # set crs to desired crs
    else:
        _shape_gdf_in_desired_crs = _shape_gdf.to_crs(epsg=crs)
    return _shape_gdf_in_desired_crs


def assign_pipeline_geometry(pipeline):
    if (
        pipeline.inlet.longitude is not None
        and pipeline.inlet.latitude is not None
        and pipeline.outlet.longitude is not None
        and pipeline.outlet.latitude is not None
    ):
        pipeline.geometry = LineString(
            [
                (pipeline.inlet.longitude, pipeline.inlet.latitude),
                (pipeline.outlet.longitude, pipeline.outlet.latitude),
            ]
        )
    return pipeline


def plot_network_pipeline_flow_results(network, shapefile_path: Path, crs=4326):
    _shape_gdf_in_desired_crs = read_and_convert_shapefile(shapefile_path, crs=crs)

    fig, ax = plt.subplots(figsize=(6, 8))

    _shape_gdf_in_desired_crs.plot(ax=ax, color="gray", edgecolor="gray", alpha=0.1)

    pipeline_flows = [p.flow_rate for p in network.pipelines.values()]

    for pipeline in network.pipelines.values():
        pipeline = assign_pipeline_geometry(pipeline)

    lines = []
    colors = []  # Color for each line
    linewidths = [x / 10 for x in pipeline_flows]  # Linewidth for each line

    for _idx, _pipe in network.pipelines.items():
        # Assuming the LineString geometries are in 'geometry' column
        if isinstance(_pipe.geometry, LineString):
            x, y = _pipe.geometry.xy
            lines.append(np.column_stack((x, y)))
            color = "#FABE50"  # Example fixed color, can be made dynamic

            alpha = 1

            colors.append(
                (*matplotlib.colors.to_rgb(color), alpha)
            )  # Convert color to RGBA

    # Create a LineCollection
    lc = LineCollection(lines, colors=colors, linewidths=linewidths)
    # Add the collection to the plot
    ax.add_collection(lc)

    # ax.set_xlim(4, 16)
    # ax.set_ylim(46, 56)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    plt.tight_layout()
    plt.show()
    return None
