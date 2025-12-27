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
import warnings

# Optional imports for different backends
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

try:
    import contextily as ctx
    CONTEXTILY_AVAILABLE = True
except ImportError:
    CONTEXTILY_AVAILABLE = False


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


def plot_network_pipeline_flow_results(network, backend="auto", shapefile_path=None, 
                                       basemap_provider=None, crs=4326, 
                                       cartopy_projection=None, figsize=(10, 8), 
                                       pipeline_color="#FABE50", alpha=1.0, 
                                       line_width_scale=1.0, min_line_width=0.5, 
                                       margin_factor=0.2, **kwargs):
    """
    Plot network pipeline flow results with multiple backend options.
    
    Parameters:
    -----------
    network : Network
        The gas network object containing pipelines and flow data
    backend : str, default "auto"
        Plotting backend: "auto", "shapefile", "cartopy", "contextily", or "basic"
    shapefile_path : Path, optional
        Path to shapefile for background (required for "shapefile" backend)
    basemap_provider : str, optional
        Basemap provider for contextily (defaults to best available)
    crs : int, default 4326
        Coordinate reference system EPSG code
    cartopy_projection : cartopy.crs projection, optional
        Cartopy projection to use (defaults to PlateCarree)
    figsize : tuple, default (10, 8)
        Figure size in inches
    pipeline_color : str, default "#FABE50"
        Color for pipeline lines
    alpha : float, default 1.0
        Transparency for pipeline lines
    line_width_scale : float, default 1.0
        Scaling factor for line widths based on flow rates
    min_line_width : float, default 0.5
        Minimum line width for pipelines
    margin_factor : float, default 0.2
        Margin as fraction of coordinate range (0.2 = 20% of range on each side)
    **kwargs
        Additional arguments passed to plotting functions
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Determine the best available backend
    if backend == "auto":
        if shapefile_path and shapefile_path.exists():
            backend = "shapefile"
        elif CONTEXTILY_AVAILABLE:
            backend = "contextily"
        elif CARTOPY_AVAILABLE:
            backend = "cartopy"
        else:
            backend = "basic"
            warnings.warn("No enhanced backends available, using basic plotting")
    
    # Validate backend availability
    if backend == "cartopy" and not CARTOPY_AVAILABLE:
        warnings.warn("Cartopy not available, falling back to basic plotting")
        backend = "basic"
    if backend == "contextily" and not CONTEXTILY_AVAILABLE:
        warnings.warn("Contextily not available, falling back to basic plotting")
        backend = "basic"
    if backend == "shapefile" and (not shapefile_path or not shapefile_path.exists()):
        warnings.warn("Shapefile not found, falling back to available backend")
        backend = "auto"
        return plot_network_pipeline_flow_results(network, backend=backend, **kwargs)
    
    # Prepare pipeline data
    pipeline_flows = [p.flow_rate for p in network.pipelines.values()]
    
    # Assign geometry to pipelines
    for pipeline in network.pipelines.values():
        pipeline = assign_pipeline_geometry(pipeline)
    
    # Filter pipelines with valid geometry
    valid_pipelines = [(idx, pipe) for idx, pipe in network.pipelines.items() 
                      if hasattr(pipe, 'geometry') and isinstance(pipe.geometry, LineString)]
    
    if not valid_pipelines:
        raise ValueError("No pipelines have valid geometry (longitude/latitude coordinates)")
    
    # Prepare line data
    lines = []
    colors = []
    linewidths = []
    
    for idx, pipe in valid_pipelines:
        x, y = pipe.geometry.xy
        lines.append(np.column_stack((x, y)))
        colors.append((*matplotlib.colors.to_rgb(pipeline_color), alpha))
        # Scale linewidth based on flow rate (with configurable scaling and minimum)
        flow_rate = pipe.flow_rate if pipe.flow_rate else 0
        linewidth = max(abs(flow_rate) * line_width_scale / 10, min_line_width)
        linewidths.append(linewidth)
    
    # Create figure based on backend
    if backend == "cartopy":
        projection = cartopy_projection or ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': projection})
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, alpha=0.7)
        ax.add_feature(cfeature.BORDERS, alpha=0.5)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        # Set extent based on pipeline coordinates with configurable margin
        all_coords = np.concatenate([line for line in lines])
        if len(all_coords) > 0:
            x_range = all_coords[:, 0].max() - all_coords[:, 0].min()
            y_range = all_coords[:, 1].max() - all_coords[:, 1].min()
            x_margin = margin_factor * x_range
            y_margin = margin_factor * y_range
            ax.set_extent([
                all_coords[:, 0].min() - x_margin,
                all_coords[:, 0].max() + x_margin,
                all_coords[:, 1].min() - y_margin,
                all_coords[:, 1].max() + y_margin
            ], crs=ccrs.PlateCarree())
        
        # Add gridlines
        ax.gridlines(draw_labels=True, alpha=0.3)
        
    elif backend == "shapefile":
        fig, ax = plt.subplots(figsize=figsize)
        
        # Load and plot shapefile
        shape_gdf = read_and_convert_shapefile(shapefile_path, crs=crs)
        shape_gdf.plot(ax=ax, color="gray", edgecolor="gray", alpha=0.1)
        
    elif backend == "contextily":
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a GeoDataFrame from pipeline coordinates for extent calculation
        if lines:
            all_coords = np.concatenate(lines)
            # Convert to Web Mercator for contextily
            gdf_points = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(all_coords[:, 0], all_coords[:, 1]),
                crs=f"EPSG:{crs}"
            )
            gdf_points = gdf_points.to_crs(epsg=3857)  # Web Mercator
            
            # Set axis limits with configurable margin
            bounds = gdf_points.total_bounds
            x_range = bounds[2] - bounds[0]  # max_x - min_x
            y_range = bounds[3] - bounds[1]  # max_y - min_y
            x_margin = margin_factor * x_range
            y_margin = margin_factor * y_range
            ax.set_xlim(bounds[0] - x_margin, bounds[2] + x_margin)
            ax.set_ylim(bounds[1] - y_margin, bounds[3] + y_margin)
            
            # Add basemap with multiple provider fallbacks
            basemap_providers = []
            if basemap_provider:
                basemap_providers.append(basemap_provider)
            
            # Add default reliable providers (only use available ones)
            reliable_providers = []
            
            # Try CartoDB.Positron (most reliable)
            try:
                reliable_providers.append(ctx.providers.CartoDB.Positron)
            except AttributeError:
                pass
            
            # Try OpenStreetMap.Mapnik
            try:
                reliable_providers.append(ctx.providers.OpenStreetMap.Mapnik)
            except AttributeError:
                pass
            
            # Add direct URLs as fallbacks
            reliable_providers.extend([
                "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",  # Direct OSM URL
                "https://tile.openstreetmap.org/{z}/{x}/{y}.png",   # Alternative OSM URL
                "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png"  # Another OSM server
            ])
            
            basemap_providers.extend(reliable_providers)
            
            basemap_loaded = False
            for provider in basemap_providers:
                try:
                    ctx.add_basemap(ax, crs=gdf_points.crs, source=provider, alpha=0.7)
                    basemap_loaded = True
                    print(f"Successfully loaded basemap using: {provider}")
                    break
                except Exception as e:
                    continue  # Try next provider
            
            if not basemap_loaded:
                warnings.warn("Failed to load any basemap providers. Using basic plot without background.")
                # Don't change backend to basic, just continue without background
                ax.set_facecolor('lightblue')  # Use a light background color
        else:
            backend = "basic"
    
    if backend == "basic":
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('lightgray')
    
    # Add pipeline lines
    if lines:
        # For contextily, convert coordinates to Web Mercator
        if backend == "contextily" and CONTEXTILY_AVAILABLE:
            converted_lines = []
            for line in lines:
                gdf_line = gpd.GeoDataFrame(
                    geometry=[LineString(line)], crs=f"EPSG:{crs}"
                ).to_crs(epsg=3857)
                converted_coords = np.array(gdf_line.geometry.iloc[0].coords)
                converted_lines.append(converted_coords)
            lines = converted_lines
        
        # Create LineCollection
        lc = LineCollection(lines, colors=colors, linewidths=linewidths, **kwargs)
        ax.add_collection(lc)
    
    # Style the plot
    if backend not in ["cartopy", "contextily"]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    
    # Add title with backend info
    title = f"Gas Network Pipeline Flow Results ({backend.title()} Backend)"
    ax.set_title(title, fontsize=12, pad=20)
    
    # Add colorbar or legend if flow rates vary significantly
    if len(set(pipeline_flows)) > 1:
        # Create a simple legend showing flow rate ranges
        from matplotlib.lines import Line2D
        flow_min, flow_max = min(pipeline_flows), max(pipeline_flows)
        legend_elements = [
            Line2D([0], [0], color=pipeline_color, lw=max(flow_min/10, 0.5), 
                  label=f'Min flow: {flow_min:.1f} sm³/s'),
            Line2D([0], [0], color=pipeline_color, lw=max(flow_max/10, 0.5), 
                  label=f'Max flow: {flow_max:.1f} sm³/s')
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.8)
    
    plt.tight_layout()
    
    return fig, ax
