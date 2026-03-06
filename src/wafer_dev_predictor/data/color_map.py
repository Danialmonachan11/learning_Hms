# -----------------------------------------------------------------------------#
#                                                                              #
#                         False-color maps                                     #
#                                                                              #
# -----------------------------------------------------------------------------#
#

"""
Generate false-color maps including a histogram of PointCloud data using Plotly. 
"""

__author__ = " []"
# History
# 2025-07-22: v1.0  - initial release
#
# -----------------------------------------------------------------------------#
#                                                                              #
#                Copyright (c) 2025,  Netherlands B.V.                     #
#                         All rights reserved                                  #
#                                                                              #
# -----------------------------------------------------------------------------#

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from color_scale import DARK_RAINBOW
from fastlibrary import Topography


def false_color_map_with_histogram(
    topo: Topography,
    title: str,
    z_min: float | None = None,
    z_max: float | None = None,
    colorscale: list = DARK_RAINBOW,
    show_ticks: bool = False,
) -> go.Figure:
    
    """"
    Generate a false-color map of a `PointCloud` with a histogram of the z-values.

    Arguments:
        point_cloud: Input `PointCloud`
        title: Title of the plot
        z_min: Minimum z-value of the color scale. If the parameter is `None` take the minium z-value of the `point_cloud`
        z_max: Maximum z-value of the color scale. If the parameter is `None` take the maximum z-value of the `point_cloud`
        colorscale: `plotly` colorscale
        show_ticks: Flag to show x/y tick labels
    """

    # linear interpolation of the measurement surface
    z= topo.z_map*1e9

    # plot range settings
    data_range = (np.nanmin(z), np.nanmax(z))
    if not z_min:
        z_min = data_range[0]
    if not z_max:
        z_max = data_range[1]
    
    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.85, 0.15, 0.02],
        specs=[[{"type": "heatmap"}, {"type": "xy"}, {"type": "xy"}]],
        horizontal_spacing=0.0,
    )

    # plot interpolated false-color map
    fig.add_trace(
        go.Heatmap(
            z=z,
            colorscale=colorscale,
            showscale=False,
            zsmooth="best",
            zmin=z_min,
            zmax=z_max,
        ),
        row=1,
        col=1,
    )

    # statistics = f"PV: {point_cloud.pv:.1f} {point_cloud.units.z} | Min: {point_cloud.min:.1f} {point_cloud.units.z} | Max: {point_cloud.max:.1f} {point_cloud.units.z} | RMSD: {point_cloud.rmsd:.1f} {point_cloud.units.z}"

    # if isinstance(point_cloud.meta_data.file_name, list):
    #     file_information = (
    #         f"Source 1: {point_cloud.meta_data.file_name[0]}<br>Source 2: {point_cloud.meta_data.file_name[1]}"
    #     )
    # else:
    #     file_information = f"Source: {point_cloud.meta_data.file_name}"

    # fig.update_xaxes(
    #     title=f"{statistics}<br>{file_information}",
    #     title_font=dict(size=20),
    #     col=1,
    # )

    if not show_ticks:
        fig.update_xaxes(showticklabels=False, col=1)
        fig.update_yaxes(showticklabels=False, col=1)

    # generate histogram data
    z_values = z.flatten()
    z_values = z_values[~np.isnan(z_values)]
    NUMBER_OF_BINS = 200
    counts, edges = np.histogram(z_values, bins=np.arange(z_min, z_max, (z_max - z_min) / NUMBER_OF_BINS))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # add histogram
    fig.add_trace(
        go.Bar(
            y=centers,
            x=-counts,
            orientation="h",
            marker=dict(color=centers, colorscale=colorscale, cmin=z_min, cmax=z_max, line=dict(width=0)),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(showticklabels=False, col=2)
    fig.update_yaxes(showticklabels=False, range=[z_min, z_max], col=2)

    # add color bar
    centers = np.arange(z_min, z_max, (z_max - z_min) / 400)  # type: ignore

    fig.add_trace(
        go.Bar(
            y=centers,
            x=np.full_like(centers, 1.0),
            orientation="h",
            marker=dict(color=centers, colorscale=colorscale, cmin=z_min, cmax=z_max, line=dict(width=0)),
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.update_xaxes(showticklabels=False, side="right", col=3)
    fig.update_yaxes(
        showticklabels=True,
        side="right",
        title="height (nm)",
        title_font=dict(size=24),
        tickfont=dict(size=24),
        col=3,
    )

    fig.update_layout(
        title=title,
        title_font=dict(size=20),
        font_family="Arial",
        font_color="black",
        autosize=True,
        margin=dict(l=5, r=10, t=60, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        bargap=0,
    )
    fig.update_xaxes(showgrid=True, zeroline=False, row=1, col=1)
    fig.update_yaxes(showgrid=True, zeroline=False, scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(showgrid=False, zeroline=False, row=1, col=2)
    fig.update_yaxes(showgrid=False, zeroline=False, row=1, col=2)

    data_aspect_ratio = (z.shape[0]/z.shape[1])

    # fig.update_layout(height=600, width=min(1400, 500 * data_aspect_ratio + 300))

    return fig

