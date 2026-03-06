"""
Dash web app for wafer image labeling and anomaly analysis.

Tab 1 — Labeling:
    Displays a table of wafer measurements, allows clicking rows to view
    false-color topography images, and provides buttons to label each
    measurement as normal or anomalous.

Tab 2 — Anomaly Analysis:
    Select a measurement and run one of three methods to estimate the process
    baseline and extract the anomaly map:
      - Robust Poly (recommended): IRLS fit to the full image — no region
        selection needed, automatically ignores defect pixels.
      - Polynomial: manual region selection via box-draw on the original image.
      - Gaussian: global low-pass filter via fastlibrary.

Tab 3 — Diff Map Browser:
    Browse pre-computed step-to-step difference maps from generated_files.parquet
    (loaded via zygo_reader). Each click shows 3 stacked images:
      - Raw diff map (as stored)
      - IRLS process baseline (smooth component)
      - Anomaly signal (diff − baseline)
    Label buttons save Normal/Anomaly/Skip to labeled_diff_maps.parquet.

Run with: py -3.11 app.py  (from src/wafer_dev_predictor/ directory)
"""

import os
import re
import sys
from pathlib import Path

# Allow bare imports of color_map / color_scale and analysis subpackage
sys.path.insert(0, str(Path(__file__).parent / "data"))

try:
    import zygo_reader
    ZYGO_READER_AVAILABLE = True
except ImportError:
    ZYGO_READER_AVAILABLE = False

import numpy as np
import dash
import polars as pl
import fastlibrary as fl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dash_table, dcc, html, Input, Output, State, callback, no_update

from color_map import false_color_map_with_histogram
from color_scale import DARK_RAINBOW
from analysis.prediction import (
    predict_normal_polynomial,
    predict_normal_robust_polynomial,
    predict_normal_gaussian,
    inpaint_region,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATABASE_PATH = (
    R"T:\asm\EXE_ZdMB\10_EXE_MB_Flatness_Database"
    R"\Flatness Reports\MiQaT_Specification_Data.parquet"
)
LABELED_DB_PATH = Path("data/labeled_measurements.parquet")

# Diff map database (generated_files.parquet — pre-computed step-to-step diffs)
DIFF_DB_PATH = (
    R"T:\asm\E X E_MB\10-EXE_MB_Flatness_Database"
    R"\.db\Generated Files\generated_files.parquet"
)
DIFF_DB_ROOT = R"T:\asm\E X E_MB\10-EXE_MB_Flatness_Database"
LABELED_DIFF_PATH = Path("data/labeled_diff_maps.parquet")

# ---------------------------------------------------------------------------
# Data loading (mirrors Database_ZA_ZE.py logic)
# ---------------------------------------------------------------------------


def load_measurements() -> pl.DataFrame:
    """Load and filter wafer measurements from the parquet database."""
    measurement_information = pl.read_parquet(DATABASE_PATH)

    measurements = (
        measurement_information.filter(
            (pl.col("Side").str.contains("ZA|ZE"))
            & (pl.col("Identifier") == "GridPV")
            & (
                pl.col("ProcessStep").str.contains(
                    "after IBF|after coating|after HI bonding|after Z polishing|before IBF"
                )
            )
        )
        .filter(
            ~(
                pl.col("ProcessStep").str.contains("after IBF run").fill_null(False)
                & ~pl.col("Tags").str.contains("after IBF").fill_null(False)
            )
        )
        .select(["Serial", "ProcessStep", "Tags", "Side", "MeasurementDate", "SourcePath"])
        .unique()
    ).sort(by="MeasurementDate")

    measurements = measurements.group_by(["Serial", "ProcessStep", "Side"]).last()
    return measurements


def load_or_init_labels(measurements: pl.DataFrame) -> pl.DataFrame:
    """Load existing labels or create a fresh label column."""
    if LABELED_DB_PATH.exists():
        labeled = pl.read_parquet(LABELED_DB_PATH)
        # Merge labels into current measurements (left join keeps all rows)
        measurements = measurements.join(
            labeled.select(["Serial", "ProcessStep", "Side", "Label"]),
            on=["Serial", "ProcessStep", "Side"],
            how="left",
        )
    else:
        measurements = measurements.with_columns(pl.lit(None).cast(pl.Utf8).alias("Label"))
    return measurements


# ---------------------------------------------------------------------------
# Diff map helpers (Tab 3)
# ---------------------------------------------------------------------------

_SIDE_RE = re.compile(r"\b(Z[EA])\b")
_DATE_RE = re.compile(r"(\d{6})")


def _extract_side_from_path(fpath: str) -> str:
    m = _SIDE_RE.search(fpath.replace("\\", "/").split("/")[-1])
    return m.group(1) if m else "?"


def _extract_date_from_path(fpath: str) -> str:
    parts = fpath.replace("\\", "/").split("/")
    # The folder under "Difference plots/" typically starts with YYMMDD
    for part in reversed(parts):
        m = _DATE_RE.match(part.strip())
        if m:
            raw = m.group(1)
            return f"20{raw[:2]}-{raw[2:4]}-{raw[4:]}"
    return ""


def load_diff_maps() -> pl.DataFrame:
    """
    Load and filter the generated_files.parquet to the ZA/ZE HI-bonding diff maps.

    Returns a DataFrame with columns:
        filePath, fullPath, Side, DateStr, Label (null initially)
    """
    gen = pl.read_parquet(DIFF_DB_PATH)
    filtered = gen.filter(
        (pl.col("description") == "Side difference map - Zygo file")
        & pl.col("filePath").str.contains(r".*Z.*HI.*coating")
    )
    paths = filtered["filePath"].to_list()
    sides = [_extract_side_from_path(p) for p in paths]
    dates = [_extract_date_from_path(p) for p in paths]
    full_paths = [os.path.join(DIFF_DB_ROOT, p) for p in paths]

    return pl.DataFrame(
        {
            "filePath": paths,
            "fullPath": full_paths,
            "Side": sides,
            "DateStr": dates,
            "Label": [None] * len(paths),
        }
    )


def load_or_init_diff_labels(diff_df: pl.DataFrame) -> pl.DataFrame:
    """Merge saved labels into the diff map DataFrame."""
    if LABELED_DIFF_PATH.exists():
        saved = pl.read_parquet(LABELED_DIFF_PATH)
        diff_df = diff_df.join(
            saved.select(["filePath", "Label"]),
            on="filePath",
            how="left",
            suffix="_saved",
        )
        # Coalesce: prefer the saved label
        if "Label_saved" in diff_df.columns:
            diff_df = diff_df.with_columns(
                pl.coalesce([pl.col("Label_saved"), pl.col("Label")]).alias("Label")
            ).drop("Label_saved")
    return diff_df


def _save_diff_labels(data: list[dict]) -> None:
    """Persist diff map labels to parquet."""
    import pandas as pd

    LABELED_DIFF_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    labeled = df[df["Label"].notna() & (df["Label"] != "")]
    if not labeled.empty:
        labeled[["filePath", "Label"]].to_parquet(LABELED_DIFF_PATH, index=False)


# ---------------------------------------------------------------------------
# Initialise data
# ---------------------------------------------------------------------------
measurements_df = load_measurements()
measurements_df = load_or_init_labels(measurements_df)

# Diff map database (Tab 3) — may not be available on all machines
diff_maps_df: pl.DataFrame = pl.DataFrame(
    {"filePath": [], "fullPath": [], "Side": [], "DateStr": [], "Label": []}
)
diff_maps_available = False
try:
    diff_maps_df = load_diff_maps()
    diff_maps_df = load_or_init_diff_labels(diff_maps_df)
    diff_maps_available = True
    print(f"[Tab 3] Loaded {len(diff_maps_df)} diff maps.")
except Exception as _diff_err:
    print(f"[Tab 3] Diff map DB not available: {_diff_err}")

# Convert to list-of-dicts for the Dash DataTable
table_data = measurements_df.to_pandas().to_dict("records")
table_columns = [
    {"name": "Serial", "id": "Serial"},
    {"name": "ProcessStep", "id": "ProcessStep"},
    {"name": "Side", "id": "Side"},
    {"name": "Tags", "id": "Tags"},
    {"name": "MeasurementDate", "id": "MeasurementDate"},
    {"name": "Label", "id": "Label"},
]

# Tab 3 — Diff map table
diff_table_data = diff_maps_df.to_pandas().to_dict("records")
diff_table_columns = [
    {"name": "Side", "id": "Side"},
    {"name": "Date", "id": "DateStr"},
    {"name": "File", "id": "filePath"},
    {"name": "Label", "id": "Label"},
]
_diff_side_options = [{"label": "All", "value": "All"}, {"label": "ZA", "value": "ZA"}, {"label": "ZE", "value": "ZE"}]

# ---------------------------------------------------------------------------
# Tab 2 — Diff map table data (separate copy from Tab 3)
# ---------------------------------------------------------------------------
t2_diff_table_data = [dict(r) for r in diff_table_data]
t2_diff_table_columns = [
    {"name": "Side", "id": "Side"},
    {"name": "Date", "id": "DateStr"},
    {"name": "File", "id": "filePath"},
]


# ---------------------------------------------------------------------------
# Tab 2 — Figure helper
# ---------------------------------------------------------------------------


def _make_heatmap_fig(
    z: np.ndarray,
    title: str,
    colorscale,
    z_min: float | None = None,
    z_max: float | None = None,
) -> go.Figure:
    """
    Build a heatmap + histogram + colorbar figure directly from a z array (nm).

    Mirrors the layout of false_color_map_with_histogram but accepts raw arrays,
    enabling reuse for predicted and anomaly figures without a Topography object.
    No scaleanchor so the heatmap fills the fixed graph height (good for wide images).
    """
    z = np.asarray(z, dtype=float)
    valid = z[~np.isnan(z)]

    if z_min is None:
        z_min = float(np.nanmin(z)) if valid.size else -1.0
    if z_max is None:
        z_max = float(np.nanmax(z)) if valid.size else 1.0
    # Guard against flat arrays
    if abs(z_max - z_min) < 1e-10:
        z_min -= 1.0
        z_max += 1.0

    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.85, 0.13, 0.02],
        horizontal_spacing=0.0,
        specs=[[{"type": "heatmap"}, {"type": "xy"}, {"type": "xy"}]],
    )

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

    n_bins = 200
    counts, edges = np.histogram(valid, bins=np.linspace(z_min, z_max, n_bins + 1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig.add_trace(
        go.Bar(
            y=centers,
            x=-counts,
            orientation="h",
            marker=dict(
                color=centers,
                colorscale=colorscale,
                cmin=z_min,
                cmax=z_max,
                line=dict(width=0),
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(showticklabels=False, col=2)
    fig.update_yaxes(showticklabels=False, range=[z_min, z_max], col=2)

    centers_cb = np.linspace(z_min, z_max, 400)
    fig.add_trace(
        go.Bar(
            y=centers_cb,
            x=np.ones(400),
            orientation="h",
            marker=dict(
                color=centers_cb,
                colorscale=colorscale,
                cmin=z_min,
                cmax=z_max,
                line=dict(width=0),
            ),
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.update_xaxes(showticklabels=False, col=3)
    fig.update_yaxes(
        showticklabels=True,
        side="right",
        title="height (nm)",
        title_font=dict(size=14),
        tickfont=dict(size=12),
        col=3,
    )

    fig.update_xaxes(showticklabels=False, showgrid=True, zeroline=False, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=True, zeroline=False, col=1)

    fig.update_layout(
        title=title,
        title_font=dict(size=16),
        font_family="Arial",
        autosize=True,
        margin=dict(l=5, r=10, t=50, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        bargap=0,
    )
    return fig


# ---------------------------------------------------------------------------
# Dash App
# ---------------------------------------------------------------------------
app = dash.Dash(__name__, title="Wafer Dev Predictor")

BTN_STYLE = {
    "margin": "5px",
    "padding": "10px 20px",
    "color": "white",
    "border": "none",
    "cursor": "pointer",
    "fontSize": "15px",
    "borderRadius": "4px",
}

CTRL_LABEL_STYLE = {"fontSize": "12px", "fontWeight": "bold", "marginBottom": "3px"}

app.layout = html.Div(
    [
        html.H2(
            "Wafer Dev Predictor",
            style={"margin": "0 0 12px 0", "fontSize": "20px"},
        ),
        dcc.Tabs(
            [
                # ================================================================
                # TAB 1 — Labeling
                # ================================================================
                dcc.Tab(
                    label="Labeling",
                    children=[
                        html.Div(
                            [
                                # Left: scrollable table
                                html.Div(
                                    dash_table.DataTable(
                                        id="measurements-table",
                                        columns=table_columns,
                                        data=table_data,
                                        row_selectable="single",
                                        selected_rows=[],
                                        filter_action="native",
                                        sort_action="native",
                                        page_action="none",
                                        fixed_rows={"headers": True},
                                        style_table={
                                            "height": "calc(100vh - 130px)",
                                            "overflowY": "auto",
                                            "overflowX": "auto",
                                        },
                                        style_cell={
                                            "textAlign": "left",
                                            "padding": "6px 10px",
                                            "fontSize": "13px",
                                            "whiteSpace": "normal",
                                            "minWidth": "80px",
                                        },
                                        style_header={
                                            "fontWeight": "bold",
                                            "backgroundColor": "#e8e8e8",
                                            "position": "sticky",
                                            "top": 0,
                                        },
                                        style_data_conditional=[
                                            {
                                                "if": {"filter_query": '{Label} = "Normal_ZA"'},
                                                "backgroundColor": "#d4edda",
                                            },
                                            {
                                                "if": {"filter_query": '{Label} = "Normal_ZE"'},
                                                "backgroundColor": "#d4edda",
                                            },
                                            {
                                                "if": {"filter_query": '{Label} = "Anomaly_ZA"'},
                                                "backgroundColor": "#f8d7da",
                                            },
                                            {
                                                "if": {"filter_query": '{Label} = "Anomaly_ZE"'},
                                                "backgroundColor": "#f8d7da",
                                            },
                                        ],
                                    ),
                                    style={
                                        "width": "38%",
                                        "paddingRight": "12px",
                                        "boxSizing": "border-box",
                                    },
                                ),
                                # Right: image + buttons
                                html.Div(
                                    [
                                        dcc.Loading(
                                            id="loading-image",
                                            type="circle",
                                            children=dcc.Graph(
                                                id="image-viewer",
                                                style={"height": "calc(100vh - 220px)"},
                                                config={"responsive": True},
                                            ),
                                        ),
                                        html.Div(
                                            [
                                                html.Button(
                                                    "Normal_ZA",
                                                    id="btn-normal-za",
                                                    n_clicks=0,
                                                    style={**BTN_STYLE, "backgroundColor": "#28a745"},
                                                ),
                                                html.Button(
                                                    "Normal_ZE",
                                                    id="btn-normal-ze",
                                                    n_clicks=0,
                                                    style={**BTN_STYLE, "backgroundColor": "#28a745"},
                                                ),
                                                html.Button(
                                                    "Anomaly_ZA",
                                                    id="btn-anomaly-za",
                                                    n_clicks=0,
                                                    style={**BTN_STYLE, "backgroundColor": "#dc3545"},
                                                ),
                                                html.Button(
                                                    "Anomaly_ZE",
                                                    id="btn-anomaly-ze",
                                                    n_clicks=0,
                                                    style={**BTN_STYLE, "backgroundColor": "#dc3545"},
                                                ),
                                            ],
                                            style={"textAlign": "center", "paddingTop": "8px"},
                                        ),
                                        html.Div(
                                            id="label-status",
                                            style={
                                                "textAlign": "center",
                                                "padding": "6px",
                                                "fontSize": "14px",
                                                "fontWeight": "bold",
                                            },
                                        ),
                                    ],
                                    style={"width": "62%", "boxSizing": "border-box"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexDirection": "row",
                                "alignItems": "flex-start",
                                "paddingTop": "10px",
                            },
                        ),
                    ],
                ),
                # ================================================================
                # TAB 2 — Anomaly Removal
                # ================================================================
                dcc.Tab(
                    label="Anomaly Removal",
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "row",
                                "alignItems": "flex-start",
                                "paddingTop": "10px",
                                "height": "calc(100vh - 130px)",
                            },
                            children=[
                                # ---- Left: filters + table ----
                                html.Div(
                                    style={
                                        "width": "38%",
                                        "paddingRight": "12px",
                                        "boxSizing": "border-box",
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "height": "100%",
                                    },
                                    children=[
                                        html.Div(
                                            style={
                                                "display": "flex",
                                                "gap": "12px",
                                                "alignItems": "flex-end",
                                                "marginBottom": "8px",
                                            },
                                            children=[
                                                html.Div(
                                                    [
                                                        html.Div("Side", style=CTRL_LABEL_STYLE),
                                                        dcc.Dropdown(
                                                            id="t2-side-filter",
                                                            options=_diff_side_options,
                                                            value="All",
                                                            clearable=False,
                                                            style={"minWidth": "90px"},
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    id="t2-count",
                                                    style={"fontSize": "12px", "color": "#666", "paddingBottom": "4px"},
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            dash_table.DataTable(
                                                id="t2-diff-table",
                                                columns=t2_diff_table_columns,
                                                data=t2_diff_table_data,
                                                row_selectable="single",
                                                selected_rows=[],
                                                filter_action="native",
                                                sort_action="native",
                                                page_action="none",
                                                fixed_rows={"headers": True},
                                                style_table={
                                                    "height": "calc(100vh - 220px)",
                                                    "overflowY": "auto",
                                                    "overflowX": "auto",
                                                },
                                                style_cell={
                                                    "textAlign": "left",
                                                    "padding": "5px 8px",
                                                    "fontSize": "12px",
                                                    "whiteSpace": "nowrap",
                                                    "overflow": "hidden",
                                                    "textOverflow": "ellipsis",
                                                    "maxWidth": "180px",
                                                },
                                                style_header={
                                                    "fontWeight": "bold",
                                                    "backgroundColor": "#e8e8e8",
                                                    "position": "sticky",
                                                    "top": 0,
                                                },
                                                tooltip_data=[
                                                    {"filePath": {"value": str(row.get("filePath", "")), "type": "markdown"}}
                                                    for row in t2_diff_table_data
                                                ],
                                                tooltip_duration=None,
                                            ),
                                            style={"flex": "1"},
                                        ),
                                    ],
                                ),
                                # ---- Right: 2 images + controls ----
                                html.Div(
                                    style={
                                        "width": "62%",
                                        "boxSizing": "border-box",
                                        "overflowY": "auto",
                                        "height": "100%",
                                    },
                                    children=[
                                        html.Div(
                                            id="t2-file-title",
                                            style={"fontSize": "13px", "fontWeight": "bold", "marginBottom": "6px", "color": "#333"},
                                        ),
                                        # Controls row
                                        html.Div(
                                            style={"display": "flex", "gap": "16px", "alignItems": "flex-end", "marginBottom": "8px"},
                                            children=[
                                                html.Div(
                                                    [
                                                        html.Div("Poly degree", style=CTRL_LABEL_STYLE),
                                                        dcc.Dropdown(
                                                            id="t2-poly-degree",
                                                            options=[
                                                                {"label": str(d), "value": d}
                                                                for d in [1, 2, 3, 4, 5, 6]
                                                            ],
                                                            value=5,
                                                            clearable=False,
                                                            style={"width": "80px"},
                                                        ),
                                                    ]
                                                ),
                                            ],
                                        ),
                                        # Image 1: Original Diff Map
                                        html.Div(
                                            [
                                                html.Span(
                                                    "Original Diff Map",
                                                    style={"fontWeight": "bold", "fontSize": "13px"},
                                                ),
                                                html.Span(
                                                    "  — draw a box to select the anomaly region",
                                                    style={"fontSize": "12px", "color": "#666"},
                                                ),
                                            ],
                                            style={"marginBottom": "2px"},
                                        ),
                                        dcc.Loading(
                                            type="circle",
                                            children=dcc.Graph(
                                                id="t2-original-graph",
                                                style={"height": "280px"},
                                                config={
                                                    "responsive": True,
                                                    "modeBarButtonsToAdd": ["select2d"],
                                                    "displayModeBar": True,
                                                },
                                            ),
                                        ),
                                        # Selection info
                                        html.Div(
                                            id="t2-selection-info",
                                            style={
                                                "padding": "6px 4px",
                                                "fontSize": "13px",
                                                "color": "#555",
                                                "fontStyle": "italic",
                                            },
                                        ),
                                        # Image 2: Cleaned (anomaly removed)
                                        html.Div(
                                            "Cleaned (Anomaly Removed)",
                                            style={
                                                "fontWeight": "bold",
                                                "fontSize": "13px",
                                                "marginTop": "8px",
                                                "marginBottom": "2px",
                                            },
                                        ),
                                        dcc.Loading(
                                            type="circle",
                                            children=dcc.Graph(
                                                id="t2-cleaned-graph",
                                                style={"height": "280px"},
                                                config={"responsive": True},
                                            ),
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                # ================================================================
                # TAB 3 — Diff Map Browser
                # ================================================================
                dcc.Tab(
                    label="Diff Map Browser",
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "row",
                                "alignItems": "flex-start",
                                "paddingTop": "10px",
                                "height": "calc(100vh - 130px)",
                            },
                            children=[
                                # ---- Left: filters + table ----
                                html.Div(
                                    style={
                                        "width": "38%",
                                        "paddingRight": "12px",
                                        "boxSizing": "border-box",
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "height": "100%",
                                    },
                                    children=[
                                        # Filter row
                                        html.Div(
                                            style={
                                                "display": "flex",
                                                "gap": "12px",
                                                "alignItems": "flex-end",
                                                "marginBottom": "8px",
                                            },
                                            children=[
                                                html.Div(
                                                    [
                                                        html.Div("Side", style=CTRL_LABEL_STYLE),
                                                        dcc.Dropdown(
                                                            id="t3-side-filter",
                                                            options=_diff_side_options,
                                                            value="All",
                                                            clearable=False,
                                                            style={"minWidth": "90px"},
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    id="t3-count",
                                                    style={"fontSize": "12px", "color": "#666", "paddingBottom": "4px"},
                                                ),
                                            ],
                                        ),
                                        # Table
                                        html.Div(
                                            dash_table.DataTable(
                                                id="t3-diff-table",
                                                columns=diff_table_columns,
                                                data=diff_table_data,
                                                row_selectable="single",
                                                selected_rows=[],
                                                filter_action="native",
                                                sort_action="native",
                                                page_action="none",
                                                fixed_rows={"headers": True},
                                                style_table={
                                                    "height": "calc(100vh - 220px)",
                                                    "overflowY": "auto",
                                                    "overflowX": "auto",
                                                },
                                                style_cell={
                                                    "textAlign": "left",
                                                    "padding": "5px 8px",
                                                    "fontSize": "12px",
                                                    "whiteSpace": "nowrap",
                                                    "overflow": "hidden",
                                                    "textOverflow": "ellipsis",
                                                    "maxWidth": "180px",
                                                },
                                                style_header={
                                                    "fontWeight": "bold",
                                                    "backgroundColor": "#e8e8e8",
                                                    "position": "sticky",
                                                    "top": 0,
                                                },
                                                style_data_conditional=[
                                                    {
                                                        "if": {"filter_query": '{Label} = "Normal"'},
                                                        "backgroundColor": "#d4edda",
                                                    },
                                                    {
                                                        "if": {"filter_query": '{Label} = "Anomaly"'},
                                                        "backgroundColor": "#f8d7da",
                                                    },
                                                    {
                                                        "if": {"filter_query": '{Label} = "Skip"'},
                                                        "backgroundColor": "#fff3cd",
                                                    },
                                                ],
                                                tooltip_data=[
                                                    {"filePath": {"value": str(row.get("filePath", "")), "type": "markdown"}}
                                                    for row in diff_table_data
                                                ],
                                                tooltip_duration=None,
                                            ),
                                            style={"flex": "1"},
                                        ),
                                    ],
                                ),
                                # ---- Right: 3 images + label buttons ----
                                html.Div(
                                    style={
                                        "width": "62%",
                                        "boxSizing": "border-box",
                                        "overflowY": "auto",
                                        "height": "100%",
                                    },
                                    children=[
                                        html.Div(
                                            id="t3-file-title",
                                            style={"fontSize": "13px", "fontWeight": "bold", "marginBottom": "6px", "color": "#333"},
                                        ),
                                        # Raw diff
                                        html.Div(
                                            "Raw Diff Map",
                                            style={"fontWeight": "bold", "fontSize": "13px", "marginBottom": "2px"},
                                        ),
                                        dcc.Loading(
                                            type="circle",
                                            children=dcc.Graph(
                                                id="t3-raw-graph",
                                                style={"height": "240px"},
                                                config={"responsive": True},
                                            ),
                                        ),
                                        # IRLS baseline
                                        html.Div(
                                            "Process Baseline (IRLS Robust Poly)",
                                            style={"fontWeight": "bold", "fontSize": "13px", "marginTop": "8px", "marginBottom": "2px"},
                                        ),
                                        dcc.Loading(
                                            type="circle",
                                            children=dcc.Graph(
                                                id="t3-baseline-graph",
                                                style={"height": "240px"},
                                                config={"responsive": True},
                                            ),
                                        ),
                                        # Anomaly signal
                                        html.Div(
                                            "Anomaly Signal (diff − baseline)",
                                            style={"fontWeight": "bold", "fontSize": "13px", "marginTop": "8px", "marginBottom": "2px"},
                                        ),
                                        dcc.Loading(
                                            type="circle",
                                            children=dcc.Graph(
                                                id="t3-anomaly-graph",
                                                style={"height": "240px"},
                                                config={"responsive": True},
                                            ),
                                        ),
                                        # Metrics
                                        html.Div(
                                            id="t3-metrics",
                                            style={
                                                "padding": "8px 4px",
                                                "fontSize": "13px",
                                                "fontWeight": "bold",
                                                "color": "#333",
                                            },
                                        ),
                                        # Label buttons
                                        html.Div(
                                            [
                                                html.Button(
                                                    "Normal",
                                                    id="t3-btn-normal",
                                                    n_clicks=0,
                                                    style={**BTN_STYLE, "backgroundColor": "#28a745"},
                                                ),
                                                html.Button(
                                                    "Anomaly",
                                                    id="t3-btn-anomaly",
                                                    n_clicks=0,
                                                    style={**BTN_STYLE, "backgroundColor": "#dc3545"},
                                                ),
                                                html.Button(
                                                    "Skip",
                                                    id="t3-btn-skip",
                                                    n_clicks=0,
                                                    style={**BTN_STYLE, "backgroundColor": "#6c757d"},
                                                ),
                                            ],
                                            style={"paddingTop": "8px"},
                                        ),
                                        html.Div(
                                            id="t3-label-status",
                                            style={
                                                "padding": "6px 4px",
                                                "fontSize": "13px",
                                                "fontWeight": "bold",
                                                "color": "#555",
                                            },
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ]
        ),
        dcc.Store(id="selected-source-path"),
        dcc.Store(id="t2-full-path"),  # stores the full file path of the selected diff map
    ],
    style={
        "padding": "10px",
        "fontFamily": "Arial, sans-serif",
        "height": "100vh",
        "boxSizing": "border-box",
    },
)


# ---------------------------------------------------------------------------
# TAB 1 Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("image-viewer", "figure"),
    Output("selected-source-path", "data"),
    Input("measurements-table", "selected_rows"),
    State("measurements-table", "data"),
    prevent_initial_call=True,
)
def display_image(selected_rows, data):
    """Load and display the wafer image for the selected table row."""
    if not selected_rows:
        return no_update, no_update

    row = data[selected_rows[0]]
    source_path = row["SourcePath"]

    try:
        topo = fl.read_zygo(source_path)
        title = f"{row['Serial']} | {row['Side']} | {row['ProcessStep']}"
        fig = false_color_map_with_histogram(topo, title=title, z_min=-50, z_max=50)
        return fig, source_path
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(
            title=f"Error loading: {e}",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return fig, None


@callback(
    Output("label-status", "children"),
    Output("measurements-table", "data"),
    Input("btn-normal-za", "n_clicks"),
    Input("btn-normal-ze", "n_clicks"),
    Input("btn-anomaly-za", "n_clicks"),
    Input("btn-anomaly-ze", "n_clicks"),
    State("measurements-table", "selected_rows"),
    State("measurements-table", "data"),
    prevent_initial_call=True,
)
def apply_label(n_normal_za, n_normal_ze, n_anomaly_za, n_anomaly_ze, selected_rows, data):
    """Apply a label to the selected row and save to parquet."""
    if not selected_rows:
        return "No row selected — please click a row first.", no_update

    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    label_map = {
        "btn-normal-za": "Normal_ZA",
        "btn-normal-ze": "Normal_ZE",
        "btn-anomaly-za": "Anomaly_ZA",
        "btn-anomaly-ze": "Anomaly_ZE",
    }
    label = label_map.get(button_id)
    if label is None:
        return no_update, no_update

    row_idx = selected_rows[0]
    row = data[row_idx]
    data[row_idx]["Label"] = label

    _save_labels(data)

    status = f"Labeled: {row['Serial']} | {row['Side']} | {row['ProcessStep']} → {label}"
    return status, data


def _save_labels(data: list[dict]) -> None:
    """Persist current labels to the labeled parquet file."""
    import pandas as pd

    LABELED_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    labeled = df[df["Label"].notna() & (df["Label"] != "")]
    if not labeled.empty:
        labeled.to_parquet(LABELED_DB_PATH, index=False)


# ---------------------------------------------------------------------------
# TAB 2 Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("t2-diff-table", "data"),
    Output("t2-count", "children"),
    Input("t2-side-filter", "value"),
)
def filter_t2_table(side_filter):
    """Filter the Tab 2 diff map table by side."""
    if side_filter == "All" or not side_filter:
        data = t2_diff_table_data
    else:
        data = [r for r in t2_diff_table_data if r.get("Side") == side_filter]
    return data, f"{len(data)} files"


@callback(
    Output("t2-original-graph", "figure"),
    Output("t2-full-path", "data"),
    Output("t2-file-title", "children"),
    Input("t2-diff-table", "selected_rows"),
    State("t2-diff-table", "data"),
    prevent_initial_call=True,
)
def load_t2_original(selected_rows, data):
    """Load selected diff map and display as the Original image."""
    empty_fig = go.Figure()
    empty_fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False})

    if not selected_rows:
        return empty_fig, None, ""

    if not ZYGO_READER_AVAILABLE:
        return empty_fig, None, "zygo_reader not installed"

    row = data[selected_rows[0]]
    full_path = row["fullPath"]
    file_name = os.path.basename(full_path)
    title_str = f"{row['Side']}  |  {row['DateStr']}  |  {file_name}"

    try:
        zdat = zygo_reader.DatReader(path_or_file_like=full_path)
        z = zdat.get_topography_nm()

        z_min = float(np.nanmin(z))
        z_max = float(np.nanmax(z))
        fig = _make_heatmap_fig(z, f"Original — {title_str}", DARK_RAINBOW, z_min, z_max)
        fig.update_layout(dragmode="select")

        return fig, full_path, title_str

    except Exception as e:
        err_fig = go.Figure()
        err_fig.update_layout(
            title=f"Error: {e}",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return err_fig, None, title_str


@callback(
    Output("t2-cleaned-graph", "figure"),
    Output("t2-selection-info", "children"),
    Input("t2-original-graph", "selectedData"),
    Input("t2-full-path", "data"),
    State("t2-poly-degree", "value"),
    prevent_initial_call=True,
)
def update_t2_cleaned(selected_data, full_path, degree):
    """
    Update the Cleaned image.

    - If no box is drawn: show a copy of the original (no changes).
    - If a box is drawn: fit polynomial to pixels OUTSIDE the box,
      replace pixels INSIDE the box with the polynomial prediction.
    """
    empty_fig = go.Figure()
    empty_fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False})

    if not full_path:
        return empty_fig, "Select a file from the table."

    if not ZYGO_READER_AVAILABLE:
        return empty_fig, "zygo_reader not installed"

    try:
        zdat = zygo_reader.DatReader(path_or_file_like=full_path)
        z = zdat.get_topography_nm()
        n_rows, n_cols = z.shape

        z_min = float(np.nanmin(z))
        z_max = float(np.nanmax(z))

        title_base = os.path.basename(full_path)

        if selected_data and "range" in selected_data:
            x_range = selected_data["range"].get("x", [])
            y_range = selected_data["range"].get("y", [])

            if len(x_range) >= 2 and len(y_range) >= 2:
                col_start = max(0, int(x_range[0]))
                col_end = min(int(x_range[1]) + 1, n_cols)
                row_start = max(0, int(min(y_range)))
                row_end = min(int(max(y_range)) + 1, n_rows)

                cleaned_z, _ = inpaint_region(
                    z,
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end,
                    degree=int(degree) if degree else 5,
                )

                fig = _make_heatmap_fig(
                    cleaned_z,
                    f"Cleaned — {title_base}",
                    DARK_RAINBOW,
                    z_min,
                    z_max,
                )
                info = (
                    f"Selected region: rows {row_start}-{row_end}, "
                    f"cols {col_start}-{col_end} — "
                    f"replaced with polynomial fit (degree {degree})"
                )
                return fig, info

        # No selection — show original as-is
        fig = _make_heatmap_fig(z, f"Cleaned — {title_base} (no selection)", DARK_RAINBOW, z_min, z_max)
        return fig, "Draw a box on the Original image to select the anomaly region."

    except Exception as e:
        err_fig = go.Figure()
        err_fig.update_layout(
            title=f"Error: {e}",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return err_fig, f"Error: {e}"


# ---------------------------------------------------------------------------
# TAB 3 Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("t3-diff-table", "data"),
    Output("t3-count", "children"),
    Input("t3-side-filter", "value"),
)
def filter_diff_table(side_filter):
    """Filter the diff map table by side."""
    if side_filter == "All" or not side_filter:
        data = diff_table_data
    else:
        data = [r for r in diff_table_data if r.get("Side") == side_filter]
    return data, f"{len(data)} files"


@callback(
    Output("t3-raw-graph", "figure"),
    Output("t3-baseline-graph", "figure"),
    Output("t3-anomaly-graph", "figure"),
    Output("t3-metrics", "children"),
    Output("t3-file-title", "children"),
    Input("t3-diff-table", "selected_rows"),
    State("t3-diff-table", "data"),
    prevent_initial_call=True,
)
def load_diff_map(selected_rows, data):
    """Load selected diff map, run IRLS, return 3 figures + metrics."""
    empty_fig = go.Figure()
    empty_fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False})

    if not selected_rows:
        return empty_fig, empty_fig, empty_fig, "", ""

    if not ZYGO_READER_AVAILABLE:
        return (
            empty_fig, empty_fig, empty_fig,
            "zygo_reader not installed — run: uv sync",
            "",
        )

    row = data[selected_rows[0]]
    full_path = row["fullPath"]
    file_name = os.path.basename(full_path)
    title_str = f"{row['Side']}  |  {row['DateStr']}  |  {file_name}"

    try:
        zdat = zygo_reader.DatReader(path_or_file_like=full_path)
        z = zdat.get_topography_nm()  # already in nm

        # IRLS robust polynomial decomposition
        predicted_z, anomaly_z, metrics = predict_normal_robust_polynomial(
            z, degree=5, n_iter=5, k_sigma=2.0
        )

        z_min = float(np.nanmin(z))
        z_max = float(np.nanmax(z))

        fig_raw = _make_heatmap_fig(z, f"Raw Diff — {title_str}", DARK_RAINBOW, z_min, z_max)
        fig_baseline = _make_heatmap_fig(
            predicted_z,
            f"Process Baseline (IRLS) — {title_str}",
            DARK_RAINBOW,
            z_min,
            z_max,
        )

        anom_abs_max = float(np.nanmax(np.abs(anomaly_z)))
        if anom_abs_max < 1e-10:
            anom_abs_max = 1.0
        fig_anomaly = _make_heatmap_fig(
            anomaly_z,
            f"Anomaly Signal — {title_str}",
            "RdBu_r",
            -anom_abs_max,
            anom_abs_max,
        )

        m = metrics
        metrics_text = (
            f"RMS: {m['rms_nm']:.1f} nm  |  "
            f"PV: {m['pv_nm']:.1f} nm  |  "
            f"Max |dev|: {m['max_nm']:.1f} nm  |  "
            f"Mean: {m['mean_nm']:.1f} nm  |  "
            f"Anomaly area: {m['anomaly_area_pct']:.0f}% > ±10 nm"
        )

        return fig_raw, fig_baseline, fig_anomaly, metrics_text, title_str

    except Exception as e:
        err_fig = go.Figure()
        err_fig.update_layout(
            title=f"Error: {e}",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return err_fig, empty_fig, empty_fig, f"Error: {e}", title_str


@callback(
    Output("t3-label-status", "children"),
    Output("t3-diff-table", "data", allow_duplicate=True),
    Input("t3-btn-normal", "n_clicks"),
    Input("t3-btn-anomaly", "n_clicks"),
    Input("t3-btn-skip", "n_clicks"),
    State("t3-diff-table", "selected_rows"),
    State("t3-diff-table", "data"),
    prevent_initial_call=True,
)
def apply_diff_label(n_normal, n_anomaly, n_skip, selected_rows, data):
    """Apply a label to the selected diff map row and persist."""
    if not selected_rows:
        return "No row selected — click a row first.", no_update

    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    label_map = {
        "t3-btn-normal": "Normal",
        "t3-btn-anomaly": "Anomaly",
        "t3-btn-skip": "Skip",
    }
    label = label_map.get(button_id)
    if label is None:
        return no_update, no_update

    row_idx = selected_rows[0]
    data[row_idx]["Label"] = label
    _save_diff_labels(data)

    # Also update the master in-memory list
    diff_table_data[row_idx]["Label"] = label

    file_name = os.path.basename(data[row_idx]["fullPath"])
    status = f"Labeled: {file_name} → {label}"
    return status, data


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Open in browser: http://127.0.0.1:8050")
    app.run(debug=True, host="127.0.0.1", port=8050)
