"""
Dash web app for wafer image labeling and anomaly analysis.

Tab 1 — Labeling:
    Displays a table of wafer measurements, allows clicking rows to view
    false-color topography images, and provides buttons to label each
    measurement as normal or anomalous.

Tab 2 — Anomaly Analysis:
    Select a measurement, mark the anomalous column range via a range slider,
    fit a 2D polynomial to the clean region, and visualise the predicted normal
    surface alongside the anomaly map with quantitative metrics.

Run with: py -3.11 app.py  (from src/wafer_dev_predictor/ directory)
"""

import sys
from pathlib import Path

# Allow bare imports of color_map / color_scale and analysis subpackage
sys.path.insert(0, str(Path(__file__).parent / "data"))

import numpy as np
import dash
import polars as pl
import fastlibrary as fl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dash_table, dcc, html, Input, Output, State, callback, no_update

from color_map import false_color_map_with_histogram
from color_scale import DARK_RAINBOW
from analysis.prediction import predict_normal_polynomial, predict_normal_gaussian

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATABASE_PATH = (
    R"T:\asm\EXE_ZdMB\10_EXE_MB_Flatness_Database"
    R"\Flatness Reports\MiQaT_Specification_Data.parquet"
)
LABELED_DB_PATH = Path("data/labeled_measurements.parquet")

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
# Initialise data
# ---------------------------------------------------------------------------
measurements_df = load_measurements()
measurements_df = load_or_init_labels(measurements_df)

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

# ---------------------------------------------------------------------------
# Tab 2 — Pre-compute dropdown options
# ---------------------------------------------------------------------------
_serials = sorted(measurements_df["Serial"].unique().to_list())
_serial_options = [{"label": s, "value": s} for s in _serials]
_initial_serial = _serials[0] if _serials else None

_process_steps_init = (
    sorted(
        measurements_df.filter(
            (pl.col("Serial") == _initial_serial) & (pl.col("Side") == "ZA")
        )["ProcessStep"]
        .unique()
        .to_list()
    )
    if _initial_serial
    else []
)
_init_step_opts = [{"label": s, "value": s} for s in _process_steps_init]
_init_step_val = _process_steps_init[0] if _process_steps_init else None


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
                # TAB 2 — Anomaly Analysis
                # ================================================================
                dcc.Tab(
                    label="Anomaly Analysis",
                    children=[
                        html.Div(
                            style={"padding": "10px"},
                            children=[
                                # ---- Controls row ----
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "flexWrap": "wrap",
                                        "gap": "16px",
                                        "alignItems": "flex-end",
                                        "padding": "10px 0 16px 0",
                                        "borderBottom": "1px solid #ddd",
                                    },
                                    children=[
                                        # Serial
                                        html.Div(
                                            [
                                                html.Div("Serial", style=CTRL_LABEL_STYLE),
                                                dcc.Dropdown(
                                                    id="t2-serial",
                                                    options=_serial_options,
                                                    value=_initial_serial,
                                                    clearable=False,
                                                    style={"minWidth": "140px"},
                                                ),
                                            ]
                                        ),
                                        # Side
                                        html.Div(
                                            [
                                                html.Div("Side", style=CTRL_LABEL_STYLE),
                                                dcc.RadioItems(
                                                    id="t2-side",
                                                    options=[
                                                        {"label": " ZA", "value": "ZA"},
                                                        {"label": " ZE", "value": "ZE"},
                                                    ],
                                                    value="ZA",
                                                    inline=True,
                                                    inputStyle={"marginRight": "4px"},
                                                    labelStyle={"marginRight": "12px"},
                                                ),
                                            ]
                                        ),
                                        # Process Step
                                        html.Div(
                                            [
                                                html.Div("Process Step", style=CTRL_LABEL_STYLE),
                                                dcc.Dropdown(
                                                    id="t2-process-step",
                                                    options=_init_step_opts,
                                                    value=_init_step_val,
                                                    clearable=False,
                                                    style={"minWidth": "200px"},
                                                ),
                                            ]
                                        ),
                                        # Method
                                        html.Div(
                                            [
                                                html.Div("Method", style=CTRL_LABEL_STYLE),
                                                dcc.RadioItems(
                                                    id="t2-method",
                                                    options=[
                                                        {
                                                            "label": " Polynomial",
                                                            "value": "Polynomial",
                                                        },
                                                        {
                                                            "label": " Gaussian",
                                                            "value": "Gaussian",
                                                        },
                                                    ],
                                                    value="Polynomial",
                                                    inline=True,
                                                    inputStyle={"marginRight": "4px"},
                                                    labelStyle={"marginRight": "12px"},
                                                ),
                                            ]
                                        ),
                                        # Poly degree
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
                                        # Gauss FWHM slider
                                        html.Div(
                                            style={"minWidth": "220px"},
                                            children=[
                                                html.Div(
                                                    "Gauss FWHM (mm)", style=CTRL_LABEL_STYLE
                                                ),
                                                dcc.Slider(
                                                    id="t2-gauss-fwhm",
                                                    min=1,
                                                    max=20,
                                                    step=0.5,
                                                    value=5,
                                                    marks={1: "1", 5: "5", 10: "10", 20: "20"},
                                                    tooltip={
                                                        "placement": "bottom",
                                                        "always_visible": True,
                                                    },
                                                ),
                                            ],
                                        ),
                                        # Analyse button
                                        html.Button(
                                            "Analyse",
                                            id="t2-analyse-btn",
                                            n_clicks=0,
                                            style={
                                                **BTN_STYLE,
                                                "backgroundColor": "#007bff",
                                                "alignSelf": "flex-end",
                                            },
                                        ),
                                    ],
                                ),
                                # ---- Original image ----
                                html.Div(
                                    "Original",
                                    style={
                                        "fontWeight": "bold",
                                        "fontSize": "14px",
                                        "marginTop": "14px",
                                        "marginBottom": "2px",
                                    },
                                ),
                                dcc.Loading(
                                    type="circle",
                                    children=dcc.Graph(
                                        id="t2-original-graph",
                                        style={"height": "300px"},
                                        config={"responsive": True},
                                    ),
                                ),
                                # ---- Range slider for anomalous columns ----
                                html.Div(
                                    style={"padding": "8px 20px 4px 20px"},
                                    children=[
                                        html.Div(
                                            "Anomalous region (% of image columns):",
                                            style={
                                                "fontSize": "13px",
                                                "fontWeight": "bold",
                                                "marginBottom": "4px",
                                            },
                                        ),
                                        dcc.RangeSlider(
                                            id="t2-anom-slider",
                                            min=0,
                                            max=100,
                                            step=1,
                                            value=[60, 100],
                                            marks={
                                                0: "0%",
                                                20: "20%",
                                                40: "40%",
                                                60: "60%",
                                                80: "80%",
                                                100: "100%",
                                            },
                                            tooltip={
                                                "placement": "bottom",
                                                "always_visible": True,
                                            },
                                        ),
                                    ],
                                ),
                                # ---- Predicted Normal image ----
                                html.Div(
                                    "Predicted Normal",
                                    style={
                                        "fontWeight": "bold",
                                        "fontSize": "14px",
                                        "marginTop": "10px",
                                        "marginBottom": "2px",
                                    },
                                ),
                                dcc.Loading(
                                    type="circle",
                                    children=dcc.Graph(
                                        id="t2-predicted-graph",
                                        style={"height": "300px"},
                                        config={"responsive": True},
                                    ),
                                ),
                                # ---- Anomaly Map ----
                                html.Div(
                                    "Anomaly Map  (actual − predicted)",
                                    style={
                                        "fontWeight": "bold",
                                        "fontSize": "14px",
                                        "marginTop": "10px",
                                        "marginBottom": "2px",
                                    },
                                ),
                                dcc.Loading(
                                    type="circle",
                                    children=dcc.Graph(
                                        id="t2-anomaly-graph",
                                        style={"height": "300px"},
                                        config={"responsive": True},
                                    ),
                                ),
                                # ---- Metrics ----
                                html.Div(
                                    id="t2-metrics",
                                    style={
                                        "padding": "10px 5px",
                                        "fontSize": "15px",
                                        "fontWeight": "bold",
                                        "color": "#333",
                                    },
                                ),
                            ],
                        )
                    ],
                ),
            ]
        ),
        dcc.Store(id="selected-source-path"),
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
    Output("t2-process-step", "options"),
    Output("t2-process-step", "value"),
    Input("t2-serial", "value"),
    Input("t2-side", "value"),
)
def update_t2_process_steps(serial, side):
    """Update process step dropdown options when serial or side changes."""
    if not serial or not side:
        return [], None

    steps = sorted(
        measurements_df.filter(
            (pl.col("Serial") == serial) & (pl.col("Side") == side)
        )["ProcessStep"]
        .unique()
        .to_list()
    )
    if not steps:
        return [], None
    return [{"label": s, "value": s} for s in steps], steps[0]


@callback(
    Output("t2-anom-slider", "value"),
    Input("t2-side", "value"),
)
def update_t2_slider_default(side):
    """Reset anomalous region slider to the canonical default when side changes."""
    if side == "ZE":
        return [0, 40]  # ZE: left 40%
    return [60, 100]  # ZA: right 40% (default)


@callback(
    Output("t2-original-graph", "figure"),
    Output("t2-predicted-graph", "figure"),
    Output("t2-anomaly-graph", "figure"),
    Output("t2-metrics", "children"),
    Input("t2-analyse-btn", "n_clicks"),
    State("t2-serial", "value"),
    State("t2-side", "value"),
    State("t2-process-step", "value"),
    State("t2-method", "value"),
    State("t2-poly-degree", "value"),
    State("t2-gauss-fwhm", "value"),
    State("t2-anom-slider", "value"),
    prevent_initial_call=True,
)
def run_analysis(n_clicks, serial, side, process_step, method, degree, fwhm_mm, slider_value):
    """
    Run polynomial or Gaussian surface prediction and return three stacked figures + metrics.

    Flow:
        1. Look up SourcePath from measurements_df
        2. Load .wrk file via fl.read_zygo
        3. Convert slider % to column indices
        4. Run prediction (polynomial or Gaussian)
        5. Build 3 Plotly figures + metrics string
    """
    empty_fig = go.Figure()
    empty_fig.update_layout(xaxis={"visible": False}, yaxis={"visible": False})

    if not all([serial, side, process_step]):
        return empty_fig, empty_fig, empty_fig, "Please select Serial, Side, and Process Step."

    row = measurements_df.filter(
        (pl.col("Serial") == serial)
        & (pl.col("Side") == side)
        & (pl.col("ProcessStep") == process_step)
    )
    if row.is_empty():
        return empty_fig, empty_fig, empty_fig, "Measurement not found in database."

    source_path = row["SourcePath"][0]

    try:
        topo = fl.read_zygo(source_path)
        z = topo.z_map * 1e9  # height map in nm
        n_rows, n_cols = z.shape

        # Convert slider percentages to column indices
        anom_col_start = int(slider_value[0] / 100 * n_cols)
        anom_col_end = int(slider_value[1] / 100 * n_cols)

        title_base = f"{serial} | {side} | {process_step}"

        if method == "Polynomial":
            predicted_z, anomaly_z, metrics = predict_normal_polynomial(
                z,
                anom_col_start=anom_col_start,
                anom_col_end=anom_col_end,
                degree=int(degree),
            )
        else:
            predicted_z, anomaly_z, metrics = predict_normal_gaussian(
                topo,
                fwhm_m=float(fwhm_mm) / 1000.0,
                anom_col_start=anom_col_start,
                anom_col_end=anom_col_end,
            )

        # Shared z range for original + predicted (comparable colorscales)
        z_min = float(min(np.nanmin(z), np.nanmin(predicted_z)))
        z_max = float(max(np.nanmax(z), np.nanmax(predicted_z)))

        fig_original = _make_heatmap_fig(
            z, f"Original — {title_base}", DARK_RAINBOW, z_min, z_max
        )
        fig_predicted = _make_heatmap_fig(
            predicted_z,
            f"Predicted Normal ({method}) — {title_base}",
            DARK_RAINBOW,
            z_min,
            z_max,
        )

        # Anomaly map: diverging colorscale (RdBu_r), symmetric around 0
        anom_abs_max = float(np.nanmax(np.abs(anomaly_z)))
        if anom_abs_max < 1e-10:
            anom_abs_max = 1.0
        fig_anomaly = _make_heatmap_fig(
            anomaly_z,
            f"Anomaly Map (actual − predicted) — {title_base}",
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

        return fig_original, fig_predicted, fig_anomaly, metrics_text

    except Exception as e:
        err_fig = go.Figure()
        err_fig.update_layout(
            title=f"Error: {e}",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return err_fig, empty_fig, empty_fig, f"Error: {e}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Open in browser: http://127.0.0.1:8050")
    app.run(debug=True, host="127.0.0.1", port=8050)
