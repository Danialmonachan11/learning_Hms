"""
Dash web app for wafer image labeling.

Displays a table of wafer measurements, allows clicking rows to view
false-color topography images, and provides buttons to label each
measurement as normal or anomalous.

Run with: py -3.11 app.py  (from src/wafer_dev_predictor/ directory)
"""

import sys
from pathlib import Path

# Allow bare imports of color_map / color_scale (mirrors existing code style)
sys.path.insert(0, str(Path(__file__).parent / "data"))

import dash
import polars as pl
import fastlibrary as fl
from dash import dash_table, dcc, html, Input, Output, State, callback, no_update

from color_map import false_color_map_with_histogram

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
# Dash App
# ---------------------------------------------------------------------------
app = dash.Dash(__name__, title="Wafer Labeling Tool")

app.layout = html.Div(
    [
        html.H1("Wafer Dev Predictor — Image Labeling Tool"),
        # ---- Data Table ----
        dash_table.DataTable(
            id="measurements-table",
            columns=table_columns,
            data=table_data,
            row_selectable="single",
            selected_rows=[],
            filter_action="native",
            sort_action="native",
            page_size=20,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "8px", "fontSize": "14px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
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
        html.Hr(),
        # ---- Image Viewer ----
        dcc.Loading(
            id="loading-image",
            type="circle",
            children=dcc.Graph(id="image-viewer", style={"height": "600px"}),
        ),
        html.Hr(),
        # ---- Labeling Buttons ----
        html.Div(
            [
                html.Button(
                    "Normal_ZA",
                    id="btn-normal-za",
                    n_clicks=0,
                    style={"margin": "5px", "padding": "10px 20px", "backgroundColor": "#28a745", "color": "white", "border": "none", "cursor": "pointer", "fontSize": "16px"},
                ),
                html.Button(
                    "Normal_ZE",
                    id="btn-normal-ze",
                    n_clicks=0,
                    style={"margin": "5px", "padding": "10px 20px", "backgroundColor": "#28a745", "color": "white", "border": "none", "cursor": "pointer", "fontSize": "16px"},
                ),
                html.Button(
                    "Anomaly_ZA",
                    id="btn-anomaly-za",
                    n_clicks=0,
                    style={"margin": "5px", "padding": "10px 20px", "backgroundColor": "#dc3545", "color": "white", "border": "none", "cursor": "pointer", "fontSize": "16px"},
                ),
                html.Button(
                    "Anomaly_ZE",
                    id="btn-anomaly-ze",
                    n_clicks=0,
                    style={"margin": "5px", "padding": "10px 20px", "backgroundColor": "#dc3545", "color": "white", "border": "none", "cursor": "pointer", "fontSize": "16px"},
                ),
            ],
            style={"textAlign": "center"},
        ),
        # ---- Status Message ----
        html.Div(id="label-status", style={"textAlign": "center", "padding": "10px", "fontSize": "16px", "fontWeight": "bold"}),
        # Hidden store for selected row info
        dcc.Store(id="selected-source-path"),
    ],
    style={"maxWidth": "1400px", "margin": "0 auto", "padding": "20px"},
)


# ---------------------------------------------------------------------------
# Callbacks
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
        # Return an empty figure with error message
        import plotly.graph_objects as go

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

    # Determine which button was clicked
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

    # Update the label in the table data
    row_idx = selected_rows[0]
    row = data[row_idx]
    data[row_idx]["Label"] = label

    # Save to parquet
    _save_labels(data)

    status = f"Labeled: {row['Serial']} | {row['Side']} | {row['ProcessStep']} → {label}"
    return status, data


def _save_labels(data: list[dict]) -> None:
    """Persist current labels to the labeled parquet file."""
    import pandas as pd

    LABELED_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    # Save only rows that have a label
    labeled = df[df["Label"].notna() & (df["Label"] != "")]
    if not labeled.empty:
        labeled.to_parquet(LABELED_DB_PATH, index=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Open in browser: http://127.0.0.1:8050")
    app.run(debug=True, host="127.0.0.1", port=8050)
