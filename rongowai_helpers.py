

import pandas as pd
import re
from IPython.display import display, HTML
import netCDF4 as nc
import os
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Polygon
from geopy import distance
from multiprocessing import Pool, cpu_count
from shapely.ops import transform
from functools import partial
from math import cos, sin, pi
import scipy.spatial as sp
import pandas as pd
import sqlite3

import numpy as np
import lonboard
from lonboard.basemap import MaplibreBasemap
#from lonboard import BitmapTileLayer
from lonboard.layer_extension import PathStyleExtension

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import json


SATELLITE_STYLE = "https://raw.githubusercontent.com/go2garret/maps/main/src/assets/json/arcgis_hybrid.json"
PARALLEL = True



def plot_ddm_interactive(
    gdf,
    ds,
    ddm_var="surface_reflectivity",
    surface_type_col="sp_surface_type",
    surface_type_labels=None,
    metadata_cols=None,
    cmap="plasma",
):
    """
    Interactively plot Delay Doppler Maps (DDMs) for one randomly selected
    point per surface type.

    Parameters
    ----------
    gdf                  : GeoDataFrame with Rongowai observations (sample, ddm columns)
    ds                   : xarray Dataset opened from the Rongowai netCDF file
    ddm_var              : name of the 4D DDM variable in ds [sample, ddm, delay, doppler]
    surface_type_col     : column in gdf containing integer surface type codes
    surface_type_labels  : optional dict of {int: str} surface type labels
    metadata_cols        : list of columns to show in sample table (None = defaults)
    cmap                 : plotly colorscale name
    """
    import numpy as np
    import pandas as pd
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # --- Defaults -----------------------------------------------------------
    if metadata_cols is None:
        metadata_cols = ["sp_inc_angle", "ddm_snr", "coherence_state",
                         "sp_dist_to_coast_km"]
    metadata_cols = [c for c in metadata_cols if c in gdf.columns]

    if surface_type_labels is None:
        surface_type_labels = {}

    # --- Layout -------------------------------------------------------------
    surface_types = sorted(gdf[surface_type_col].dropna().unique().astype(int).tolist())
    surface_types = surface_types[:9]
    n_types = len(surface_types)
    n_cols  = 3
    n_rows  = int(np.ceil(n_types / n_cols))

    # --- Widgets ------------------------------------------------------------
    btn_redraw = widgets.Button(
        description="Redraw sample",
        button_style="primary",
        icon="refresh",
        layout=widgets.Layout(width="150px"),
    )
    btn_toggle_3d = widgets.ToggleButton(
        value=False,
        description="Switch to 3D",
        button_style="info",
        layout=widgets.Layout(width="140px"),
    )
    btn_toggle_scale = widgets.ToggleButton(
        value=False,
        description="Switch to independent scale",
        button_style="",
        layout=widgets.Layout(width="230px"),
    )
    w_cmap = widgets.Dropdown(
        options=['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
             'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
             'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
             'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
             'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
             'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
             'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
             'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
             'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
             'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
             'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
             'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
             'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
             'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
             'ylorrd'],
        value=cmap,
        description="Colorscale:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="240px"),
    )

    out_plot = widgets.Output()
    out_info = widgets.Output()

    current_sample = {"rows": None}

    # --- Sampling -----------------------------------------------------------
    def draw_sample():
        rows = {}
        for st in surface_types:
            subset = gdf[gdf[surface_type_col].astype(int) == st]
            rows[st] = subset.sample(1).iloc[0] if len(subset) > 0 else None
        current_sample["rows"] = rows
        return rows

    # --- DDM extraction -----------------------------------------------------
    def extract_ddm(row):
        try:
            sample_idx = int(row["sample"])
            ddm_idx    = int(row["ddm"])
            return ds[ddm_var].isel(sample=sample_idx, ddm=ddm_idx).values.astype(float)
        except Exception:
            return None

    # --- Subplot title with stats ------------------------------------------
    def make_title(st, row, arr=None):
        label = surface_type_labels.get(st, f"Type {st}")
        if row is None:
            return f"{label}<br><small>no data</small>"
        if arr is not None:
            finite_vals = arr[np.isfinite(arr)]
            if len(finite_vals) > 0:
                stats_str = (
                    f"min={np.nanmin(finite_vals):.3g}  "
                    f"med={np.nanmedian(finite_vals):.3g}  "
                    f"max={np.nanmax(finite_vals):.3g}"
                )
                return f"{label}<br><small>{stats_str}</small>"
        return label

    # --- Build figure -------------------------------------------------------
    def build_figure(rows, use_3d, independent_scale):
        is_3d = use_3d

        # --- Extract all DDM arrays first so titles can include stats -------
        all_vals   = []
        ddm_arrays = {}
        for st in surface_types:
            row = rows.get(st)
            arr = extract_ddm(row) if row is not None else None
            ddm_arrays[st] = arr
            if arr is not None:
                all_vals.extend(arr[np.isfinite(arr)].tolist())

        # --- Shared scale ---------------------------------------------------
        if all_vals and not independent_scale:
            shared_zmin = float(np.percentile(all_vals, 2))
            shared_zmax = float(np.percentile(all_vals, 98))
        else:
            shared_zmin = shared_zmax = None

        # --- Titles built after arrays are available ------------------------
        titles = [
            make_title(st, rows.get(st), ddm_arrays.get(st))
            for st in surface_types
        ]
        while len(titles) < n_rows * n_cols:
            titles.append("")

        specs = [
            [{"type": "surface" if is_3d else "heatmap"} for _ in range(n_cols)]
            for _ in range(n_rows)
        ]

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=titles,
            specs=specs,
            column_widths=[0.22, 0.22, 0.22],
            horizontal_spacing=0.05,
            vertical_spacing=0.18,
        )

        colorscale     = w_cmap.value
        colorbar_title = "Percentile" if independent_scale else ddm_var

        for i, st in enumerate(surface_types):
            r   = i // n_cols + 1
            c   = i % n_cols  + 1
            arr = ddm_arrays.get(st)

            if arr is None:
                continue

            # --- Scale and display array ------------------------------------
            if independent_scale:
                arr_display = np.full_like(arr, np.nan, dtype=float)
                finite_mask = np.isfinite(arr)
                if finite_mask.any():
                    flat  = arr[finite_mask]
                    ranks = np.argsort(np.argsort(flat))
                    arr_display[finite_mask] = ranks / (len(ranks) - 1) * 100
                zmin, zmax = 0.0, 100.0
            else:
                arr_display = arr
                zmin        = shared_zmin
                zmax        = shared_zmax

            # --- Axis labels ------------------------------------------------
            try:
                delay_vals = ds["delay"].values.tolist()
                dopp_vals  = ds["doppler"].values.tolist()
            except Exception:
                delay_vals = list(range(arr.shape[0]))
                dopp_vals  = list(range(arr.shape[1]))

            # --- Single colorbar on right, first trace only -----------------
            showscale = (i == 0)
            colorbar  = dict(
                x=1.02,
                y=0.5,
                yanchor="middle",
                len=0.8,
                thickness=15,
                tickfont=dict(size=9),
                title=dict(text=colorbar_title, side="right", font=dict(size=9)),
            ) if showscale else None

            # --- Trace ------------------------------------------------------
            if is_3d:
                trace = go.Surface(
                    z=arr_display,
                    x=dopp_vals,
                    y=delay_vals,
                    colorscale=colorscale,
                    cmin=zmin, cmax=zmax,
                    showscale=showscale,
                    colorbar=colorbar,
                )
            else:
                trace = go.Heatmap(
                    z=arr_display,
                    x=dopp_vals,
                    y=delay_vals,
                    colorscale=colorscale,
                    zmin=zmin, zmax=zmax,
                    showscale=showscale,
                    colorbar=colorbar,
                )

            fig.add_trace(trace, row=r, col=c)

            # --- Axis labels on outermost subplots only ---------------------
            if not is_3d:
                axis_suffix = "" if i == 0 else str(i + 1)
                fig.update_layout(**{
                    f"xaxis{axis_suffix}": dict(
                        title="Doppler bin" if r == n_rows else ""
                    ),
                    f"yaxis{axis_suffix}": dict(
                        title="Delay bin" if c == 1 else ""
                    ),
                })

        mode_label  = "3D Surface" if is_3d else "2D Heatmap"
        scale_label = "independent scale" if independent_scale else "shared scale"

        fig.update_layout(
            title=dict(
                text=f"DDMs — {mode_label}, {scale_label}",
                font=dict(size=14),
            ),
            height=440 * n_rows,
            margin=dict(l=60, r=120, t=80, b=60),
        )

        # Make subplot titles render HTML (enables <br> and <small>)
        for annotation in fig.layout.annotations:
            annotation.update(font=dict(size=11))

        return fig

    # --- Render -------------------------------------------------------------
    def render(resample):
        if resample or current_sample["rows"] is None:
            rows = draw_sample()
        else:
            rows = current_sample["rows"]

        with out_info:
            clear_output(wait=True)
            info_rows = []
            for st, row in rows.items():
                if row is not None:
                    info_rows.append({
                        "surface_type": surface_type_labels.get(st, f"Type {st}"),
                        "sample":       int(row["sample"]),
                        "ddm":          int(row["ddm"]),
                        **{c: f"{row[c]:.3g}" if pd.notna(row[c]) else ""
                           for c in metadata_cols},
                    })
            if info_rows:
                from itables import show as itshow
                import itables.options as ito
                ito.warn_on_undocumented_option = False
                itshow(
                    pd.DataFrame(info_rows),
                    caption="Selected samples",
                    classes="display compact",
                    paging=False,
                    searching=False,
                )

        with out_plot:
            clear_output(wait=True)
            fig = build_figure(
                rows,
                use_3d=btn_toggle_3d.value,
                independent_scale=btn_toggle_scale.value,
            )
            fig.show()

    # --- Event handlers -----------------------------------------------------
    def on_redraw(_):
        render(resample=True)

    def on_toggle_3d(change):
        btn_toggle_3d.description = "Switch to 2D" if change["new"] else "Switch to 3D"
        render(resample=False)

    def on_toggle_scale(change):
        btn_toggle_scale.description = (
            "Switch to shared scale" if change["new"] else "Switch to independent scale"
        )
        render(resample=False)

    def on_cmap_change(_):
        render(resample=False)

    btn_redraw.on_click(on_redraw)
    btn_toggle_3d.observe(on_toggle_3d, names="value")
    btn_toggle_scale.observe(on_toggle_scale, names="value")
    w_cmap.observe(on_cmap_change, names="value")

    # --- Layout -------------------------------------------------------------
    controls = widgets.HBox([
        btn_redraw,
        btn_toggle_3d,
        btn_toggle_scale,
        w_cmap,
    ])

    display(widgets.VBox([
        controls,
        out_info,
        out_plot,
    ]))

    render(resample=True)

def plot_rongowai_interactive(
    gdf,
    filters=None,
    default_x="sp_inc_angle",
    default_y="surface_reflectivity_peak",
    default_color="coherence_metric",
    df_dict=None,
):
    """
    Interactive exploratory plotting for Rongowai L1 data.

    Parameters
    ----------
    gdf           : GeoDataFrame with Rongowai observations
    filters       : optional dict of {column: (min, max)} baseline filters
    default_x     : default X axis variable
    default_y     : default Y axis variable
    default_color : default colour variable
    df_dict       : optional data dictionary DataFrame for long names in labels
    """
    import numpy as np
    import pandas as pd
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import plotly.express as px
    import plotly.graph_objects as go

    # --- Prepare DataFrame --------------------------------------------------
    df = gdf.drop(columns="geometry").copy()
    num_cols = sorted([
        c for c in df.columns
        if np.issubdtype(df[c].dtype, np.number)
    ])

    # Long name lookup from data dictionary
    def long_name(col):
        if df_dict is not None and "Name" in df_dict.columns and "Long_name" in df_dict.columns:
            row = df_dict[df_dict["Name"] == col]["Long_name"].values
            if len(row) and row[0] not in ("", "<none>"):
                return f"{col} — {row[0]}"
        return col

    # --- Apply baseline filter dict -----------------------------------------
    filters = filters or {}
    baseline_mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in filters.items():
        if col in df.columns:
            baseline_mask &= df[col].between(lo, hi) | df[col].isna()
    df_baseline = df[baseline_mask].copy()

    # --- Widgets ------------------------------------------------------------
    CMAPS = ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
             'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
             'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
             'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
             'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
             'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
             'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
             'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
             'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
             'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
             'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
             'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
             'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
             'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
             'ylorrd',]

    def make_dropdown(desc, default):
        val = default if default in num_cols else num_cols[0]
        return widgets.Dropdown(
            options=num_cols, value=val,
            description=desc,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="320px"),
        )

    w_x      = make_dropdown("X axis:", default_x)
    w_y      = make_dropdown("Y axis:", default_y)
    w_color  = make_dropdown("Colour:", default_color)
    w_hist   = make_dropdown("Histogram:", default_x)
    w_cmap   = widgets.Dropdown(
        options=CMAPS, value="plasma",
        description="Colormap:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="220px"),
    )
    w_opacity = widgets.FloatSlider(
        value=0.6, min=0.1, max=1.0, step=0.05,
        description="Opacity:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="320px"),
        continuous_update=False,
    )
    w_bins = widgets.IntSlider(
        value=50, min=5, max=200, step=5,
        description="Bins:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="320px"),
        continuous_update=False,
    )
    w_kde = widgets.Checkbox(
        value=True, description="Overlay KDE",
        layout=widgets.Layout(width="150px"),
    )
    w_nan = widgets.Checkbox(
        value=True, description="Drop NaN rows (axes only)",
        layout=widgets.Layout(width="220px"),
    )

    # --- Interactive range filters on top of baseline ----------------------
    filter_rows   = []
    filter_sliders = {}

    for col in num_cols:
        if col not in df_baseline.columns:
            continue
        col_data = df_baseline[col].dropna()
        if len(col_data) == 0:        # skip columns with no valid data after baseline filter
            continue
        col_min = float(col_data.min())
        col_max = float(col_data.max())
        if col_min == col_max:
            continue

        span = col_max - col_min
        step = 10 ** (np.floor(np.log10(span)) - 2) if span > 0 else 0.01
        step = round(step, 10)

        slider = widgets.FloatRangeSlider(
            value=[col_min, col_max],
            min=col_min, max=col_max, step=step,
            description=col,
            style={"description_width": "160px"},
            layout=widgets.Layout(width="560px"),
            continuous_update=False,
            readout=True,
        )
        filter_sliders[col] = {"slider": slider, "min": col_min, "max": col_max}
        filter_rows.append(slider)

    btn_reset_filters = widgets.Button(
        description="Reset filters",
        layout=widgets.Layout(width="130px"),
    )

    filters_accordion = widgets.Accordion(
        children=[widgets.VBox(
            filter_rows + [btn_reset_filters],
            layout=widgets.Layout(max_height="300px", overflow_y="auto"),
        )],
    )
    filters_accordion.set_title(0, "Interactive filters (applied on top of baseline)")
    filters_accordion.selected_index = None   # collapsed by default

    # --- Output areas -------------------------------------------------------
    out_stats   = widgets.Output()
    out_scatter = widgets.Output()
    out_hist    = widgets.Output()
    out_pairs   = widgets.Output()

    # --- Core logic ---------------------------------------------------------
    def get_filtered_df():
        mask = pd.Series(True, index=df_baseline.index)
        for col, w in filter_sliders.items():
            lo, hi = w["slider"].value
            # Only apply if moved from default
            if lo > w["min"] or hi < w["max"]:
                mask &= df_baseline[col].between(lo, hi) | df_baseline[col].isna()
        return df_baseline[mask]

    def show_stats(dff):
        with out_stats:
            clear_output(wait=True)
            cols = [w_x.value, w_y.value, w_color.value, w_hist.value]
            cols = list(dict.fromkeys(cols))   # deduplicate, preserve order
            sub  = dff[cols].astype(float)
            stats = sub.agg(["min", "mean", "median", "max", "std"]).T
            stats["nan_count"] = sub.isna().sum()
            stats.index.name = "variable"
            stats = stats.reset_index()
            stats.insert(1, "long_name", stats["variable"].apply(long_name))
            for c in ["min", "mean", "median", "max", "std"]:
                stats[c] = stats[c].apply(lambda x: f"{x:.4g}")
            from itables import show as itshow
            import itables.options as ito
            ito.warn_on_undocumented_option = False
            itshow(
                stats,
                caption=f"Summary statistics — {len(dff)} rows"
                        + (f" (baseline filtered from {len(df)} total)" if filters else ""),
                classes="display compact",
                paging=False,
                searching=False,
            )

    def show_scatter(dff):
        with out_scatter:
            clear_output(wait=True)
            plot_df = dff.copy()
            if w_nan.value:
                plot_df = plot_df.dropna(subset=[w_x.value, w_y.value, w_color.value])
            fig = px.scatter(
                plot_df,
                x=w_x.value,
                y=w_y.value,
                color=w_color.value,
                color_continuous_scale=w_cmap.value,
                opacity=w_opacity.value,
                hover_data=list(plot_df.columns),
                labels={
                    w_x.value:     long_name(w_x.value),
                    w_y.value:     long_name(w_y.value),
                    w_color.value: long_name(w_color.value),
                },
                title=f"{long_name(w_y.value)} vs {long_name(w_x.value)}"
                      f" — coloured by {long_name(w_color.value)}"
                      f" (n={len(plot_df):,})",
                render_mode="webgl",
            )
            fig.update_layout(height=500, margin=dict(l=60, r=20, t=50, b=60))
            fig.show()

    def show_histogram(dff):
        with out_hist:
            clear_output(wait=True)
            col      = w_hist.value
            plot_df  = dff[[col]].dropna()
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=plot_df[col],
                nbinsx=w_bins.value,
                name=col,
                marker_color="steelblue",
                opacity=0.75,
            ))
            if w_kde.value:
                try:
                    from scipy.stats import gaussian_kde
                    kde_x = np.linspace(plot_df[col].min(), plot_df[col].max(), 300)
                    kde_y = gaussian_kde(plot_df[col])(kde_x)
                    # Scale KDE to histogram counts
                    bin_width = (plot_df[col].max() - plot_df[col].min()) / w_bins.value
                    kde_y_scaled = kde_y * len(plot_df) * bin_width
                    fig.add_trace(go.Scatter(
                        x=kde_x, y=kde_y_scaled,
                        mode="lines",
                        name="KDE",
                        line=dict(color="crimson", width=2),
                    ))
                except ImportError:
                    pass   # scipy not available, skip KDE silently

            fig.update_layout(
                title=f"Histogram — {long_name(col)} (n={len(plot_df):,})",
                xaxis_title=long_name(col),
                yaxis_title="Count",
                height=400,
                bargap=0.05,
                margin=dict(l=60, r=20, t=50, b=60),
            )
            fig.show()

    def show_pairs(dff):
        with out_pairs:
            clear_output(wait=True)
            cols    = list(dict.fromkeys([w_x.value, w_y.value, w_color.value]))
            plot_df = dff[cols].dropna()
            fig = px.scatter_matrix(
                plot_df,
                dimensions=cols,
                color=w_color.value,
                color_continuous_scale=w_cmap.value,
                opacity=w_opacity.value,
                labels={c: c for c in cols},
                title=f"Scatter matrix — {len(plot_df):,} rows",
            )
            fig.update_traces(diagonal_visible=False, marker=dict(size=2))
            fig.update_layout(height=550, margin=dict(l=60, r=20, t=50, b=60))
            fig.show()

    def refresh(_):
        dff = get_filtered_df()
        show_stats(dff)
        show_scatter(dff)
        show_histogram(dff)
        show_pairs(dff)

    def on_reset_filters(_):
        for col, w in filter_sliders.items():
            w["slider"].value = [w["min"], w["max"]]

    btn_reset_filters.on_click(on_reset_filters)

    # Observe all controls
    for w in [w_x, w_y, w_color, w_hist, w_cmap, w_opacity, w_bins, w_kde, w_nan]:
        w.observe(refresh, names="value")
    for w in filter_sliders.values():
        w["slider"].observe(refresh, names="value")

    # --- Layout -------------------------------------------------------------
    scatter_controls = widgets.VBox([
        widgets.HBox([w_x, w_y, w_color]),
        widgets.HBox([w_cmap, w_opacity, w_nan]),
    ])
    hist_controls = widgets.HBox([w_hist, w_bins, w_kde])

    tab = widgets.Tab()
    tab.children = [
        widgets.VBox([scatter_controls, out_scatter]),
        widgets.VBox([hist_controls, out_hist]),
        widgets.VBox([out_pairs]),
    ]
    tab.set_title(0, "Scatter")
    tab.set_title(1, "Histogram")
    tab.set_title(2, "Scatter matrix")

    # Lazy render: only refresh a tab when it is first selected
    rendered = {0: False, 1: False, 2: False}

    def on_tab_change(change):
        idx = change["new"]
        if not rendered[idx]:
            dff = get_filtered_df()
            if idx == 0:
                show_scatter(dff)
            elif idx == 1:
                show_histogram(dff)
            elif idx == 2:
                show_pairs(dff)
            rendered[idx] = True

    tab.observe(on_tab_change, names="selected_index")

    display(widgets.VBox([
        filters_accordion,
        widgets.HTML("<b>Summary statistics:</b>"),
        out_stats,
        tab,
    ]))

    # Initial render — scatter tab only
    dff = get_filtered_df()
    show_stats(dff)
    show_scatter(dff)
    rendered[0] = True


def map_rongowai_interactive(
    gdf,
    radius_attr="fresnel_minor",
    gdf_points=None,
    gdf_track=None,
    default_attribute="surface_reflectivity_peak",
    default_cmap="plasma",
    selected_cols=None,
):
    """
    Interactive Rongowai map with attribute and colormap selectors.
    Automatically detects Point or Polygon geometry type.

    Parameters
    ----------
    gdf               : GeoDataFrame with Point or Polygon geometry
    radius_attr       : column to use as point radius in metres (points only)
    gdf_points        : optional GeoDataFrame of Point geometry (aircraft positions)
    gdf_track         : optional GeoDataFrame of LineString geometry (flight track)
    default_attribute : column to colour on initial render
    default_cmap      : matplotlib colormap name for initial render
    selected_cols      : columns to select (None = all)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import matplotlib.patches as mpatches
    import ipywidgets as widgets
    import lonboard
    from lonboard import SolidPolygonLayer
    from lonboard.basemap import MaplibreBasemap
    from lonboard.layer_extension import PathStyleExtension
    from IPython.display import display, clear_output

    # --- Detect geometry type -----------------------------------------------
    geom_type = gdf.geometry.geom_type.iloc[0]
    is_polygon = "Polygon" in geom_type
    
    # Numeric columns for attribute selector
    num_cols = sorted([
        c for c in gdf.columns
        if c != "geometry" and np.issubdtype(gdf[c].dtype, np.number)
    ])

    CMAPS = [
        "plasma", "viridis", "inferno", "magma", "cividis",
        "RdYlGn", "RdYlBu", "coolwarm", "turbo", "rainbow",
        "tab10", "tab20","Set2", "Accent", "Dark2", "Paired",
    ]


    # --- Widgets ------------------------------------------------------------
    w_attribute = widgets.Dropdown(
        options=num_cols,
        value=default_attribute if default_attribute in num_cols else num_cols[0],
        description="Attribute:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="350px"),
    )
    w_cmap = widgets.Dropdown(
        options=CMAPS,
        value=default_cmap if default_cmap in CMAPS else CMAPS[0],
        description="Colormap:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="350px"),
    )
    w_percentile = widgets.IntRangeSlider(
        value=[5, 95],
        min=0, max=100, step=1,
        description="Color range (percentile):",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="450px"),
        continuous_update=False,
    )
    btn_render = widgets.Button(
        description="Render map",
        button_style="primary",
        layout=widgets.Layout(width="120px"),
    )

    out_map      = widgets.Output()
    out_colorbar = widgets.Output()

    # --- Build layers -------------------------------------------------------
    def build_map(_):
        attribute      = w_attribute.value
        cmap_name      = w_cmap.value
        pct_lo, pct_hi = w_percentile.value

        with out_map:
            clear_output(wait=True)
        with out_colorbar:
            clear_output(wait=True)

        try:
            # Colours
            vals = gdf[attribute].to_numpy(dtype=float)
            valid_mask = ~np.isnan(vals)
            vmin = float(np.nanpercentile(vals, pct_lo))
            vmax = float(np.nanpercentile(vals, pct_hi))
            if vmin == vmax:
                vmax = vmin + 1.0

            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap(cmap_name)

            rgba = np.zeros((len(vals), 4), dtype=np.uint8)
            if valid_mask.any():
                rgba[valid_mask] = (cmap(norm(vals[valid_mask])) * 255).astype(np.uint8)
            rgba[~valid_mask]   = [128, 128, 128, 180]
            rgba[valid_mask, 3] = 200

            # Tooltip columns
            tcols        = selected_cols or [c for c in gdf.columns if c != "geometry"]
            cols_to_pass = list(set(tcols + ["geometry"]))

            # --- Observation layer (auto-detected) --------------------------
            if is_polygon:
                observations = SolidPolygonLayer.from_geopandas(
                    gdf[cols_to_pass],
                    auto_downcast=False,
                    get_fill_color=rgba,
                    pickable=True,
                )
            else:
                radii = gdf[radius_attr].to_numpy(dtype=float)
                radii = np.where(np.isnan(radii), 200.0, radii)
                observations = lonboard.ScatterplotLayer.from_geopandas(
                    gdf[cols_to_pass],
                    auto_downcast=False,
                    get_fill_color=rgba,
                    get_radius=radii,
                    radius_min_pixels=2,
                    radius_max_pixels=50,
                    pickable=True,
                )

            layers = [observations]

            # --- Flight track -----------------------------------------------
            if gdf_track is not None:
                try:
                    track_layer = lonboard.PathLayer.from_geopandas(
                        gdf_track,
                        get_color=[255, 255, 255, 255],
                        get_width=2,
                        width_min_pixels=1,
                        width_max_pixels=3,
                        get_dash_array=[6, 4],
                        dash_justified=True,
                        extensions=[PathStyleExtension(dash=True)],
                        pickable=False,
                    )
                    layers.append(track_layer)
                except Exception as e:
                    print(f"Warning: could not add track layer — {e}")

            # --- Aircraft position points -----------------------------------
            if gdf_points is not None:
                try:
                    point_cols  = [c for c in gdf_points.columns if c != "geometry"]
                    cols_pts    = point_cols + ["geometry"]
                    cross_layer = lonboard.ScatterplotLayer.from_geopandas(
                        gdf_points[cols_pts],
                        auto_downcast=False,
                        get_fill_color=[255, 255, 255, 245],
                        get_radius=20,
                        radius_min_pixels=2,
                        radius_max_pixels=3,
                        stroked=True,
                        filled=True,
                        pickable=True,
                    )
                    layers.append(cross_layer)
                except Exception as e:
                    print(f"Warning: could not add points layer — {e}")

            # --- Map --------------------------------------------------------
            m = lonboard.Map(
                layers=layers,
                basemap=MaplibreBasemap(style=SATELLITE_STYLE),
            )

            # --- Colorbar ---------------------------------------------------
            with out_colorbar:
                fig, ax = plt.subplots(figsize=(5, 0.8))
                fig.patch.set_alpha(0)
                cb = fig.colorbar(
                    cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=ax, orientation="horizontal",
                )
                cb.set_label(attribute, fontsize=10)
                nan_patch = mpatches.Patch(
                    facecolor="#808080", edgecolor="none", label="NaN"
                )
                ax.legend(
                    handles=[nan_patch],
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.4),
                    frameon=False, fontsize=9,
                )
                plt.subplots_adjust(bottom=0.4)
                plt.show()

            with out_map:
                display(m)

        except Exception as e:
            import traceback
            with out_map:
                clear_output(wait=True)
                print(f"ERROR: {e}")
                traceback.print_exc()

    btn_render.on_click(build_map)

    # --- Layout -------------------------------------------------------------
    geom_label = "Polygon" if is_polygon else "Point"
    controls = widgets.VBox([
        widgets.HTML(f"<b>Geometry type detected: {geom_label}</b>"),
        widgets.HBox([w_attribute, w_cmap]),
        w_percentile,
        btn_render,
    ])

    display(widgets.VBox([
        controls,
        out_colorbar,
        out_map,
    ]))

    # Initial render
    build_map(None)


def map_rongowai(
    gdf,
    attribute="surface_reflectivity_peak",
    radius_attr="fresnel_minor",
    cmap_name="plasma",
    tooltip_cols=None,
    gdf_points=None,          # optional GeoDataFrame of cross markers
    gdf_track=None,           # optional GeoDataFrame of flight track polyline
):
    """
    Render Rongowai point data on a satellite basemap using lonboard.

    Parameters
    ----------
    gdf          : GeoDataFrame with Point geometry (Rongowai observations)
    attribute    : column to colour points by (NaN → grey)
    radius_attr  : column to use as point radius in metres
    cmap_name    : matplotlib colormap name
    tooltip_cols : columns to show in click tooltip (None = all)
    gdf_points   : optional GeoDataFrame of Point geometry, shown as crosses
    gdf_track    : optional GeoDataFrame of LineString geometry, shown as dotted line
    """

    # --- colour by attribute ------------------------------------------------
    vals = gdf[attribute].to_numpy(dtype=float)
    valid_mask = ~np.isnan(vals)
    vmin, vmax = float(np.nanpercentile(vals, 5)), float(np.nanpercentile(vals, 95))

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    rgba = np.zeros((len(vals), 4), dtype=np.uint8)
    if valid_mask.any():
        rgba[valid_mask] = (cmap(norm(vals[valid_mask])) * 255).astype(np.uint8)
    rgba[~valid_mask]   = [128, 128, 128, 180]
    rgba[valid_mask, 3] = 200

    # --- radius from fresnel_minor ------------------------------------------
    radii = gdf[radius_attr].to_numpy(dtype=float)
    radii = np.where(np.isnan(radii), 200.0, radii)

    # --- tooltip columns ----------------------------------------------------
    if tooltip_cols is None:
        tooltip_cols = [c for c in gdf.columns if c != "geometry"]
    cols_to_pass = list(set(tooltip_cols + ["geometry"]))

    display_cols     = [c for c in tooltip_cols if c != "geometry"]
    tooltip_template = "\n".join(f"{col}: {{{col}}}" for col in display_cols)

    # --- Rongowai observation points ----------------------------------------
    observations = lonboard.ScatterplotLayer.from_geopandas(
        gdf[cols_to_pass],
        get_fill_color=rgba,
        get_radius=radii,
        radius_min_pixels=2,
        radius_max_pixels=50,
        pickable=True,
    )
    layers = [observations]

    # --- optional flight track polyline (dotted) ----------------------------
    if gdf_track is not None:
        try:
            track_layer = lonboard.PathLayer.from_geopandas(
                gdf_track,
                get_color=[255, 255, 255, 255],
                get_width=2,
                width_min_pixels=1,
                width_max_pixels=3,
                get_dash_array=[6, 4],        # note: get_dash_array not dash_array
                dash_justified=True,
                extensions=[PathStyleExtension(dash=True)],   # dash=True is required
                pickable=False,
            )
            layers.append(track_layer)
        except Exception as e:
            print(f"Warning: could not add track layer — {e}")

    # --- optional cross markers with tooltip --------------------------------
    if gdf_points is not None:
        try:
            # lonboard has no native cross/X marker; use a small IconLayer
            # or approximate with two overlapping thin ScatterplotLayers.
            # Most practical: use ScatterplotLayer with a + shape via
            # stroked=True and filled=False to suggest a cross.
            point_cols    = [c for c in gdf_points.columns if c != "geometry"]
            point_tooltip = "\n".join(f"{col}: {{{col}}}" for col in point_cols)
            cols_pts      = point_cols + ["geometry"]

            cross_layer = lonboard.ScatterplotLayer.from_geopandas(
                gdf_points[cols_pts],
                get_fill_color=[255, 255, 255, 245],  
                get_radius=20,
                radius_min_pixels=2,
                radius_max_pixels=3,
                stroked=True,
                filled=True,
                pickable=True,
            )
            layers.append(cross_layer)

            # override map tooltip to also cover the cross layer — lonboard
            # uses a single tooltip template across all pickable layers,
            # so we merge both templates
            merged_tooltip = tooltip_template + "\n---\n" + point_tooltip

        except Exception as e:
            print(f"Warning: could not add points layer — {e}")
            merged_tooltip = tooltip_template
    else:
        merged_tooltip = tooltip_template

    # --- map ----------------------------------------------------------------
    m = lonboard.Map(
        layers=[observations, cross_layer, track_layer],  # no BitmapTileLayer
        basemap=MaplibreBasemap(style=SATELLITE_STYLE),
        #tooltip=merged_tooltip,
    )


    return m, vmin, vmax

def plot_colorbar(
    attribute="surface_reflectivity_peak",
    vmin=0.0,
    vmax=1.0,
    cmap_name="plasma",
    nan_colour="#808080",
    figsize=(5, 0.6),
):
    """Render a standalone horizontal colourbar beneath the map."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cb = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_name)),
        cax=ax,
        orientation="horizontal",
    )
    cb.set_label(attribute, fontsize=10)

    nan_patch = Patch(facecolor=nan_colour, edgecolor="none", label="NaN")
    ax.legend(
        handles=[nan_patch],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.4),
        frameon=False,
        fontsize=9,
    )

    plt.tight_layout()
    plt.show()

def showDataDict(df_dict):
    """
    Display an interactive data dictionary viewer with filters and variable detail panel.

    Parameters
    ----------
    df_dict : pd.DataFrame
        Data dictionary loaded from the L1 data dictionary Excel file.
    """
    import pandas as pd
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    from itables import show
    import itables.options as itables_options

    itables_options.warn_on_undocumented_option = False

    # Replace \n with <br> in Comment column for HTML rendering
    df_dict = df_dict.copy()
    df_dict["Comment"] = df_dict["Comment"].str.replace("\n", "<br>", regex=False)

    COMMENT_COL = df_dict.columns.get_loc("Comment")

    # --- Filter widgets -----------------------------------------------------
    data_type_options = ["All"] + sorted(df_dict["Data_type"].unique().tolist())
    dimension_options = ["All"] + sorted(df_dict["Dimensions"].unique().tolist())

    w_datatype = widgets.Dropdown(
        options=data_type_options,
        value="All",
        description="Data type:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="350px"),
    )
    w_dimensions = widgets.Dropdown(
        options=dimension_options,
        value="All",
        description="Dimensions:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="350px"),
    )
    w_variable = widgets.Dropdown(
        options=df_dict["Name"].tolist(),
        description="Variable:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="350px"),
    )
    w_search = widgets.Text(
        placeholder="Search variable names...",
        description="Search:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="350px"),
    )

    out_table  = widgets.Output()
    out_detail = widgets.Output()

    # --- Helper functions ---------------------------------------------------
    def get_filtered_df():
        mask = pd.Series([True] * len(df_dict), index=df_dict.index)
        if w_datatype.value != "All":
            mask &= df_dict["Data_type"] == w_datatype.value
        if w_dimensions.value != "All":
            mask &= df_dict["Dimensions"] == w_dimensions.value
        if w_search.value.strip():
            mask &= df_dict["Name"].str.contains(w_search.value.strip(), case=False, na=False)
        return df_dict[mask].reset_index(drop=True)

    def update_variable_dropdown(filtered_df):
        names = filtered_df["Name"].tolist()
        current = w_variable.value
        w_variable.options = names if names else ["<no results>"]
        if current in names:
            w_variable.value = current
        elif names:
            w_variable.value = names[0]

    def show_detail(name):
        with out_detail:
            clear_output(wait=True)
            if name == "<no results>":
                return
            row = df_dict[df_dict["Name"] == name]
            if row.empty:
                return
            detail = row.T.reset_index()
            detail.columns = ["Field", "Value"]
            show(
                detail,
                paging=False,
                searching=False,
                info=False,
                ordering=False,
                escape=False,
                classes="display compact",
                columnDefs=[{"width": "160px", "targets": 0}],
                caption=f"Variable: {name}",
            )

    def show_table():
        with out_table:
            clear_output(wait=True)
            filtered = get_filtered_df()
            show(
                filtered,
                caption=f"{len(filtered)} variables",
                classes="display compact",
                paging=True,
                pageLength=15,
                escape=False,
                columnDefs=[
                    {"width": "140px", "targets": 0},
                    {"width": "180px", "targets": 1},
                    {"width": "120px", "targets": 2},
                    {"width": "80px",  "targets": 3},
                    {"width": "160px", "targets": 4},
                    {"width": "300px", "targets": COMMENT_COL},
                    {"width": "180px", "targets": 6},
                ],
            )

    # --- Event handlers -----------------------------------------------------
    def on_filter_change(change):
        filtered = get_filtered_df()
        update_variable_dropdown(filtered)
        show_table()

    def on_variable_change(change):
        show_detail(w_variable.value)

    w_datatype.observe(on_filter_change, names="value")
    w_dimensions.observe(on_filter_change, names="value")
    w_search.observe(on_filter_change, names="value")
    w_variable.observe(on_variable_change, names="value")

    # --- Layout and initial render ------------------------------------------
    filters = widgets.HBox([w_datatype, w_dimensions, w_search])
    display(
        widgets.VBox([
            widgets.HTML("<h3>Rongowai L1 Data Dictionary</h3>"),
            filters,
            widgets.HTML("<b>Variable detail:</b>"),
            w_variable,
            out_detail,
            widgets.HTML("<b>Filtered variable list:</b>"),
            out_table,
        ])
    )

    show_table()
    show_detail(w_variable.value)

def showGeoDataFrame(gdf, caption="", df_dict=None, maxBytes=10240):
    """
    Display an interactive GeoDataFrame viewer with column selection,
    value filtering, and summary statistics.

    Parameters
    ----------
    gdf       : GeoDataFrame to display
    caption   : optional string caption for the table
    df_dict   : optional data dictionary DataFrame for column long names
    """
    import numpy as np
    import pandas as pd
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    from itables import show
    import itables.options as itables_options
    import warnings

    itables_options.warn_on_undocumented_option = False

    df = gdf.drop(columns="geometry").copy()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()

    # --- Column selector ----------------------------------------------------
    col_selector = widgets.SelectMultiple(
        options=all_cols,
        value=all_cols,
        description="Columns:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="350px", height="160px"),
    )
    btn_all  = widgets.Button(description="Select all",  layout=widgets.Layout(width="110px"))
    btn_none = widgets.Button(description="Clear all",   layout=widgets.Layout(width="110px"))

    # --- Numeric filter controls -------------------------------------------
    filter_widgets = {}

    for col in num_cols:
        col_min = float(np.nanmin(df[col]))
        col_max = float(np.nanmax(df[col]))
        if col_min == col_max:
            continue

        span = col_max - col_min
        step = 10 ** (np.floor(np.log10(span)) - 2) if span > 0 else 0.01
        step = round(step, 10)

        slider = widgets.FloatRangeSlider(
            value=[col_min, col_max],
            min=col_min, max=col_max, step=step,
            continuous_update=False,
            layout=widgets.Layout(width="400px"),
            readout=True,          # show values on slider itself
        )
        txt_min = widgets.FloatText(
            value=col_min,
            description="Min:",
            style={"description_width": "30px"},
            layout=widgets.Layout(width="160px"),
        )
        txt_max = widgets.FloatText(
            value=col_max,
            description="Max:",
            style={"description_width": "30px"},
            layout=widgets.Layout(width="160px"),
        )

        def make_sync(s, tmin, tmax, cmin, cmax):
            def on_slider(change):
                tmin.value = change["new"][0]
                tmax.value = change["new"][1]
                refresh(None)
            def on_text(change):
                lo = max(cmin, min(tmin.value, tmax.value))
                hi = min(cmax, max(tmin.value, tmax.value))
                s.value = [lo, hi]
                refresh(None)
            s.observe(on_slider, names="value")
            tmin.observe(on_text, names="value")
            tmax.observe(on_text, names="value")

        make_sync(slider, txt_min, txt_max, col_min, col_max)
        filter_widgets[col] = {"slider": slider, "txt_min": txt_min, "txt_max": txt_max}

    def build_filter_accordion():
        """Build accordion with one panel per numeric column that is currently selected."""
        selected_cols = list(col_selector.value)
        cols_with_filters = [c for c in selected_cols if c in filter_widgets]
        if not cols_with_filters:
            return widgets.HTML("<i>No numeric columns selected.</i>")

        children = []
        titles   = []
        for col in cols_with_filters:
            w = filter_widgets[col]
            panel = widgets.VBox([
                w["slider"],
                widgets.HBox([w["txt_min"], w["txt_max"]]),
            ], layout=widgets.Layout(padding="8px"))
            children.append(panel)
            titles.append(col)

        acc = widgets.Accordion(children=children)
        for i, t in enumerate(titles):
            acc.set_title(i, t)
        acc.selected_index = None   # all collapsed by default
        return acc

    btn_toggle_filters = widgets.ToggleButton(
        value=False,
        description="Show value filters",
        layout=widgets.Layout(width="160px"),
    )
    filters_container = widgets.VBox([])

    def on_toggle_filters(change):
        if change["new"]:
            filters_container.children = [build_filter_accordion()]
            btn_toggle_filters.description = "Hide value filters"
        else:
            filters_container.children = []
            btn_toggle_filters.description = "Show value filters"

    btn_toggle_filters.observe(on_toggle_filters, names="value")

    # --- Output areas -------------------------------------------------------
    out_stats = widgets.Output()
    out_table = widgets.Output()

    # --- Core logic ---------------------------------------------------------
    def get_filtered_df():
        mask = pd.Series(True, index=df.index)
        for col, w in filter_widgets.items():
            lo, hi = w["slider"].value
            mask &= df[col].between(lo, hi) | df[col].isna()
        selected = list(col_selector.value) or all_cols
        return df.loc[mask, selected]

    def show_stats(filtered):
        with out_stats:
            clear_output(wait=True)
            # Cast to float to avoid integer overflow in std/mean
            numeric = filtered.select_dtypes(include="number").astype(float)
            if numeric.empty:
                return
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                stats = numeric.agg(["min", "mean", "median", "max", "std"]).T
            stats.index.name = "variable"
            stats = stats.reset_index()
            if df_dict is not None and "Name" in df_dict.columns and "Long_name" in df_dict.columns:
                name_map = df_dict.set_index("Name")["Long_name"].to_dict()
                stats.insert(1, "long_name", stats["variable"].map(name_map).fillna(""))
            for c in ["min", "mean", "median", "max", "std"]:
                stats[c] = stats[c].apply(lambda x: f"{x:.4g}")
            show(
                stats,
                caption=f"Summary statistics — {len(filtered)} rows",
                classes="display compact",
                paging=False,
                searching=False,
                maxBytes=maxBytes,
            )

    def show_table(filtered):
        with out_table:
            clear_output(wait=True)
            show(
                filtered,
                caption=f"{caption}  ({len(filtered)} rows)",
                classes="display compact",
                paging=True,
                pageLength=15,
                layout={"top": "searchBuilder"},
                maxBytes=maxBytes,        # limit data size to specified amount
            )

    def refresh(_):
        filtered = get_filtered_df()
        show_stats(filtered)
        show_table(filtered)
        # Rebuild accordion if filters are visible so it reflects column selection
        if btn_toggle_filters.value:
            filters_container.children = [build_filter_accordion()]

    # --- Button handlers ----------------------------------------------------
    btn_all.on_click(lambda _: setattr(col_selector, "value", all_cols))
    btn_none.on_click(lambda _: setattr(col_selector, "value", []))
    col_selector.observe(refresh, names="value")

    # --- Layout -------------------------------------------------------------
    col_controls = widgets.VBox([
        widgets.HTML("<b>Select columns:</b>"),
        col_selector,
        widgets.HBox([btn_all, btn_none]),
    ])

    display(widgets.VBox([
        widgets.HTML(f"<h3>{caption}</h3>") if caption else widgets.HTML(""),
        widgets.HBox([
            col_controls,
            widgets.VBox([
                btn_toggle_filters,
                filters_container,
            ], layout=widgets.Layout(margin="0 0 0 24px")),
        ]),
        widgets.HTML("<b>Summary statistics:</b>"),
        out_stats,
        widgets.HTML("<b>Data:</b>"),
        out_table,
    ]))

    refresh(None)

def l1_to_gpkg_single(l1_file, gpkg_file, verbose=True):
    ds = nc.Dataset(l1_file)
    data_dict = {}
    data_dict['data'] = rongowai_readdatafile(ds, vars=None, verbose=verbose)
    data_dict['spatial'] = rongowai_gather(data_dict['data'], vars=None)
    data_dict['flightinfo'] = rongowai_flightinfo(ds)
    data_dict['ac'] = rongowai_ac_spatial(data_dict['data'], vars=None)
    data_dict['flightvector'] = rongowai_flightvector(data_dict['ac'], data_dict['flightinfo'])

    data_dict['spatial'].to_file(gpkg_file, layer='spatial', driver="GPKG")
    data_dict['ac'].to_file(gpkg_file, layer='ac', driver="GPKG")
    data_dict['flightvector'].to_file(gpkg_file, layer='flightvector', driver="GPKG")

    return gpkg_file

def parse_rongowai_files(file_list):
    """Parse Rongowai L1 filenames into a structured DataFrame."""
    records = []
    pattern = r"(\d{8})-(\d{6})_([A-Z]{4})-([A-Z]{4})_(L\d+)\.nc$"
    
    for fname in file_list:
        m = re.search(pattern, fname)
        if m:
            date_str, time_str, dep, dest, level = m.groups()
            dt = pd.to_datetime(f"{date_str} {time_str}", format="%Y%m%d %H%M%S", utc=True)
            records.append({
                "filename":     fname,
                "datetime_utc": dt,
                "date_utc":     dt.date(),
                "time_utc":     dt.strftime("%H:%M:%S"),
                "departure":    dep,
                "destination":  dest,
                "level":        level,
            })
        else:
            print(f"Warning: could not parse '{fname}'")
    
    df = pd.DataFrame(records)
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    return df

def rongowai_flightinfo(ds):
    ncatt = ds.ncattrs()
    ncatt_dict = {}
    for att in ncatt:
        ncatt_dict[att] = ds.getncattr(att)
    ncatt_dict = pd.DataFrame((ncatt_dict), index=[0])
    ncatt_dict['delay_resolution'] = ds.variables['delay_resolution'][:]
    ncatt_dict['dopp_resolution'] = ds.variables['dopp_resolution'][:]
    ncatt_dict['coh_int'] = ds.variables['coh_int'][:]
    return ncatt_dict

def rongowai_readdatafile(ds, vars=None, verbose=True):
    data = {}
    if vars is None:
        vars = [var for var in ds.variables.keys() if len(ds.variables[var].shape) >= 1]
    # if verbose:
    #     print("\n -------------------------> variables= ",vars)

    # These are always loaded for positioning and indices
    vars = [var for var in vars if var not in ["sp_lat", "sp_lon", "sp_alt", "ddm", "ac_lat", "ac_lon", "ac_alt", "sample"]]

    # Load positioning variables and indices
    data["sp_lon"] = ds.variables["sp_lon"][:]
    data["sp_lat"] = ds.variables["sp_lat"][:]
    data["sp_alt"] = ds.variables["sp_alt"][:]
    data["ddm"] = ds.variables["ddm"][:]
    data["sample"] = ds.variables["sample"][:]
    data["ac_lat"] = ds.variables["ac_lat"][:]
    data["ac_lon"] = ds.variables["ac_lon"][:]
    data["ac_alt"] = ds.variables["ac_alt"][:]

    # Get other required variables
    varget = vars
    for v in varget:
        try:
            data[v] = ds.variables[v][:]
        except KeyError:
            print("Variable \"{}\" not found in input file: {}".format(v, ds))
            vars = [var for var in vars if var != v]
    return data

def rongowai_gather(ncdata, vars=None):
    if vars is None:
        vars = ncdata.keys()

    ddm_channel = np.tile(ncdata["ddm"], len(ncdata["sp_lon"]))
    sample = np.repeat(ncdata["sample"], len(ncdata["ddm"]))
    obs_id = (sample * len(ncdata['ddm']))+ddm_channel

    sp_lat = ncdata['sp_lat']
    sp_lon = ncdata['sp_lon']
    sp_alt = ncdata['sp_alt']

    df = pd.DataFrame({
        "obs_id": obs_id,
        "sample": sample,
        "sp_lat": sp_lat.flatten(),
        "sp_lon": sp_lon.flatten(),
        "sp_alt": sp_alt.flatten(),
        "ddm": ddm_channel,
    })

    # process selected variables: extend over all selected variables with correct dimensions
    varget = []
    for v in vars:
        if np.shape(ncdata[v]) == np.shape(ncdata["sp_lon"]):
            varget.append(v)

    varget = [v for v in varget if v not in ("sp_lat", "sp_lon", "sp_alt", "ddm", "sample", "obs_id")]
    for v in varget:
        df[v] = ncdata[v].flatten()


    # filter out rows where sp_lat or sp_lon is NaN
    df = df[~(df['sp_lat'].isna() | df['sp_lon'].isna())]

    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.sp_lon, df.sp_lat, df.sp_alt), crs='EPSG:4326')
    return df

def rongowai_ac_spatial(ncdata, vars=None, verbose=True):
    if vars is None:
        vars = ncdata.keys()

    ac_lat = ncdata['ac_lat']
    ac_lon = ncdata['ac_lon']
    ac_alt = ncdata['ac_alt']
    sample = ncdata['sample']

    ncac = pd.DataFrame({'ac_lat': ac_lat, 'ac_lon': ac_lon, 'ac_alt': ac_alt, 'sample': sample})
    ncac = ncac[['sample', 'ac_lat', 'ac_lon', 'ac_alt']]
    ncac = gpd.GeoDataFrame(ncac, geometry=gpd.points_from_xy(ncac.ac_lon, ncac.ac_lat, ncac.ac_alt), crs='EPSG:4326')

    varget = []
    for v in vars:
        if isinstance(ncdata[v], np.ndarray) and ncdata[v].shape == sample.shape:
            varget.append(v)

    varget = [v for v in varget if v not in ['ac_lat', 'ac_lon', 'ac_alt', 'sample']]

    namesncac = list(ncac.columns)

    for v in varget:
        ncac[v] = ncdata[v]

    namesncac = [namesncac[0], namesncac[1], namesncac[2], namesncac[3]] + varget + [namesncac[4]]
    finalncac = ncac[namesncac]
    return finalncac

def rongowai_flightvector(ncac, flightinfo=None, ncin_file=None, verbose=True):
    # create a GeoDataFrame with a Point geometry column
    gdf_points = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(ncac.ac_lon, ncac.ac_lat, ncac.ac_alt),
    crs='EPSG:4326')

    # extract the point coordinates as a list
    coords = list(gdf_points.geometry.values)

    # create a LineString from the point coordinates
    lines = LineString(coords)

    flightvector = flightinfo
    flightvector['geometry'] = lines

    flightvector = gpd.GeoDataFrame(flightvector, geometry= 'geometry', crs='EPSG:4326')
    return flightvector

def rongowai_quicklook_segment(pts, crs=None):
    if pts.shape[0] <= 1:
        return None
    coords = list(pts.geometry.apply(lambda p: (p.x, p.y)))
    sfc = gpd.GeoSeries([LineString(coords)], crs=crs)
    ln = gpd.GeoDataFrame({'geometry': sfc})
    ln['sample_start'] = pts['sample'].min()
    ln['sample_end'] = pts['sample'].max()
    return ln

def rongowai_quicklook(spatialpts):
    allsegs = None
    for i in range(20):
        # ordered by sample id
        ddmpts = spatialpts.loc[spatialpts['ddm'] == i].sort_values(by=['sample'])

        if len(ddmpts) > 0:
            coord1 = ddmpts[['sp_lat', 'sp_lon']].values
            coord2 = ddmpts[['sp_lat', 'sp_lon']].shift(-1).values
            coord2[-1:] =  coord1[-1:]
            # calculate distance
            ddmpts['dist'] = [distance.distance(coord1, coord2).m for coord1, coord2 in zip(coord1, coord2)]
            dist_gt_500 = ddmpts['dist'] > 500
            indices = np.where(dist_gt_500)[0] + 1
            # split into segments
            segments =  np.split(ddmpts,indices)
            segments = [rongowai_quicklook_segment(seg, spatialpts.crs) for seg in segments if len(seg) > 0]
            segments = gpd.GeoDataFrame(pd.concat(segments, ignore_index=True), crs=spatialpts.crs)
            segments['ddm'] = i
            segments = segments[['ddm', 'geometry', 'sample_start', 'sample_end']]
            if allsegs is None:
                allsegs = segments
            else:
                allsegs = pd.concat([allsegs, segments])
    return allsegs

def ellipse_stretch(center_x, center_y, majoraxis, minoraxis, rotate_angle=0,
                    velocity_x=0, velocity_y=0,
                    center_z=None, resn=30,
                    costheta=None, sintheta=None):

    if costheta is None:
        costheta = np.cos(np.linspace(0, 2 * np.pi, num=resn))
    if sintheta is None:
        sintheta = np.sin(np.linspace(0, 2 * np.pi, num=resn))

    # create ellipse with major/ minor axes, centered on x/ y
    x = center_x - minoraxis * costheta
    y = center_y - majoraxis * sintheta

    # rotate if angle != 0
    if rotate_angle != 0:
        co = cos(-rotate_angle * pi / 180)
        si = sin(-rotate_angle * pi / 180)

        adjx = x - center_x
        adjy = y - center_y

        x = (co * adjx - si * adjy) + center_x
        y = (si * adjx + co * adjy) + center_y

    # save to data frame
    xy = pd.DataFrame({"x": x, "y": y})
    # account for smeering along track
    if velocity_x != 0 or velocity_y != 0:

        # shift zone forward/ back - duplicating points
        # xy = xy.append(pd.DataFrame({"x": x + (velocity_x/2), "y": y + (velocity_y/2)}))
        # xy = xy.append(pd.DataFrame({"x": x - (velocity_x/2), "y": y - (velocity_y/2)}))
        ##################
        xy1 = pd.DataFrame({"x": x + (velocity_x/2), "y": y + (velocity_y/2)})
        xy2 = pd.DataFrame({"x": x - (velocity_x/2), "y": y - (velocity_y/2)})
        # print(xy1.head())
        combined_xy = pd.concat([xy1,xy2])
        xy = pd.concat([xy,combined_xy])
        ###########3
        # Convert DataFrame to numpy array
        points = xy.to_numpy()
        # Get the convex hull of the points
        hull = sp.ConvexHull(points)

        # Extract the coordinates of the hull vertices
        hull_verts = points[hull.vertices, :]

        # Create a new DataFrame to store the hull vertices
        xy = pd.DataFrame({"x": hull_verts[:,0], "y": hull_verts[:,1]})

    # add z if needed
    if center_z is not None:
        xy["z"] = center_z

    return xy


def pts_to_polygon(pts, xyz=["x","y","z"], crs=4326):
    # print(type(pts))
    # print(len(pts))
    # print(pts)

    if len(xyz) == 2:
        xydata = [(row[xyz[0]], row[xyz[1]]) for _, row in pts.iterrows()]
    elif len(xyz) == 3:
        xydata = [(row[xyz[0]], row[xyz[1]], row[xyz[2]]) for _, row in pts.iterrows()]
    else:
        raise ValueError("xyz should be of length 2 (for 2D data) or 3 (for adding a Z dimension)")
    polygon = Polygon(xydata)
    return polygon


def rongowai_fresnelzones(pts, major='fresnel_major', minor='fresnel_minor', angle='fresnel_orientation',
                          resn=30, verbose=True,
                          velocity_x='sp_vel_x', velocity_y='sp_vel_y',
                          calcstretch=False,
                          parallel=False, cores=None):

    """Calculate Fresnel zones for a set of points"""
    if cores is None:
        cores = cpu_count()
    if verbose:
        if calcstretch:
            print(f" Calculating Fresnel zone integrations - mode: parallel (CPU cores={cores})")
        else:
            print(f" Calculating Fresnel zones - mode: parallel (CPU cores={cores})")

    if not isinstance(pts, gpd.GeoDataFrame):
        raise TypeError("pts must be a geopandas GeoDataFrame")

    if not 'Point' in pts['geometry'].geom_type.values:

        raise ValueError("pts must contain only point geometries")

    major = pts[major] if isinstance(major, str) else major
    minor = pts[minor] if isinstance(minor, str) else minor
    angle = pts[angle] if isinstance(angle, str) else angle

    # if not all(isinstance(val, (int, float)) for val in [major, minor, angle]) or \
    if len(major) != len(minor) or len(minor) != len(angle) or len(angle) != len(pts):
        raise ValueError("major, minor, and angle must be numeric vectors of length nrow(pts), or the name of the variables in pts.")


    if calcstretch:
        velocity_x = pts[velocity_x].values if isinstance(velocity_x, str) else velocity_x
        velocity_y = pts[velocity_y].values if isinstance(velocity_y, str) else velocity_y

        if not (len(velocity_x) == pts.shape[0] and len(velocity_y) == pts.shape[0]):
            raise ValueError("velocity_x and velocity_y must be numeric vectors of length nrow(pts), or the name of the variables in pts.")

    else:
        # To match the length of other components
        velocity_x = np.repeat(0, pts.shape[0])
        velocity_y = np.repeat(0, pts.shape[0])


    major = major/2
    minor = minor/2

    coords = list(pts.geometry.apply(lambda p: (p.x, p.y, p.z)))
    coords = pts.geometry.x,pts.geometry.y,pts.geometry.z

    if np.shape(coords)[0] == 3:
        center_z = coords[:][2]
    else:
        center_z = np.repeat(np.nan, coords.shape[0])

    v = np.vectorize(ellipse_stretch, otypes=[object])

    v = v(center_x=coords[:][0], center_y=coords[:][1], majoraxis=major, minoraxis=minor, center_z=center_z,
          rotate_angle=angle, velocity_x=velocity_x, velocity_y=velocity_y,
          costheta=None, sintheta=None)


    # v = np.vectorize(ellipse_stretch)(center_x=center_x, # lon
    #                               center_y=center_y, # lat
    #                               majoraxis=major,
    #                               minoraxis=minor,
    #                               center_z=center_z,
    #                               rotate_angle=angle,
    #                               velocity_x=velocity_x,
    #                               velocity_y=velocity_y,
    #                               costheta=None,
    #                               sintheta=None)
                                #   costheta=np.cos(np.linspace(0, 2*np.pi, resn)),
                                #   sintheta=np.sin(np.linspace(0, 2*np.pi, resn)))

    # if parallel:
    #     with Pool(cores) as p:
    #         func = partial(pts_to_polygon, xyz=["x", "y", "z"], crs=(pts.crs))

    #         if verbose:
    #             v = p.starmap(func, zip(v, True),int(len(v)/5))
    #         else:
    #             v = p.starmap(func, zip(v, False),int(len(v)/5))

    if parallel:
        with Pool(cores) as p:
            # func = partial(pts_to_polygon, xyz=["x", "y", "z"], crs=(pts.crs))
            v = p.map(pts_to_polygon,v)
            # if verbose:
            #     iterable_true = [True] * len(v)
            #     v = p.starmap(func, zip(v, iterable_true[0]))
            #     # v = p.starmap(func, zip(v, True),int(len(v)/5))
            # else:
            #     iterable_false = [False] * len(v)
            #     v = p.starmap(func, zip(v, iterable_false[0]))
            #     # v = p.starmap(func, zip(v, False),int(len(v)/5))
    else:
        v = [pts_to_polygon(v[i], ["x", "y", "z"], (pts).to_crs(pts.crs)) for i in range(len(v))]


    if parallel:
        p.close()

    return v

def fresnel(nc):

    crs = nc.crs
    # Transform the CRS of the GeoDataFrame
    nc = nc.to_crs(epsg=2193)

    # Filter out rows with NaN values in the specified columns
    nc = nc[~(nc['fresnel_major'].isna() | nc['fresnel_minor'].isna() | nc['fresnel_orientation'].isna())]
    # nc = nc[0:25]

    # Apply the fresnel zones function to the GeoDataFrame
    poly_list =  rongowai_fresnelzones(nc, parallel=PARALLEL)

    nc['poly'] = poly_list
    nc['geometry'] = nc['poly']
    nc = nc.drop('poly', axis = 1)

    # Transform the GeoDataFrame back to the original CRS
    nc = nc.to_crs(crs)
    return nc

def fresnel_integration(nc):

    crs = nc.crs
    # Transform the CRS of the GeoDataFrame
    nc = nc.to_crs(epsg=2193)

    # Filter out rows with NaN values in the specified columns
    nc = nc[~(nc['fresnel_major'].isna() | nc['fresnel_minor'].isna() | nc['fresnel_orientation'].isna())]
    # nc = nc[0:25]

    # Apply the fresnel zones function to the GeoDataFrame
    poly_list =  rongowai_fresnelzones(nc, parallel=PARALLEL, calcstretch=True)

    nc['poly'] = poly_list
    nc['geometry'] = nc['poly']
    nc = nc.drop('poly', axis = 1)

    # Transform the GeoDataFrame back to the original CRS
    nc = nc.to_crs(crs)
    return nc

def insert_into_ncs(filename, status):
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect('ncsdb.sqlite')
        cursor = conn.cursor()

        # Insert data into the 'ncs' table
        cursor.execute("INSERT INTO nc (filename, status) VALUES (?, ?)", (filename, status))

        # Commit the changes and close the connection
        conn.commit()
        conn.close()

        print(f"Inserted '{filename}' with status '{status}' into the 'nc' table.")
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")

def check_filename_exists(filename_to_check):
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect('ncsdb.sqlite')
        cursor = conn.cursor()

        # Execute a SELECT query to check if the filename exists
        cursor.execute("SELECT COUNT(*) FROM nc WHERE filename=?", (filename_to_check,))
        result = cursor.fetchone()

        # Check the result
        if result[0] > 0:
            return True
        else:
            return False

        # Close the connection
        conn.close()
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")

if __name__ == '__main__':
    # starts here
    verbose = True
    parallel = True
    calcstretch=True
    # specify the directory path to find input nc files
    ncpath ='O:/NZ/L1/nc'
    # specify the directory path to write output gpkg files
    gpkgpath ='O:/NZ/L1/gpkg'

    # check if the input directory exists
    if not os.path.exists(ncpath):
        # raise exception
        raise FileNotFoundError(f"Directory not found: {ncpath}")

    # check if the output directory exists
    if not os.path.exists(gpkgpath):
        # create the directory if it doesn't exist
        os.makedirs(gpkgpath)

    # get all filenames in the directory
    filenames = os.listdir(ncpath)

    # filter for files with .nc extension
    nc_files = filter(lambda filename: filename.endswith('.nc'), filenames)

    # if(verbose):
    #     print(list(nc_files))
    #     print(len(list(nc_files))," nc files found. Starting to precess...........")
    for i, ncname in enumerate(nc_files):
        # process each file
        try:
            print(f"File {i}: {ncname}")
            inputfile = os.path.join(ncpath,ncname)
            outfile = ncname+".gpkg"
            gpkg = os.path.join(gpkgpath,outfile)
            status = "NA"
            # skip the file if processed already (in case output file already exists)
            # otherwise process the file
            if not os.path.exists(gpkg):
            # if not check_filename_exists(ncname):
                # print("**** ===> ",nc.Dataset(inputfile))
                ds = nc.Dataset(inputfile)
                data_dict = {}

                data_dict['ncname'] = ncname
                data_dict['ncpath'] = ncpath
                data_dict['ncpath'] = os.path.join(ncpath,ncname)

                data_dict['varlist'] = list(ds.variables.keys())
                datadimnames = ['sample','ddm','delay','doppler']
                datadimvalues = list(ds['raw_counts'].shape)
                data_dict['datadim'] = dict(zip(datadimnames,datadimvalues))
                data_dict['flightinfo'] = rongowai_flightinfo(ds)
                data_dict['data'] = rongowai_readdatafile(ds, vars=None, verbose=verbose)
                data_dict['spatial'] = rongowai_gather(data_dict['data'], vars=None)
                # print(data_dict['spatial'].head())
                data_dict['ac'] =rongowai_ac_spatial(data_dict['data'], vars=None)
                data_dict['flightvector'] =rongowai_flightvector(data_dict['ac'],data_dict['flightinfo'])
                data_dict['quicklook'] = rongowai_quicklook(data_dict['spatial'])
                # data_dict['fresnel'] = fresnel(data_dict['spatial'])
                # data_dict['fresnel_integration'] = fresnel_integration(data_dict['spatial'])

                # write output - with different layers
                data_dict['ac'].to_file(gpkg, layer='ss-ac', driver="GPKG")
                data_dict['flightvector'].to_file(gpkg, layer='ss-flightvector', driver="GPKG")
                data_dict['spatial'].to_file(gpkg, layer='ss-spatial', driver="GPKG")
                data_dict['quicklook'].to_file(gpkg, layer='ss-quicklook', driver="GPKG")
                # data_dict['fresnel'].to_file(gpkg, layer='ss-fresnel', driver="GPKG")
                # data_dict['fresnel_integration'].to_file(gpkg, layer='ss-fresnel_integration', driver="GPKG")
                # status = "processed"
            else:
                print(f"SKIPPing File {i}: {ncname}")
        except Exception as e:
            status = str(e)
            print(status)
        # finally:
        #     insert_into_ncs(ncname, status)
