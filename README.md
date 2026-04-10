# Exploring GNSS-R Data from Rongowai

*Prof. Matthew Wilson and Xander Cai*  
*Geospatial Research Institute Toi Hangarau, University of Canterbury, Christchurch, New Zealand*  
*Version 1.0.0 — April 2026*

---

## Overview

This repository provides an interactive Jupyter notebook for exploring Level 1 (L1) GNSS-R data from the **Rongowai** mission. Rongowai is a GNSS Reflectometry (GNSS-R) sensor onboard an Air New Zealand Q300 aircraft ([ZK-NHA](https://www.flightradar24.com/data/aircraft/zk-nfa)), operating across the New Zealand regional network several times per day. Since its launch in 2022, over 7,000 flights have been recorded, producing an exceptionally rich dataset for Earth surface observation.

The aim of the notebook is to provide an interactive walkthrough of the data to build familiarity with its structure and content. After completing it, users should feel confident enough to start developing their own analyses.

For background on the mission, visit the [Rongowai website](https://spoc.auckland.ac.nz/). For a comprehensive introduction to GNSS-R, see the open access book *Fundamentals and Applications of GNSS Reflectometry* by [Yang & Wang (2025)](https://doi.org/10.1007/978-981-96-4554-1).

---

## What the Notebook Does

### 1. Environment setup and data access
- Checks the Python environment for required packages and reports any missing ones
- Lists available L1 netCDF files and parses filenames (date, UTC time, departure/destination airports, data level) into a searchable interactive table
- Randomly selects or allows the user to specify a file; loads it with `xarray`

### 2. Data dictionary
- Loads the L1 [data dictionary](https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-docs/cygnss/open/L1/docs/L1_Dict_v2_4.xlsx) as an interactive, filterable widget
- Filter by variable name, data type, or dimension; view full field details for any selected variable
- Key variables to explore: `ddm_ant`, `ddm_snr`, `surface_reflectivity`, `surface_reflectivity_peak`

### 3. Data extraction and GeoPackage export
- Extracts all 2D data (dimensions: `sample`, `ddm`) from the netCDF and saves to a GeoPackage (`.gpkg`) compatible with QGIS and other GIS software
- Three layers are written: specular point observations (`spatial`), aircraft positions (`ac`), and flight track polyline (`flightvector`)
- The 4D DDM data (`surface_reflectivity` array) is not saved to the GeoPackage — it is accessed on-the-fly from the open `xarray` Dataset

### 4. Interactive data tables
- Loads the three GeoPackage layers into GeoDataFrames
- Displays them as interactive tables with multi-select column chooser, per-column value range filters, and summary statistics

### 5. Interactive map — specular points
- Maps LHCP specular point observations coloured by a user-selected variable (default: `surface_reflectivity_peak`), with point radius scaled by `fresnel_minor`
- Overlays the aircraft track (dashed white line) and aircraft position markers
- Satellite imagery basemap with controls for attribute, colormap, and colour stretch percentile
- Suggested exercises: explore `ddm_snr`, `sp_surface_type`, `sp_inc_angle`, `fresnel_minor/major`, `sv_num`

### 6. Fresnel zone estimation and mapping
- Computes elliptical Fresnel zone polygons from `fresnel_minor`, `fresnel_major`, and `fresnel_orientation`
- Two variants: instantaneous Fresnel zones (`fresnel`) and motion-stretched zones (`fresnel_integration`) accounting for aircraft/satellite movement during the 1 s observation window
- Results saved to the GeoPackage and mapped with the same interactive controls
- *Note: computation is slow — several minutes per file*

### 7. Interactive data exploration
- Applies a user-defined quality filter dictionary as a baseline (e.g. by coherence, incidence angle, surface type, SNR)
- Tabbed plotting widget: **Scatter** (any X, Y, colour variable), **Histogram** (with KDE overlay), and **Scatter matrix** (pairwise plots)
- Summary statistics update dynamically with filters
- Suggested exercises: explore how surface reflectivity varies with SNR, surface type, and incidence angle

### 8. DDM visualisation by surface type
- Randomly samples one observation per surface type from the filtered data
- Extracts the corresponding 2D DDM (`surface_reflectivity [sample, ddm, delay, doppler]`) from the netCDF
- Displays DDMs in a 3-column interactive grid with toggle controls for 2D/3D view and shared/independent colour scale
- Per-subplot titles show surface type label and min/median/max statistics
- **Redraw** button resamples new random observations

### 9. Summary and next steps
The notebook closes with a summary of what was covered and suggestions for further analysis, including combining Rongowai data with supplementary geospatial datasets (elevation models, soil type maps) and aggregating observations to a grid for temporal analysis.

---

## Data Access

Rongowai L1 data are publicly available from NASA PODAAC:

- **Dataset page**: https://podaac.jpl.nasa.gov/dataset/RONGOWAI_L1_SDR_V1.0
- **Directory listing**: https://cmr.earthdata.nasa.gov/virtual-directory/collections/C2784494745-POCLOUD/temporal
- **Programmatic download**: [PODAAC Data Subscriber](https://github.com/podaac/data-subscriber) (requires a free [NASA Earthdata login](https://urs.earthdata.nasa.gov/))

---

## Repository Structure

```
.
├── rongowai_notebook.ipynb     # Main Jupyter notebook
├── rongowai_helpers.py         # All helper functions (see below)
├── environment.yml             # Conda environment specification
├── L1_Dict_v2_4.xlsx           # L1 data dictionary
├── .env.example                # API key template — copy to .env and fill in
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-org/rongowai-notebook.git
cd rongowai-notebook
```

### 2. Create the conda environment

```bash
conda env create --file environment.yml
conda activate rongowai
python -m ipykernel install --user --name rongowai --display-name "Python (Rongowai)"
```

Setting up from scratch takes several minutes but is recommended to avoid dependency conflicts with other Python work.

### 3. Configure data paths

Edit the configuration cell near the top of the notebook to point to your local L1 data directory.

### 4. Configure API key (optional — for 3D satellite basemap)

A free Maptiler API key enables a 3D terrain satellite basemap. Get one at [maptiler.com/cloud](https://www.maptiler.com/cloud/) (no credit card required). Copy `.env.example` to `.env` and add your key:

```
# .env
MAPTILER_KEY=your_key_here
```

The notebook loads this automatically via `python-dotenv`. Without a key the map works using a 2D satellite tile basemap.

### 5. Launch the notebook

```bash
jupyter lab rongowai_notebook.ipynb
```

---

## `rongowai_helpers.py` — Function Reference

All helper functions are in `rongowai_helpers.py`, imported in the notebook as:

```python
import rongowai_helpers as rongowai
```

---

### Data processing

#### `parse_rongowai_files(file_list)`
Parses a list of Rongowai L1 filenames into a structured pandas DataFrame with columns for UTC date, time, departure airport, destination airport, and data level. Handles filenames with leading folder paths. Returns a DataFrame sorted by datetime.

```python
file_list = [f.as_posix() for f in L1_data.glob("*.nc")]
df_files = rongowai.parse_rongowai_files(file_list)
```

#### `l1_to_gpkg_single(l1_file, gpkg_file, verbose=False)`
Extracts all 2D spatial data (dimensions: `sample`, `ddm`) from a single L1 netCDF file and writes it to a GeoPackage with three layers: `spatial`, `ac`, and `flightvector`.

```python
rongowai.l1_to_gpkg_single(selected_file, gpkg_file)
```

#### `fresnel(gdf)`
Computes instantaneous Fresnel zone ellipse polygons for all observations and returns a GeoDataFrame of polygons. Saved to the GeoPackage as the `fresnel` layer.

```python
gdf_fresnel = rongowai.fresnel(gdf)
gdf_fresnel.to_file(gpkg_file, layer='fresnel', driver="GPKG")
```

#### `fresnel_integration(gdf)`
As `fresnel()`, but applies motion stretch to account for aircraft and satellite movement during the 1 s observation window. Returns more physically accurate Fresnel zone footprints. Saved as the `fresnel_integration` layer.

```python
gdf_fresnel_integration = rongowai.fresnel_integration(gdf)
gdf_fresnel_integration.to_file(gpkg_file, layer='fresnel_integration', driver="GPKG")
```

---

### Visualisation

#### `showDataDict(df_dict)`
Displays the L1 data dictionary as an interactive widget with dropdown and free-text filters, a sortable variable list table, and a detail panel for the selected variable.

```python
df_dict = pd.read_excel("L1_Dict_v2_4.xlsx", dtype=str).fillna("<none>")
rongowai.showDataDict(df_dict)
```

#### `showGeoDataFrame(gdf, caption="", df_dict=None, maxBytes=0)`
Displays a GeoDataFrame as an interactive table with multi-select column chooser, per-column value range filters (sliders + text inputs), and a summary statistics table above the data. Pass `df_dict` to add long variable names to the stats table.

```python
rongowai.showGeoDataFrame(gdf, caption=gpkg_file.name, df_dict=df_dict)
```

#### `map_rongowai_interactive(gdf, radius_attr="fresnel_minor", gdf_points=None, gdf_track=None, default_attribute="surface_reflectivity_peak", default_cmap="plasma", selected_cols=None)`
GPU-accelerated interactive map (via `lonboard`) with controls for attribute, colormap, and colour stretch percentile. Automatically detects Point vs Polygon geometry, switching between `ScatterplotLayer` (points scaled by `radius_attr`) and `SolidPolygonLayer` (for Fresnel zone polygons). Optional aircraft position markers and flight track overlay. A colourbar updates with each render.

```python
rongowai.map_rongowai_interactive(
    filtered_gdf,
    gdf_points=gdf_ac,
    gdf_track=gdf_flightvector,
)
```

#### `plot_rongowai_interactive(gdf, filters=None, default_x="sp_inc_angle", default_y="surface_reflectivity_peak", default_color="coherence_metric", df_dict=None)`
Tabbed interactive plotting widget with **Scatter**, **Histogram** (with optional KDE), and **Scatter matrix** tabs. Accepts a `filters` dict as a baseline, with additional interactive range sliders on top. Summary statistics update with all filters applied. Tabs render lazily on first selection for performance.

```python
filter_dict = {
    "coherence_metric": (1, 9999),
    "sp_inc_angle":     (0, 60),
    "ddm_snr":          (2, 9999),
}
rongowai.plot_rongowai_interactive(
    filtered_gdf,
    filters=filter_dict,
    default_x="sp_inc_angle",
    default_y="surface_reflectivity_peak",
    default_color="coherence_metric",
    df_dict=df_dict,
)
```

#### `plot_ddm_interactive(gdf, ds, ddm_var="surface_reflectivity", surface_type_col="sp_surface_type", surface_type_labels=None, metadata_cols=None, cmap="plasma")`
Randomly samples one observation per surface type and plots its 2D DDM as a heatmap or 3D surface in a 3-column grid. Controls: **Redraw** (new random sample), **Switch to 3D/2D**, **Switch to shared/independent scale**, and colormap selector. Subplot titles show the surface type label and min/median/max statistics. A selected samples table is shown above the plots.

```python
rongowai.plot_ddm_interactive(
    filtered_gdf,
    ds,
    ddm_var="surface_reflectivity",
    surface_type_labels={    
        -1: "-1: Ocean", 
        1: "1: Artificial",
        2: "2: Barely vegetated",
        3: "3: Inland water",
        4: "4: Crop",
        5: "5: Grass",
        6: "6: Shrub",
        7: "7: Forest",
    },
    metadata_cols=["sp_inc_angle", "ddm_snr", "coherence_metric",
                   "sp_surface_type", "sp_dist_to_coast_km"],
)
```

#### `plot_colorbar(attribute, vmin, vmax, cmap_name, nan_colour="#808080", figsize=(5, 0.8))`
Renders a standalone horizontal colourbar with a grey NaN swatch. Used internally by `map_rongowai_interactive` but can also be called independently.

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `xarray` / `netCDF4` | Reading Rongowai L1 netCDF files |
| `geopandas` | Spatial data handling and GeoPackage I/O |
| `lonboard` | GPU-accelerated interactive maps in Jupyter |
| `plotly` | Interactive scatter, histogram, and DDM plots |
| `itables` | Interactive sortable/filterable data tables |
| `ipywidgets` | UI controls (dropdowns, sliders, buttons, tabs) |
| `matplotlib` | Colourbars and static plots |
| `scipy` | KDE estimation in histograms |
| `python-dotenv` | Secure API key management |
| `tqdm` | Progress bars for batch processing |

---

## Notes on Data Structure

- Each `sample` corresponds to a set of simultaneous observations recorded at `ddm_timestamp_utc`, with up to 20 values per sample corresponding to `ddm` indices 0–19.
- `ddm_ant` distinguishes LHCP (`2`, indices 0–9) from RHCP (`3`, indices 10–19). LHCP and RHCP observations at the same index are co-located (same `sp_lat`, `sp_lon`, `sv_num`, and `ddm_timestamp_utc`). The map cells above filter to LHCP (`ddm_ant == 2`) to avoid stacked duplicate points.
- Variables prefixed `sp_*` relate to the specular point (dimensions: `sample`, `ddm`); `ac_*` to the aircraft (dimension: `sample`); `tx_*` to the GPS transmitting satellite.
- DDM data (`surface_reflectivity`, dimensions: `sample`, `ddm`, `delay`, `doppler`) are 5×40 arrays and are extracted on-the-fly from the open `xarray` Dataset — they are not stored in the GeoPackage.

---

## Further Reading

1. Bai, D., Ruf, C.S. and Moller, D., 2025. Calibration of the polarimetric GNSS-R sensor in the Rongowai mission. *IEEE Transactions on Geoscience and Remote Sensing*. https://doi.org/10.1109/TGRS.2025.3558131
2. Carreno-Luengo H, Ruf CS, Gleason S, Russel A. Latest progress on Rongowai polarimetric GNSS-R airborne mission. In *IGARSS 2024* (pp. 6828–6830). IEEE. https://doi.org/10.1109/IGARSS53475.2024.10642216
3. Moller, D., Orzel, K., Andreadis, K. and Wilson, M., 2024. An airborne GNSS-R driven low-latency flood assessment development. In *IGARSS 2024* (pp. 1370–1373). IEEE.
4. Moller, D., Wilson, M. et al., 2022. Rongowai: A pathfinder NASA/NZ GNSS-R initiative supporting SDG-15. In *IGARSS 2022* (pp. 4212–4215). IEEE. https://doi.org/10.1109/IGARSS46834.2022.9884397
5. Peng, J., Cardellach, E., Li, W., Ribó, S. and Rius, A., 2025. Impact of right-hand polarized signals in GNSS-R water detection algorithms. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 18, pp. 5646–5655.
6. Wilson, M. et al., 2025. Satellite sensing into agricultural practices: Cal/Val campaign of satellite, airborne and ground GNSS-Reflectometry of soil moisture. In *IGARSS 2025* (pp. 1117–1120). IEEE.
7. Wilson, M., Savarimuthu, S., Moller, D., Cai, X. and Ruf, C., 2024. Estimation of soil moisture from Rongowai GNSS-R using machine learning. In *MIGARS 2024* (pp. 1–4). IEEE.
8. Xu, Z. et al., 2026. GATENet: Spectral-auxiliary attention network for airborne GNSS-R based topographic estimation. *IEEE Geoscience and Remote Sensing Letters*. https://doi.org/10.1109/LGRS.2026.3668630

---

## Authors

- **Prof. Matthew Wilson** — Geospatial Research Institute Toi Hangarau, University of Canterbury
- **Xander Cai** — Geospatial Research Institute Toi Hangarau, University of Canterbury

---

## Licence

This notebook and associated helper code are released under the [MIT Licence](LICENSE).

Rongowai L1 data are provided by NASA PODAAC and subject to their [data use policy](https://podaac.jpl.nasa.gov/dataset/RONGOWAI_L1_SDR_V1.0).
