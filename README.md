# Urban Change Detection — Chiang Mai

Automated Land Use / Land Cover (LULC) change detection web map. Classifies
built-up (urban) area from satellite imagery using a deep learning model and
visualizes urban growth over time for Chiang Mai province, Thailand.

**Live demo:** https://nutto-cyber.github.io/Webproject/

## Overview

- **Classification:** [Dynamic World V1](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1) — a near-real-time land cover model (CNN semantic segmentation) run on Sentinel-2 imagery via Google Earth Engine. The "built" probability band is thresholded to produce a binary urban mask per year.
- **Change detection:** automatic per-pixel diff between a fixed baseline year (2020) and a selectable comparison year, classified into new / unchanged / lost urban area.
- **Area of interest:** Chiang Mai province, using the official administrative boundary (FAO GAUL), not a bounding box.
- **District breakdown:** per-district (amphoe) statistics computed with a grouped `reduceRegions` call, click a row to zoom to that district.

## Tech stack

| Layer | Tool |
|---|---|
| Satellite data & classification | Google Earth Engine (`earthengine-api`), Dynamic World V1 |
| Area/district statistics | Earth Engine `reduceRegion` / `reduceRegions` (FAO GAUL admin boundaries) |
| Web map | Leaflet.js, vanilla HTML/CSS/JS (no build step) |
| Basemap | OpenStreetMap tiles |

## Project structure

```
.
├── UrbanChangeMap.html          # Self-contained web map (open directly in a browser)
├── urban_change_detection.py    # Earth Engine pipeline: computes tiles + stats, writes urban_change_result.json
└── urban_change_result.json     # Last computed output (tile URLs + area stats)
```

## Running it locally

```bash
pip install earthengine-api
python urban_change_detection.py   # authenticate with your own GEE project first
```

The script prints province- and district-level area statistics (km²) and writes
`urban_change_result.json`. Copy the `tile_urls` / `pair_data` / `district_stats_km2`
values from that file into the `RESULT` object at the top of the `<script>` block
in `UrbanChangeMap.html`, then open the file in a browser.

> **Note:** Earth Engine tile URLs (`tile_fetcher.url_format`) are signed and
> expire after a period of time. If map layers stop loading, re-run the script
> to refresh the URLs embedded in `UrbanChangeMap.html`.

## Features

- Toggle between baseline year, comparison year, and change view
- Select the comparison year from a dropdown (baseline fixed at 2020)
- District table sorted by new urban area, click a row to zoom to it
- Light / dark theme toggle
- KPI stat tiles (new / unchanged / lost / non-urban area in km²)

## Data source

Google Earth Engine — `GOOGLE/DYNAMICWORLD/V1` (10m resolution, 2015–present) and
`FAO/GAUL/2015/level1` / `level2` for Thailand administrative boundaries.
