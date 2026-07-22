"""
Urban (built-up) area change detection for Bangkok Metropolitan Region.
Classification comes from Dynamic World V1 (Google/WRI/NASA), a deep-learning
(CNN semantic segmentation) land cover model run on Sentinel-2 imagery.
Change is derived automatically by diffing the thresholded "built" probability
band between two years.
"""

import json

import ee

GEE_PROJECT_ID = "peak-plasma-485606-u1"

# จังหวัดเชียงใหม่ (ขอบเขตจริงจาก FAO/GAUL, ไม่ใช่แค่ bounding box)
PROVINCE_NAME = "Chiang Mai"

BASELINE_YEAR = 2020
COMPARE_YEARS = [2022, 2023, 2025]  # selectable "after" years in the UI

URBAN_THRESHOLD = 0.5  # Dynamic World "built" probability cutoff
STATS_SCALE = 30  # meters, for area reduceRegion (coarser than native 10m for speed)

OUTPUT_JSON = "urban_change_result.json"
OUTPUT_HTML = "UrbanChangeMap.html"


def get_aoi():
    adm1 = ee.FeatureCollection("FAO/GAUL/2015/level1")
    province = adm1.filter(
        ee.Filter.And(
            ee.Filter.eq("ADM0_NAME", "Thailand"),
            ee.Filter.eq("ADM1_NAME", PROVINCE_NAME),
        )
    )
    return province.geometry()


def get_bounds(aoi):
    coords = aoi.bounds().coordinates().getInfo()[0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return [min(lons), min(lats), max(lons), max(lats)]


def built_probability(aoi, year):
    start, end = f"{year}-01-01", f"{year}-12-31"
    col = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(aoi)
        .filterDate(start, end)
        .select("built")
    )
    return col.mean().clip(aoi)


def area_km2(mask_image, aoi):
    pixel_area = ee.Image.pixelArea().divide(1e6)  # km^2
    value = (
        pixel_area.updateMask(mask_image)
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=STATS_SCALE,
            maxPixels=1e9,
            bestEffort=True,
        )
        .get("area")
    )
    result = value.getInfo()
    return round(result, 2) if result is not None else 0.0


def mask_tile_url(mask_image, color):
    map_id = mask_image.selfMask().getMapId({"min": 1, "max": 1, "palette": [color]})
    return map_id["tile_fetcher"].url_format


def change_image(urban_before, urban_after):
    # 0 = non-urban both years, 1 = new urban, 2 = unchanged urban, 3 = lost urban
    return (
        ee.Image(0)
        .where(urban_after.eq(1).And(urban_before.eq(0)), 1)
        .where(urban_after.eq(1).And(urban_before.eq(1)), 2)
        .where(urban_after.eq(0).And(urban_before.eq(1)), 3)
        .rename("change")
    )


def get_districts(province_name):
    adm2 = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(
        ee.Filter.And(
            ee.Filter.eq("ADM0_NAME", "Thailand"),
            ee.Filter.eq("ADM1_NAME", province_name),
        )
    )
    return adm2.select(["ADM2_NAME"])


def class_area_by_district(districts, change_image, cls):
    pixel_area = ee.Image.pixelArea().divide(1e6)  # km^2
    area_img = pixel_area.updateMask(change_image.eq(cls))
    reduced = area_img.reduceRegions(
        collection=districts,
        reducer=ee.Reducer.sum(),
        scale=STATS_SCALE,
    )
    features = reduced.getInfo()["features"]
    return {f["properties"]["ADM2_NAME"]: (f["properties"].get("sum") or 0) for f in features}


def district_centroids(districts):
    def add_centroid(f):
        c = f.geometry().centroid(maxError=1000).coordinates()
        return f.set({"lon": c.get(0), "lat": c.get(1)})

    features = districts.map(add_centroid).getInfo()["features"]
    return {
        f["properties"]["ADM2_NAME"]: (f["properties"]["lon"], f["properties"]["lat"])
        for f in features
    }


def district_stats(province_name, change_image):
    districts = get_districts(province_name)
    centroids = district_centroids(districts)
    class_names = {0: "non_urban", 1: "new_urban", 2: "unchanged_urban", 3: "lost_urban"}

    per_district = {}
    for cls, name in class_names.items():
        areas = class_area_by_district(districts, change_image, cls)
        for dist_name, area in areas.items():
            per_district.setdefault(dist_name, {"district": dist_name})[name] = round(area, 2)

    for row in per_district.values():
        lon, lat = centroids.get(row["district"], (None, None))
        row["lon"] = lon
        row["lat"] = lat

    rows = list(per_district.values())
    rows.sort(key=lambda r: r.get("new_urban", 0), reverse=True)
    return rows


def main():
    ee.Initialize(project=GEE_PROJECT_ID)
    aoi = get_aoi()
    aoi_bounds = get_bounds(aoi)

    all_years = [BASELINE_YEAR] + COMPARE_YEARS
    mask_cache = {}

    print("Computing per-year urban mask tiles...")
    year_tile_urls = {}
    for year in all_years:
        mask = built_probability(aoi, year).gt(URBAN_THRESHOLD).clip(aoi)
        mask_cache[year] = mask
        year_tile_urls[str(year)] = mask_tile_url(mask, "e34948")

    pair_data = {}
    for year in COMPARE_YEARS:
        print(f"Computing change {BASELINE_YEAR} -> {year} ...")
        change = change_image(mask_cache[BASELINE_YEAR], mask_cache[year]).clip(aoi)
        tiles = {
            "new_urban": mask_tile_url(change.eq(1), "e34948"),
            "unchanged_urban": mask_tile_url(change.eq(2), "898781"),
            "lost_urban": mask_tile_url(change.eq(3), "2a78d6"),
        }
        stats = {
            "new_urban": area_km2(change.eq(1), aoi),
            "unchanged_urban": area_km2(change.eq(2), aoi),
            "lost_urban": area_km2(change.eq(3), aoi),
            "non_urban": area_km2(change.eq(0), aoi),
        }
        pair_data[f"{BASELINE_YEAR}_{year}"] = {"tile_urls": tiles, "stats_km2": stats}
        print(json.dumps(stats, ensure_ascii=False))

    district_after_year = COMPARE_YEARS[-1]
    print(f"Computing district-level statistics ({BASELINE_YEAR} -> {district_after_year})...")
    change_for_districts = change_image(
        mask_cache[BASELINE_YEAR], mask_cache[district_after_year]
    ).clip(aoi)
    districts = district_stats(PROVINCE_NAME, change_for_districts)

    result = {
        "province": PROVINCE_NAME,
        "aoi_bounds": aoi_bounds,
        "baseline_year": BASELINE_YEAR,
        "compare_years": COMPARE_YEARS,
        "urban_threshold": URBAN_THRESHOLD,
        "year_tile_urls": year_tile_urls,
        "pair_data": pair_data,
        "district_stats_km2": districts,
        "district_baseline": {"year_before": BASELINE_YEAR, "year_after": district_after_year},
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Top districts by new urban area:")
    for row in districts[:5]:
        print(f"  {row['district']}: {row.get('new_urban', 0)} km^2")
    print(f"Saved tile URLs + stats to {OUTPUT_JSON}")
    return result


if __name__ == "__main__":
    main()
