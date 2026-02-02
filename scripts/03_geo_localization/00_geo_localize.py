import os, sys, numpy as np, geopandas as gpd, pandas as pd
from functools import partial
from shapely.geometry import Point

sys.path.insert(1, 'scripts')
from utils import triangulation as tri

BASE_PATH = "D:/Repos/proj-streetview"
TRAJECT_FILE = f'{BASE_PATH}/data/NE/images.fgb'
COCO_FILES_PATH = {
    'trn': f'{BASE_PATH}/outputs/02_01_transform_detections/trn_COCO_panoptic_detections.json',
    'tst': f'{BASE_PATH}/outputs/02_01_transform_detections/tst_COCO_panoptic_detections.json',
    'val': f'{BASE_PATH}/outputs/02_01_transform_detections/val_COCO_panoptic_detections.json',
    'oth': f'{BASE_PATH}/outputs/02_01_transform_detections/oth_COCO_panoptic_detections.json'
}
OUTPUT_DIR = f'{BASE_PATH}/outputs/03_00_geo_localize'
os.makedirs(OUTPUT_DIR, exist_ok=True)

gdf = gpd.read_file(TRAJECT_FILE)
gdf['file_name'] = gdf['file_name'].str.replace(r'\.\w+$', '', regex=True)
numeric_columns = ['gpsimgdirection', 'gpspitch', 'gpsroll', 'gpslatitude', 'gpslongitude', 'gps_sec_s_', 'x_m_', 'y_m_', 'z_m_', 'x', 'y', 'z']
gdf[numeric_columns] = gdf[numeric_columns].astype(float)

images_df = pd.DataFrame()
ann_gdf = pd.DataFrame()
for dataset, file in COCO_FILES_PATH.items():
    cur_images_df, cur_ann_gdf = tri.load_coco_inferences(file, t_score=0.7)
    cur_images_df['dataset'] = dataset
    images_df = pd.concat([images_df, cur_images_df], ignore_index=True)
    ann_gdf = pd.concat([ann_gdf, cur_ann_gdf], ignore_index=True)

images_df['file_name'] = images_df['file_name'].str.replace(r'\.\w+$', '', regex=True)

if 'file_name' not in gdf.columns:
    raise ValueError("gdf must have a 'file_name' column to match with COCO images.")

gdf = gdf.merge(images_df[['id', 'file_name', 'width', 'height', 'dataset']], left_on='file_name', right_on='file_name', how='right')
gdf = gdf.merge(ann_gdf, left_on='id', right_on='image_id', how='right')

gdf['geometry'] = gdf['geometry_y']
gdf.set_geometry('geometry')

grouped = tri.spatial_temporal_group_sort(gdf,
    groupby_col='image_id',
    time_column="gps_sec_s_",
    x_col="x_m_", y_col="y_m_", z_col="z_m_",
    radius=10.0
)


# To pass an offset to cylin_pano_proj_ray while keeping the (frame_id, frame) signature, use a lambda or functools.partial to "bind" the offset argument:
out_gdf, candidate_list, ray_list, intersection_objs = tri.triangulation_peeling(
    grouped, 
    partial(tri.cylin_pano_proj_ray, offset=[0,0,0,0.5,0,-0.3]),
    intersection_threshold=0.5,
    clustering_threshold=1,
    candidate_update_threshold=1,
    candidate_missing_limit=5,
    radius=20,
    mask_area_control=False,
    height_control=True
)


out_file_triangulation_peeling = os.path.join(OUTPUT_DIR, 'triangulation_peeling.fgb') 
out_gdf.to_file(out_file_triangulation_peeling)
print(f"Saved file {out_file_triangulation_peeling}")

# Convert intersection_objs to GeoDataFrame
intersection_gdf = gpd.GeoDataFrame(
    [{  "geometry": Point(inter.point),
        "ray_pair": inter.ray_pair,
        "dist": inter.dist,
        "length": inter.length
    } for inter in intersection_objs],
    geometry="geometry", crs="epsg:2056"
)

out_file_triangulation_intersection = os.path.join(OUTPUT_DIR, 'triangulation_peeling_intersections.fgb') 
intersection_gdf.to_file(out_file_triangulation_intersection)
print(f"Saved file to {out_file_triangulation_intersection}")

RANGE_LIMIT = 15
MIN_INTERSECT = 2
MAX_INTERSECTION_DIST = 0.2
pred_gdf = gpd.read_file(out_file_triangulation_peeling)
gt_gdf = gpd.read_file(f'{BASE_PATH}/data/RCNE/NE_GT_3D.gpkg', layer='ne_gt_3d')
traject = pd.read_csv(TRAJECT_FILE)

# load and filter initial predictions
pred_gdf = pred_gdf[
    (pred_gdf.mean_intersection_length <= RANGE_LIMIT) 
    & (pred_gdf.mean_intersection_dist <= MAX_INTERSECTION_DIST) 
    & (np.array(pred_gdf.intersections) >= MIN_INTERSECT)]


out_file_triangulation_peeling_15m = os.path.join(OUTPUT_DIR, 'triangulation_peeling_15m.fgb')
pred_gdf.reset_index().to_file(out_file_triangulation_peeling_15m)
print(f"Saved file to {out_file_triangulation_peeling_15m}")


# Construct geometry from x_m_, y_m_, z_m_ and 2D geometry column for XY plane
traject['geometry'] = traject.apply(lambda row: Point(row['x_m_'], row['y_m_'], row['z_m_']), axis=1)
traject['geometry_xy'] = traject.apply(lambda row: Point(row['x_m_'], row['y_m_']), axis=1)

# --- Matrix calculation for filtering gt_gdf by nearby traject points in XY ---

# Get centroid XY coordinates for all GT geometries
gt_centroids_xy = np.array([[pt.x, pt.y] for pt in gt_gdf.geometry.centroid])

# Get all traject XY coordinates
traject_xy = np.array([[pt.x, pt.y] for pt in traject['geometry_xy']])

# Compute distance matrix: shape (n_gt, n_traject)
dists_matrix = np.linalg.norm(gt_centroids_xy[:, None, :] - traject_xy[None, :, :], axis=2)

# Count how many traject points are within RANGE_LIMIT meters for each GT centroid
nearby_counts = (dists_matrix <= RANGE_LIMIT).sum(axis=1)

# Filter gt_gdf: keep only rows where at least 3 traject points are within 15m
gt_gdf = gt_gdf[nearby_counts >= 3].reset_index(drop=True)

# --- Additional code block: metrics based on match distance between prediction 2D coordinates and centroid of 2D gt ---
# Only match distance below 2m can be defined as true positive.

# Prepare 2D coordinates for predictions and GT centroids
pred_points_2d = np.array([[geom.x, geom.y] for geom in pred_gdf.geometry])
gt_centroids_2d = np.array([[pt.x, pt.y] for pt in gt_gdf.geometry.centroid])

n_pred = len(pred_points_2d)
n_gt = len(gt_centroids_2d)

# Compute distance matrix: shape (n_gt, n_pred)
dist_matrix = np.linalg.norm(gt_centroids_2d[:, None, :] - pred_points_2d[None, :, :], axis=2)

# Flatten the distance matrix and sort by distance
gt_indices, pred_indices = np.unravel_index(np.argsort(dist_matrix, axis=None), dist_matrix.shape)
sorted_distances = dist_matrix[gt_indices, pred_indices]

gt_to_pred = {}
pred_matched, gt_matched = set(), set()
match_distances = []
DISTANCE_THRESHOLD = 2.0  # Only matches below 2m are considered TP

for gt_idx, pred_idx, dist in zip(gt_indices, pred_indices, sorted_distances):
    if dist >= DISTANCE_THRESHOLD:
        break  # All remaining pairs are above threshold
    if gt_idx not in gt_matched and pred_idx not in pred_matched:
        gt_to_pred[gt_idx] = pred_idx
        gt_matched.add(gt_idx)
        pred_matched.add(pred_idx)
        match_distances.append(dist)

# Now, metrics:
TP = len(gt_to_pred)  # Each GT matched to a unique pred within 2m
FP = n_pred - len(pred_matched)  # Predictions not matched to any GT (within 2m)
FN = n_gt - len(gt_to_pred)      # GTs not matched to any pred (within 2m)

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Distance statistics for matches
if match_distances:
    match_distances_np = np.array(match_distances)
    match_dist_stats = {
        'mean': np.mean(match_distances_np),
        'std': np.std(match_distances_np),
        'min': np.min(match_distances_np),
        'max': np.max(match_distances_np),
        'median': np.median(match_distances_np),
        'count': len(match_distances_np)
    }
else:
    match_dist_stats = {}

print("=== Greedy GT-to-Pred Closest Matching Metrics (2m threshold) ===")
print(f"TP: {TP}, FP: {FP}, FN: {FN}")
print(f"Precision: {precision:.3f}, \nRecall: {recall:.3f}, \nF1: {f1_score:.3f}")
print("Match distance stats:")
for k, v in match_dist_stats.items():
    print(f"  {k}: {v}")

# Get indices for false positives (pred points not matched to any GT polygon)
all_pred_indices = set(pred_gdf.index)
matched_pred_indices = pred_matched
fp_indices = np.array(sorted(list(all_pred_indices - matched_pred_indices)))

# Get indices for false negatives (GT polygons not matched to any pred point)
all_gt_indices = set(gt_gdf.index)
matched_gt_indices = set(gt_to_pred.keys())
fn_indices = np.array(sorted(list(all_gt_indices - matched_gt_indices)))

# print(f"{len(fp_indices)} False Positive indices (pred points not matched): {fp_indices}")
# print(f"{len(fn_indices)} False Negative indices (GT polygons not matched): {fn_indices}")

#print(pred_gdf.iloc[fp_indices])