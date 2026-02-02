import os, json, sys, time, tqdm, pandas as pd, geopandas as gpd

sys.path.insert(1, 'scripts')
from utils import misc
from utils.logging import get_logger
from utils.config import get_config  

logger = get_logger()
cfg = get_config("This script convert raw detections to panoptic detections.")

def read_detections_with_threshold(score_threshold):
    logger.info(f"Read detections with a threshold of {score_threshold} on the confidence score...")
    detections_df = pd.DataFrame()
    for dataset_key, path in cfg['detections_files'].items():
        with open(path) as fp:
            dets = pd.DataFrame.from_records(json.load(fp))
            dets['dataset'] = dataset_key
        detections_df = pd.concat([detections_df, dets], ignore_index=True)
    logger.info(f"Detections before thresholding : {len(detections_df)}")
    return detections_df[detections_df.score >= score_threshold]

def main():
    GEOMETRY_BUFFER = 1
    os.makedirs(cfg['output_folder'], exist_ok=True)
    detections_df = read_detections_with_threshold(cfg['score_threshold'])
    logger.info(f"Detections after thresholding : {len(detections_df)}")

    images_df = misc.read_image_info(cfg['panoptic_coco_files'], pd.read_csv(cfg['id_correspondence']))

    transformed_detections= []
    for tile_name in tqdm.tqdm(detections_df['file_name'].unique(), desc="Transform detections back to panoptic images"):
        transformed_detections.extend(
            misc.transform_annotations(tile_name, detections_df, images_df, buffer=GEOMETRY_BUFFER, id_field='det_id', category_field='det_class')
        )

    transformed_detections_gdf = gpd.GeoDataFrame(pd.DataFrame.from_records(transformed_detections), geometry='buffered_geometry')
    transformed_detections_gdf = transformed_detections_gdf[~transformed_detections_gdf.geometry.is_empty]

    for dataset in cfg['detections_files'].keys():
        logger.info(f'Working on the {dataset} dataset...')
        subset_transformed_detections_gdf = transformed_detections_gdf[transformed_detections_gdf.dataset==dataset].copy()

        logger.info('Groupping overlapping detections...')
        groupped_pairs_df = misc.group_annotations(subset_transformed_detections_gdf, verbose=True)

        merged_detections = []
        for group in tqdm.tqdm(groupped_pairs_df.group_id.unique(), desc="Merge detections in groups"):
            merged_detections.append(misc.make_new_annotation(group, groupped_pairs_df, buffer=GEOMETRY_BUFFER))
        logger.info(f"{len(merged_detections)} detections are left after merging.")

        logger.info("Transforming detections to COCO format...")
        subset_images_df = images_df[images_df.image_id.isin(subset_transformed_detections_gdf.image_id.unique())].rename(columns={'image_id': 'id'})
        cfg['categories'][0]['id'] = 0 # COCO usually starts with 1, but detectron2 starts with 0
        filepath = os.path.join(cfg['output_folder'], f'{dataset}_COCO_panoptic_detections.json')
        with open(filepath, 'w') as fp:
            json.dump(misc.assemble_coco_json(subset_images_df, merged_detections, cfg['categories']), fp)
        logger.info(f'Detections saved to {filepath}.')

if __name__ == "__main__":
    tic = time.time()
    main() 
    logger.info(f"Finished in {time.time() - tic:.2f} seconds.")