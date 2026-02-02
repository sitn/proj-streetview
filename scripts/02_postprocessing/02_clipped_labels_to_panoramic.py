import os, sys, json, time, tqdm, pandas as pd, geopandas as gpd

sys.path.insert(1, 'scripts')
import utils.misc as misc
from utils.logging import get_logger
from utils.config import get_config  

logger = get_logger()
cfg = get_config("This script transforms back the tiles labels into panoptic labels")
GEOMETRY_BUFFER = 1

def read_datasets():
    logger.info(f"Read datasets...")
    result_df = pd.DataFrame()
    for dataset_key, path in cfg['labels_files'].items():
        with open(path) as fp:
            coco_data = json.load(fp)
            tiles_df = pd.DataFrame.from_records(coco_data['images']).rename(columns={'id': 'image_id'})
            dataset_labels_df = pd.DataFrame.from_records(coco_data['annotations'])
        dataset_labels_df = dataset_labels_df.merge(tiles_df[['file_name', 'image_id']], how='left', on='image_id')
        dataset_labels_df['dataset'] = dataset_key
        result_df = pd.concat([result_df, dataset_labels_df], ignore_index=True)
    images_df = misc.read_image_info(cfg['panoptic_coco_files'], pd.read_csv(cfg['id_correspondence']))
    return images_df, result_df

def main():
    os.makedirs(cfg['output_folder'], exist_ok=True)
    images_df, clipped_labels_df = read_datasets()

    transformed_labels= []
    for tile_name in tqdm.tqdm(clipped_labels_df['file_name'].unique(), desc="Transform labels back to panoptic images"):
        transformed_labels.extend(misc.transform_annotations(tile_name, clipped_labels_df, images_df, buffer=GEOMETRY_BUFFER))

    transformed_labels_gdf = gpd.GeoDataFrame(pd.DataFrame.from_records(transformed_labels), geometry='buffered_geometry')

    for dataset in cfg['labels_files'].keys():
        logger.info(f"Working on {dataset} dataset...")
        subset_transformed_labels_gdf = transformed_labels_gdf[transformed_labels_gdf.dataset==dataset].copy()

        grouped_pairs_df = misc.group_annotations(subset_transformed_labels_gdf)
        merged_labels = []
        for group in tqdm.tqdm(grouped_pairs_df.group_id.unique(), desc="Merge labels in groups"):
            merged_labels.append(misc.make_new_annotation(group, grouped_pairs_df, buffer=GEOMETRY_BUFFER))

        subset_images_df = images_df[images_df.image_id.isin(subset_transformed_labels_gdf.image_id.unique())]
        subset_images_df = subset_images_df.rename(columns={'image_id': 'id'})
        cfg['categories'][0]['id'] = 0 # COCO usually starts with 1, but detectron2 starts with 0
        coco_dict = misc.assemble_coco_json(subset_images_df, merged_labels, cfg['categories'])

        filepath = os.path.join(cfg['output_folder'], f'{dataset}_COCO_panoptic_labels.json')
        with open(filepath, 'w') as fp:
            json.dump(coco_dict, fp)
        logger.info(f'Detections saved to {filepath}.')
    
if __name__ == "__main__":
    tic = time.time()
    main() 
    logger.info(f"Finished in {time.time() - tic:.2f} seconds.")