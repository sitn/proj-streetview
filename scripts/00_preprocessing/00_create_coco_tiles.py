import math, cv2, json, os, sys, time, tqdm, numpy as np, pandas as pd
from joblib import Parallel, delayed

sys.path.insert(1, 'scripts')
import utils.misc as misc
from utils.logging import get_logger
from utils.config import get_config

cfg = get_config("This script prepares COCO datasets")
logger = get_logger()

TILE_SIZE = cfg['tile_size']

def read_coco_dataset(coco_file_path):
    # The function expects the COCO JSON to have 'images' and 'annotations' keys.
    # The 'id' field in the images data is renamed to 'image_id' if not already present.
    logger.info(f"Reading COCO dataset from {coco_file_path}...")
    with open(coco_file_path, 'r') as fp:
        coco_data = json.load(fp)

    images_df = pd.DataFrame.from_records(coco_data['images'])
    images_df["annotations"] = [[] for _ in range(len(images_df))]
    if 'image_id' not in images_df.columns:
        images_df.rename(columns={'id':'image_id'}, inplace=True)
    
    for annotation in coco_data['annotations']:
        image_of_annotation = images_df.loc[images_df['image_id'] == annotation['image_id'], 'annotations']
        if len(image_of_annotation) < 1 :
            logger.error(f"Annotations found for image {annotation['image_id']}, but no corresponding image")
        else:
            images_df.loc[images_df['image_id'] == annotation['image_id'], 'annotations'].iloc[0].append(annotation)

    images_df['original_id'] = images_df['image_id']
    return images_df

def borderline_intersection(coord_tuples_list):
    first_coords = all(v[0] <= TILE_SIZE * 0.02 or v[0] >= TILE_SIZE * 0.98 for v in coord_tuples_list)
    second_coords = all(v[1] <= TILE_SIZE * 0.02 or v[1] >= TILE_SIZE * 0.98 for v in coord_tuples_list)
    return first_coords or second_coords

def check_bbox_plausibility(new_origin, length):
    """
    Adjusts and checks the plausibility of a bounding box's origin and length within a tile.
    Args:
        new_origin (int): The proposed new origin of the bounding box.
        length (int): The proposed length of the bounding box.
    Returns:
        tuple: A tuple containing the adjusted new_origin and length, ensuring they fit within the tile size.
    Raises:
        AssertionError: If the adjusted bounding box origin or length is outside the tile.
    """
    if new_origin < 0:
        length = min(length + new_origin, TILE_SIZE)
        new_origin = 0
    elif new_origin + length > TILE_SIZE:
        length = TILE_SIZE - new_origin
    assert all(value <= TILE_SIZE and value >= 0 for value in [new_origin, length]), "Annotation outside tile"
    return new_origin, length

def get_new_coordinate(initial_coor, tile_min): # Calculates the new coordinate for a bounding box annotation to be within a tile.
    return max(min(initial_coor-tile_min, TILE_SIZE), 0)

def image_to_tiles(image_path, output_tiles, rejected_annotations_df):
    """
    Processes an image by dividing it into tiles, applying masks on pixels corresponding to rejected annotations, and saving the tiles.
    Args:
        image_path (str): The path of the image file.
        output_tiles (list): A list of paths to the tiles that the image should be cut into.
        rejected_annotations_df (DataFrame): A DataFrame containing annotations that should be rejected (masked) on the tiles.
        tasks_dict (dict): A dictionary defining the tasks to prepare data for, including subfolder paths.
    Returns:
        dict: A dictionary of the tiles that could not be saved with tile paths as key and False as value.
    """
    tasks_dict = cfg['tasks']
    out_dirs = [tasks_dict[task]['subfolder'] for task in tasks_dict.keys() if tasks_dict[task]['prepare_data']]
    img = cv2.imread(os.path.join(image_path))
    if img is None:
        logger.error(f"Image {image_path} could not be read.")
        return {tile_path: False for tile_path in output_tiles}

    for tile_name in output_tiles:
        files_exist = all([
            os.path.exists(os.path.join(out_dir, tile_name)) or os.path.exists(os.path.join(out_dir, os.path.basename(tile_name))) 
            for out_dir in out_dirs
        ])
        if files_exist and not cfg['overwrite_images']:
            continue

        i = int(tile_name.split("_")[-1].rstrip(".jpg"))
        j = int(tile_name.split("_")[-2])
        tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
        img_cut = img[i:i+TILE_SIZE, j:j+TILE_SIZE]
        try:
            tile[:] = img_cut
             # Draw a black mask on rejected annotations
            annotations_to_mask_df = rejected_annotations_df[rejected_annotations_df.file_name == tile_name]
            for ann in annotations_to_mask_df.itertuples():
                bbox = [int(b) for b in ann.bbox]
                bounds = [bbox[0], bbox[1], min(bbox[0]+round(bbox[2]*1.1), TILE_SIZE), min(bbox[1]+round(bbox[3]*1.1), TILE_SIZE)]
                cv2.rectangle(img=tile, pt1=(bounds[0], bounds[1]), pt2=(bounds[2], bounds[3]), color=(0, 0, 0), thickness=-1)
            
            if 'coco' in tasks_dict:
                if not cv2.imwrite(os.path.join(tasks_dict['coco']['subfolder'], tile_name), tile) :
                    logger.error(f'Tile {tile_name} could not be produced for coco task')
            if 'yolo' in tasks_dict:
                if not cv2.imwrite(os.path.join(tasks_dict['yolo']['subfolder'], tile_name), tile) :
                    logger.error(f'Tile {tile_name} could not be produced for yolo task')
        except:
            logger.warn(f"Tile {tile_name} was not perfectly cut into tiles of size {TILE_SIZE}. Left: {img.shape[1]-TILE_SIZE}, Top: {img.shape[0]-TILE_SIZE}")

def select_low_tiles(tiles_df, clipping_params_dict, excluded_height_ratio=1/2):
    """
    Select tiles that are above a certain height ratio.
    Args:
        tiles_df (DataFrame): A DataFrame containing the tiles.
        excluded_height_ratio (float, optional): The height ratio under which tiles should be excluded. Defaults to 1/2.
    Returns:
        DataFrame: A DataFrame containing the selected tiles.
    """
    _tiles_df = tiles_df.reset_index(drop=True)
    aoi = _tiles_df.loc[0, 'AOI']
    if "height" in clipping_params_dict[aoi].keys():
        image_height = clipping_params_dict[aoi]["height"]
    elif 'lb4' in clipping_params_dict[aoi].keys():
        image_height = clipping_params_dict[aoi]['lb4']["height"]
    else:
        image_height = clipping_params_dict[aoi]['else']["height"]

    _tiles_df["row_level"] = _tiles_df["file_name"].apply(lambda x: int(x.split("_")[-1].rstrip(".jpg")))
    low_tiles_df = _tiles_df[_tiles_df["row_level"] >= image_height*excluded_height_ratio].reset_index(drop=True)
    return low_tiles_df.drop(columns=["row_level"])

def read_coco_files(created_dirs, created_files):
    original_coco_dict = {}
    id_correspondence_df = pd.DataFrame()
    max_id = 0
    for aoi, coco_file in cfg['original_COCO_files'].items():
        images_df = read_coco_dataset(coco_file)
        images_df['image_id'] = images_df['image_id'] + max_id
        images_df["AOI"] = aoi
        original_coco_dict[aoi] = images_df
        max_id += images_df.image_id.max() + 1
        id_correspondence_df = pd.concat([id_correspondence_df, images_df[["AOI", 'image_id', 'original_id']]], ignore_index=True)
    for out_dir in created_dirs:
        filepath = os.path.join(out_dir, "original_ids.csv")
        id_correspondence_df.to_csv(filepath, index=False)
        created_files.append(filepath)
    logger.info(f"Found {sum([len(df) for df in original_coco_dict.values()])} images")
    return original_coco_dict

def read_validated_coco_files(original_coco_dict):
    logger.info(f"Reading validated COCO files...")
    validated_coco_files = cfg['validated_COCO_files'] if 'validated_COCO_files' in cfg else None
    valid_coco_dict = {}
    if isinstance(validated_coco_files, dict): # Case: training
        for aoi, coco_file in validated_coco_files.items():
            images_df = read_coco_dataset(coco_file)
            images_df['image_id'] = images_df.drop(columns='image_id').merge(original_coco_dict[aoi], how='left', on='original_id').image_id
            assert images_df['image_id'].isna().sum() == 0, "Validated COCO dataset contains images that are not in the original COCO dataset."
            valid_coco_dict[aoi] = images_df
    else: # Case: inference-only
        valid_coco_dict = {key: pd.DataFrame() for key in original_coco_dict.keys()}
    
    nbr_annotations = sum([len(ann_list) for df in valid_coco_dict.values() for ann_list in df.annotations])
    nbr_images = sum([len(df) for df in valid_coco_dict.values()])
    logger.info(f"Found {nbr_images} images corresponding to {nbr_annotations} validated annotations")
    return valid_coco_dict

def split_datasets(original_coco_dict, valid_coco_dict):
    if all(df.empty for df in valid_coco_dict.values()):
        for df in original_coco_dict.values():
            df['dataset'] = 'oth'
    elif cfg['test_only']:
        logger.warning('Test-only mode activated. All annotations will be stored in the test set.')
        for aoi, valid_coco_df in valid_coco_dict.items():
            valid_coco_df["dataset"] = "tst"
            original_coco_df = original_coco_dict[aoi]
            original_coco_df.loc[original_coco_df["image_id"].isin(valid_coco_df["image_id"]), "dataset"] = "tst"
            original_coco_df.loc[original_coco_df.dataset.isna(), "dataset"] = "oth"
    else:
        logger.info("Splitting images into train, val and test sets based on ratio 70% / 15% / 15%...")
        for aoi, valid_coco_df in valid_coco_dict.items():
            trn_tiles = valid_coco_df.sample(frac=0.7, random_state=cfg['seed'])
            val_tiles = valid_coco_df[~valid_coco_df["image_id"].isin(trn_tiles["image_id"])].sample(frac=0.5, random_state=cfg['seed'])
            tst_tiles = valid_coco_df[~valid_coco_df["image_id"].isin(trn_tiles["image_id"].to_list() + val_tiles["image_id"].to_list())]

            original_coco_df = original_coco_dict[aoi]
            valid_coco_df["dataset"] = None
            for dataset, df in {"trn": trn_tiles, "val": val_tiles, "tst": tst_tiles}.items():
                valid_coco_df.loc[valid_coco_df["image_id"].isin(df["image_id"]), "dataset"] = dataset
                original_coco_df.loc[original_coco_df["image_id"].isin(df["image_id"]), "dataset"] = dataset
            original_coco_df.loc[original_coco_df.dataset.isna(), "dataset"] = "oth"

            assert all(valid_coco_df["dataset"].notna()), "Not all images were assigned to a dataset"
            nbr_oth = len(original_coco_df[original_coco_df.dataset=='oth'])
            logger.info(f"{aoi} : Train: {len(trn_tiles)} | Eval: {len(val_tiles)} | Test: {len(tst_tiles)} | Other (without validated annotations): {nbr_oth}")

def get_tiles_from_image(image, p, tile_counter):
    tiles = []
    for i in range(p["padding_y"], p["height"]-p["padding_y"]-p["overlap_y"], TILE_SIZE-p["overlap_y"]):
        for j in range(0, p["width"]-p["overlap_x"], TILE_SIZE-p["overlap_x"]):
            file_name = f"{os.path.basename(image.file_name).rstrip('.jpg')}_{j}_{i}.jpg"
            tiles.append({"file_name": file_name, 'min_x':j, 'min_y':i, 'id':tile_counter})
            tile_counter+=1
    return pd.DataFrame(tiles), tile_counter

def clip_into_tiles(original_coco_dict, valid_coco_dict, ratio_wo_annotation):
    logger.info(f"Iterate through annotations and clip them into tiles")
    tot_tiles_with_ann, tot_tiles_without_ann, excluded_annotations, tile_counter = 0, 0, 0, 0
    gt_tiles_df, oth_tiles_df, clipped_annotations_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rejected_annotations_df = pd.DataFrame(columns=['id', 'file_name', 'bbox', 'image_name'])
    for aoi, original_coco_df in original_coco_dict.items():
        
        valid_coco_dict_aoi = valid_coco_dict[aoi]
        assert valid_coco_dict_aoi['annotations'].duplicated().any(), f"Duplicated annotations for AOI {aoi}"
        
        for image in tqdm.tqdm(original_coco_df.itertuples(), desc=f"Defining tiles for {aoi}", total=len(original_coco_df)):
            
            image_path = os.path.join(cfg['image_dir'][aoi], image.file_name)
            if not os.path.exists(image_path):
                logger.error(f"Image {image.file_name} not found")
                continue
            
            p = cfg['clipping_params'][aoi]
            if aoi == 'SZH':
                p = p['lb4'] if 'lb4' in image.file_name else p['else']

            tiles_df, tile_counter = get_tiles_from_image(image, p, tile_counter)
            tiles_df["height"] = TILE_SIZE
            tiles_df['width'] = TILE_SIZE
            tiles_df['dataset'] = image.dataset
            tiles_df['original_image'] = image_path
            tiles_df['original_id'] = image.original_id
            tiles_df['AOI'] = aoi

            valid_annotations = valid_coco_dict_aoi.loc[valid_coco_dict_aoi.image_id==image.image_id, 'annotations'].iloc[0]
            #logger.info(f"Image {image.file_name} cut into {len(tiles_df)} tiles with {len(valid_annotations)} valid annotations")

            annotations = []
            tile_annotations_df = pd.DataFrame(columns=['image_id', 'object_id', 'id', 'bbox', 'area', 'category_id', 'iscrowd', 'segmentation'])
            if not valid_coco_dict_aoi.empty : # Case: training
                border_annotations = 0
                for ann in image.annotations:

                    # Check if annotation is valid
                    validated_ann = [a for a in valid_annotations if a["id"] == ann["id"]]
                    if len(validated_ann) == 1 :
                        ann = validated_ann[0]
                    origin_x, origin_y, width, height = ann["bbox"]

                    # Check if annotation is outside the image
                    if origin_x + width <= 0 or origin_x >= p["width"] or origin_y + height <= 0 or origin_y >= p["height"]:
                        logger.warning(f"Annotation {ann['id']} is outside the image {image.file_name}. Bbox: {[round(value) for value in ann['bbox']]}.")
                        excluded_annotations += 1
                        continue

                    if origin_y > p["height"] - p["padding_y"]:
                        border_annotations += 1
                        continue
                
                    for _, tile in tiles_df.iterrows():
                        tile_max_x = tile['min_x'] + TILE_SIZE
                        tile_max_y = tile['min_y'] + TILE_SIZE

                        # Check if annotation is outside the tile
                        if origin_x >= tile_max_x or origin_x + width <= tile['min_x'] or origin_y >= tile_max_y or origin_y + height <= tile['min_y']:
                            continue
                        # else, scale coordinates and clip if necessary
                        x1, new_width = check_bbox_plausibility(origin_x - tile['min_x'], width)
                        y1, new_height = check_bbox_plausibility(origin_y - tile['min_y'], height)
                        new_coords_tuples = [(x1, y1), (x1 + new_width, y1 + new_height)]
                        if borderline_intersection(new_coords_tuples):
                            border_annotations += 1
                            continue

                        if len(validated_ann) == 0: # Rejected annotation
                            rejected_annotations_df = pd.concat((rejected_annotations_df, pd.DataFrame.from_records([{
                                "id": ann["id"], 
                                "file_name": tile["file_name"], 
                                "bbox": [x1, y1, new_width, new_height], 
                                'image_name': image.file_name,
                            }])), ignore_index=True)
                        else:
                            # segmentation
                            old_coords = ann["segmentation"][0]
                            coords = [get_new_coordinate(old_coords[0],  tile['min_x']), get_new_coordinate(old_coords[1], tile['min_y'])] # set first coordinates
                            new_coords_tuples = []
                            for i in range(2, len(old_coords), 2):
                                new_x = get_new_coordinate(old_coords[i],  tile['min_x'])
                                new_y = get_new_coordinate(old_coords[i+1], tile['min_y'])
                                if new_x in [0, TILE_SIZE] and coords[-2] == new_x or new_y in [0, TILE_SIZE] and coords[-1] == new_y or (new_x, new_y) in new_coords_tuples:
                                    continue 
                                new_coords_tuples.append((new_x, new_y))
                                coords.extend([new_x, new_y])
                            assert all(value <= TILE_SIZE and value >= 0 for value in coords), "Mask outside tile"
                            if borderline_intersection(new_coords_tuples):
                                border_annotations += 1
                                continue

                            annotations.append(dict(
                                object_id = ann["object_id"],
                                image_id=tile['id'],
                                category_id=1,  # Currently, single class
                                iscrowd=int(ann["iscrowd"]),
                                bbox=[x1, y1, new_width, new_height],
                                area=misc.segmentation_to_polygon([coords]).area,
                                segmentation=[coords]
                            ))

                tile_annotations_df = pd.DataFrame(annotations, columns=['image_id'] if len(annotations) == 0 else annotations[0].keys())
                assert tile_annotations_df.shape[0] + rejected_annotations_df[rejected_annotations_df.image_name == image.file_name].shape[0] + border_annotations + excluded_annotations >= len(image.annotations), "Missing annotations"
            
            #logger.info(f"{image.file_name} : {len(tile_annotations_df)} annotations")
            if ratio_wo_annotation != 0 and len(annotations) == 0:
                # TODO: artifacts of old code. Change to respect the ratio of tiles w/o annotation even with tiles from image w/o annotations
                gt_tiles_df = pd.concat((gt_tiles_df, tiles_df.sample(n=1, random_state=cfg['seed'])), ignore_index=True)
                tot_tiles_without_ann += 1
            else:
                condition_annotations = tiles_df["id"].isin(tile_annotations_df["image_id"].unique())
                clipped_annotations_df = pd.concat([clipped_annotations_df, tile_annotations_df], ignore_index=True)

                # Separate tiles w/ and w/o annotations
                tiles_with_ann_df = tiles_df[condition_annotations]
                tot_tiles_with_ann += tiles_with_ann_df.shape[0]

                gt_tiles_df = pd.concat((gt_tiles_df, tiles_with_ann_df), ignore_index=True)
                selected_tiles = tiles_with_ann_df.file_name.unique().tolist()

                if ratio_wo_annotation != 0:
                    nbr_tiles_without_ann = math.ceil(len(tiles_with_ann_df) * ratio_wo_annotation/(1 - ratio_wo_annotation))
                    tiles_without_ann_df = tiles_df[~condition_annotations]
                    low_tiles_df = select_low_tiles(tiles_without_ann_df, cfg['clipping_params'], excluded_height_ratio=1/2)
                    if len(low_tiles_df) >= nbr_tiles_without_ann:
                        added_empty_tiles_df = low_tiles_df.sample(n=nbr_tiles_without_ann, random_state=cfg['seed'])
                    else:
                        added_empty_tiles_df = pd.concat([
                            low_tiles_df, 
                            tiles_without_ann_df[~tiles_without_ann_df["file_name"].isin(low_tiles_df["file_name"].unique())].sample(
                                n=nbr_tiles_without_ann-len(low_tiles_df), random_state=cfg['seed']
                            )
                        ], ignore_index=True)

                    tot_tiles_without_ann += added_empty_tiles_df.shape[0]
                    gt_tiles_df = pd.concat((gt_tiles_df, added_empty_tiles_df), ignore_index=True)
                    selected_tiles = selected_tiles + added_empty_tiles_df.file_name.unique().tolist()

            tiles_without_ann_df = tiles_df[~(condition_annotations | tiles_df["file_name"].isin(selected_tiles))].copy()
            if not tiles_without_ann_df.empty:
                tiles_without_ann_df = select_low_tiles(tiles_without_ann_df, cfg['clipping_params'], excluded_height_ratio=1/2)
                oth_tiles_df = pd.concat([oth_tiles_df, tiles_without_ann_df], ignore_index=True)
    
    clipped_annotations_df = clipped_annotations_df.reset_index(names=['id'])
    logger.info(f"Found {tot_tiles_with_ann} tiles with annotations and {tot_tiles_without_ann} tiles without annotations for training.")
    return gt_tiles_df, oth_tiles_df, clipped_annotations_df, rejected_annotations_df
 
def main():
    RATIO_WO_ANNOTATIONS = cfg['ratio_wo_annotations']
    MAKE_OTHER_DATASET = cfg['make_other_dataset']
    created_dirs, created_files = [], []
    assert 'coco' in cfg['tasks'] or 'yolo' in cfg['tasks'], "At least one of 'coco' or 'yolo must be in the tasks."

    if 'coco' in cfg['tasks']:
        os.makedirs(cfg['tasks']['coco']['subfolder'], exist_ok=True)
        created_dirs.append(cfg['tasks']['coco']['subfolder'])
    if 'yolo' in cfg['tasks']:
        os.makedirs(cfg['tasks']['yolo']['subfolder'], exist_ok=True)
        created_dirs.append(cfg['tasks']['yolo']['subfolder'])
    
    for aoi, params in cfg['clipping_params'].items():
        logger.info(f"Overlap {round(params['overlap_x']/TILE_SIZE*100,1)}% in X and {round(params['overlap_y']/TILE_SIZE*100,1)}% in Y for {aoi}")

    original_coco_dict = read_coco_files(created_dirs, created_files)
    valid_coco_dict = read_validated_coco_files(original_coco_dict)
    split_datasets(original_coco_dict, valid_coco_dict)

    if all(df.empty for df in valid_coco_dict.values()):
        logger.info("No validated annotations found. Only inference is possible.")
        MAKE_OTHER_DATASET = True
        RATIO_WO_ANNOTATIONS = 0

    gt_tiles_df, oth_tiles_df, clipped_annotations_df, rejected_annotations_df = clip_into_tiles(original_coco_dict, valid_coco_dict, RATIO_WO_ANNOTATIONS)

    # Convert images to tiles
    images_to_tiles_dict = gt_tiles_df.groupby('original_image')['file_name'].apply(list).to_dict()
    gt_tiles_df.drop(columns='original_image', inplace=True)

    Parallel(n_jobs=10, backend="loky")(delayed(image_to_tiles)(
            image, corresponding_tiles, rejected_annotations_df
        ) for image, corresponding_tiles in tqdm.tqdm(images_to_tiles_dict.items(), desc="Converting images to tiles")
    )
    
    if MAKE_OTHER_DATASET:
        logger.info(f"Kept {oth_tiles_df.shape[0]} tiles without annotations in the other dataset.")
        images_to_tiles_dict = oth_tiles_df.groupby('original_image')['file_name'].apply(list).to_dict()
        oth_tiles_df.drop(columns='original_image', inplace=True)

        Parallel(n_jobs=10, backend="loky")(delayed(image_to_tiles)(
                image, corresponding_tiles, pd.DataFrame(columns=rejected_annotations_df.columns)
            ) for image, corresponding_tiles in tqdm.tqdm(images_to_tiles_dict.items(), desc="Converting images to tiles")
        )
    del images_to_tiles_dict


    duplicates = clipped_annotations_df.drop(columns='id').astype({'bbox': str, 'segmentation': str}, copy=True).duplicated()
    if any(duplicates):
        logger.warning(f"Found {duplicates.sum()} duplicated annotations with different ids. Removing them...")
        clipped_annotations_df = clipped_annotations_df[~duplicates].reset_index(drop=True)

    # Create COCO dicts
    dataset_tiles_dict = {
        key: gt_tiles_df[gt_tiles_df["dataset"] == key].drop(columns="dataset").reset_index(drop=True) 
        for key in gt_tiles_df["dataset"].unique()
    }
    if MAKE_OTHER_DATASET:
        dataset_tiles_dict["oth"] = oth_tiles_df.drop(columns="dataset").reset_index(drop=True)

    for dataset in dataset_tiles_dict.keys():
        # Split annotations
        dataset_annotations = clipped_annotations_df[clipped_annotations_df["image_id"].isin(dataset_tiles_dict[dataset]["id"])].copy()
        dataset_annotations = dataset_annotations.astype({"id": int, "category_id": int, "iscrowd": int}, copy=False)
        logger.info(f"Found {len(dataset_annotations)} annotations in the {dataset} dataset.")

        coco_dict = misc.assemble_coco_json(dataset_tiles_dict[dataset], dataset_annotations, cfg['categories'])

        if 'coco' in cfg['tasks']:
            logger.info(f"Creating COCO file for {dataset}")
            with open(os.path.join(cfg['tasks']['coco']['subfolder'], f"COCO_{dataset}.json"), "w") as fp:
                json.dump(coco_dict, fp, indent=4)
            created_files.append(os.path.join(cfg['tasks']['coco']['subfolder'], f"COCO_{dataset}.json"))

        if 'yolo' in cfg['tasks']:
            logger.info(f"Creating COCO file for the annotation transformation to YOLO.")
            dataset_tiles_dict[dataset]["file_name"] = [os.path.basename(f) for f in dataset_tiles_dict[dataset]["file_name"]]
            coco_dict = misc.assemble_coco_json(dataset_tiles_dict[dataset], dataset_annotations, cfg['categories'])
            with open(os.path.join(cfg['tasks']['yolo']['subfolder'], dataset + '.json'), 'w') as fp:
                json.dump(coco_dict, fp, indent=4)
            created_files.append(os.path.join(cfg['tasks']['yolo']['subfolder'], f"{dataset}.json"))

    logger.success("Done! The following files have been created:")
    [logger.success(file) for file in created_files]
    logger.success(f"Tiles were written in {', '.join(created_dirs)}.")
    
        
if __name__ == "__main__":
    tic = time.time()
    logger.info(f"Starting...")
    main()
    logger.info(f"Done in {round(time.time() - tic, 2)} seconds.")