import os, json, sys, shutil, tqdm, time, pandas as pd, multiprocessing as mp
from ultralytics.data.converter import convert_coco

sys.path.insert(1, 'scripts')
from utils.logging import get_logger
from utils.config import get_config

logger = get_logger()
cfg = get_config("This script prepares YOLO datasets from the COCO ones and copy images between datasets.")

def coco_to_yolo():
    if os.path.exists(cfg['output_folder']):
        shutil.rmtree(cfg['output_folder'])
    # cf. https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/converter.py#L228
    convert_coco(labels_dir=cfg['tiles_folder'], save_dir=cfg['output_folder'], use_segments=True)
    logger.success(f"YOLO dataset was created in {cfg['output_folder']}.")

def copy_image(args):
    file_name, tiles_folder, dataset_dir = args
    image_path = os.path.join(tiles_folder, os.path.basename(file_name))
    dest_path = os.path.join(dataset_dir, os.path.basename(file_name))
    shutil.copy(image_path, dest_path)

def copy_images():
    output_folder_images = os.path.join(cfg['output_folder'], 'images')
    os.makedirs(output_folder_images, exist_ok=True)

    for dataset, coco_file in cfg['datasets_filenames'].items():
        dataset_dir = os.path.join(output_folder_images, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        with open(os.path.join(cfg['tiles_folder'], coco_file)) as fp:
            images_df = pd.DataFrame(json.load(fp)['images']).drop_duplicates(subset=["file_name"])

        args = [(row['file_name'], cfg['tiles_folder'], dataset_dir) for row in images_df.to_dict(orient="records")]
        with mp.Pool(processes=min(8, mp.cpu_count())) as pool:
            for _ in tqdm.tqdm(
                pool.imap_unordered(copy_image, args),
                total=len(args),
                desc=f"Copying images for {dataset} dataset"
            ): pass
    logger.success(f"Datasets were created in {output_folder_images}.")

        
if __name__ == "__main__":
    tic = time.time()
    logger.info(f"Converting coco tiles to yolo tiles...")
    coco_to_yolo()
    logger.info(f"Copy tiles images...")
    copy_images()
    logger.info(f"Done in {round(time.time() - tic, 2)} seconds.")