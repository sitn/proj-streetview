import cv2, json, os, sys, time, tqdm, numpy as np, pandas as pd

sys.path.insert(1, 'scripts')
from utils.logging import get_logger
from utils.config import get_config  

logger = get_logger()
cfg = get_config("This script prepares COCO datasets.")

IMAGE_IDS = cfg['image_ids'] if 'image_ids' in cfg.keys() else []
NBR_IMAGES_PER_DATASET = 50
COLORS_DICT = {
    "TP": (0, 255, 0),
    "FP": (247, 195, 79),
    "FN": (37, 168, 249),
    "oth": (0, 0, 0)
}

def read_coco_data():
    logger.info("Reading COCO datasets...")
    images, annotations = [], []
    for dataset in cfg['tagged_coco_files'].keys():
        with open(cfg['tagged_coco_files'][dataset]) as fp:
            coco_data = json.load(fp)
        if isinstance(coco_data, dict):
            images.extend(coco_data['images'])
            annotations.extend(coco_data['annotations'])
            logger.info(f"Dataset '{dataset}' has {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations.")
        else:
            annotations.extend(coco_data)
            for im_dataset in cfg['coco_file_for_images'].keys():
                with open(cfg['coco_file_for_images'][im_dataset]) as fp:
                    image_info = json.load(fp)['images']
                images.extend(image_info)
            logger.info(f"Dataset '{dataset}' has {len(image_info)} images and {len(coco_data)} annotations.")

    images_df = pd.DataFrame.from_records(images).drop_duplicates(subset=["file_name"])
    annotations_df = pd.DataFrame.from_records(annotations)
    return images_df, annotations_df

def main():
    os.makedirs(cfg['output_folder'], exist_ok=True)
    images_df, annotations_df = read_coco_data()

    print(annotations_df)

    if len (IMAGE_IDS) > 0:
        sample_images_df = images_df[images_df['id'].isin(IMAGE_IDS)]
    else:
        sample_images_df = images_df.sample(frac=1, random_state=42)    # sample = shuffle rows to create mixity in output

    if 'id' not in annotations_df.columns:
        annotations_df['id'] = [det_id if det_id == np.nan else label_id for det_id, label_id in zip(annotations_df.det_id, annotations_df.label_id)]

    logger.info("Tagging sample images...")
    if 'tag' in annotations_df.columns and any(annotations_df.tag.isna()):
        score_threshold = annotations_df.loc[annotations_df.tag.notna(), 'score'].min()
        logger.info(f'A threshold of {score_threshold} is applied on the score.')
        annotations_df.loc[annotations_df.tag.isna(), 'tag'] = 'oth'
        annotations_df = annotations_df[annotations_df.score >= score_threshold]

    images_pro_dataset = {key: 0 for key in annotations_df["dataset"].unique()}
    for coco_image in tqdm.tqdm(sample_images_df.itertuples(), desc="Tagging images"):
        if all([im_nbr >= NBR_IMAGES_PER_DATASET for im_nbr in images_pro_dataset.values()]):
            break

        corresponding_annotations = annotations_df[annotations_df["image_id"] == coco_image.id].reset_index(drop=True)
        if corresponding_annotations.empty:
            continue
        dataset = corresponding_annotations.loc[0, 'dataset']
        if images_pro_dataset[dataset] >= NBR_IMAGES_PER_DATASET:
            continue
        images_pro_dataset[dataset] += 1

        input_path = os.path.join(cfg['image_dir'][coco_image.AOI], os.path.basename(coco_image.file_name))
        im = cv2.imread(input_path)
        if im is None:
            logger.warning(f"Image {input_path} not found.")
            continue

        for ann in corresponding_annotations.itertuples():
            color = COLORS_DICT[ann.tag] if 'tag' in corresponding_annotations.columns else  (255, 0, 0)
            bbox = [int(b) for b in ann.bbox]
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)
            text_position = {
                "trn": (bbox[0], bbox[1]-10),
                "val": (bbox[0], bbox[1] + bbox[3] + 20),
                "tst": (bbox[0], bbox[1] + bbox[3] + 20),
                "oth": (bbox[0], bbox[1] + bbox[3] + 20)
            }[ann.dataset]
            txt = f"{ann.dataset} {ann.id} {round(ann.score, 2)} {ann.tag if 'tag' in corresponding_annotations.columns else ''}"
            cv2.putText(im, txt, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        output_filename = f'{dataset}_det_{coco_image.file_name.split("/")[-1]}'.replace('tif', 'png')
        cv2.imwrite(os.path.join(cfg['output_folder'], output_filename), im)

    logger.success(f"Done! {NBR_IMAGES_PER_DATASET*len(images_pro_dataset.keys())} images were tagged and saved in {cfg['output_folder']}")

if __name__ == "__main__":
    tic = time.time()
    main() 
    logger.info(f"Finished in {time.time() - tic:.2f} seconds.")