import json, os, sys, tqdm, time, pandas as pd, ultralytics

sys.path.insert(1, 'scripts')
from utils.logging import get_logger
from utils.config import get_config   

logger = get_logger()
cfg = get_config("This script makes detections with a given YOLO model.")

def yolo_to_coco_annotations(result, image_id, image_file):
    annotations = []
    for det_index in range(len(result.boxes.cls)):
        category_id = int(result.boxes.cls[det_index])

        score = round(result.boxes.conf[det_index].tolist(), 3)
        area = int(result.masks.data[det_index].sum().tolist())
        if area == 0:
            logger.warn(f"Found an empty mask with score {score}...")
            continue

        annotations.append({
            "image_id": image_id,
            "bbox": [int(coord) for coord in result.boxes.xywh[det_index].tolist()],
            "area": area,
            "score": score,
            "det_class": category_id,
            "segmentation": [[int(coord) for coord in result.masks.xy[det_index].flatten().tolist()]],
            "file_name": image_file
        })
    return annotations

def main():
    logger.info(f'Working with the model "{cfg['model']}"...')
    os.makedirs(cfg['output_dir'], exist_ok=True)
    written_files = []
    for dataset, path in cfg['dataset_images_folder'].items():
        logger.info(f"Working on dataset {dataset}...")

        STREAM_MODE = True
        BATCH_SIZE = 64
        HALF = True
        results = ultralytics.YOLO(cfg['model'])(path, conf=0.05, half=HALF, imgsz=cfg['tile_size'], batch=BATCH_SIZE, retina_masks=True, project=cfg['output_dir'], exist_ok=True, verbose=False, stream=STREAM_MODE)

        with open(os.path.join(cfg['image_infos'], f'{dataset}.json'), 'r') as fp:
            image_infos_dict = json.load(fp)['images']
        images_infos_df = pd.DataFrame.from_records(image_infos_dict)[['file_name', 'id']]
        images_infos_df.set_index('file_name', inplace=True)

        coco_detections = []
        for result in tqdm.tqdm(results, desc=f"Performing inference on {dataset}"):
            file_name = os.path.basename(result.path)
            coco_detections.append(yolo_to_coco_annotations(result, int(images_infos_df.at[file_name, 'id']), file_name))

        logger.info(f"Inference of dataset {dataset} written in {cfg['output_dir']}")

        flat_coco_detections = [item for sublist in coco_detections for item in sublist]
        logger.success(f"Done! {len(flat_coco_detections)} annotations were produced.")
        for i in tqdm.tqdm(range(len(flat_coco_detections)), desc="Assigning IDs"):
            flat_coco_detections[i]['det_id'] = i

        filepath = os.path.join(cfg['output_dir'], f'inference_{dataset}.json')
        logger.info(f"Saving annotations to {filepath}...")
        with open(filepath, 'w') as fp:
            json.dump(flat_coco_detections, fp)
        written_files.append(filepath)

    logger.success(f"The following files were written:")
    [logger.success(filepath) for filepath in written_files]


if __name__ == "__main__":
    tic = time.time()
    main()
    logger.success(f"Done in {round(time.time() - tic, 2)} seconds.")