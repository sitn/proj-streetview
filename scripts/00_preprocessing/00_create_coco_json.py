import json, os, tqdm
from PIL import Image

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def add_missing_images_to_coco(coco_json_path, image_dir, output_json_path):
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    # Existing filenames and max image ID
    existing_filenames = {img["file_name"] for img in images}
    max_image_id = max((img["id"] for img in images), default=0)

    for filename in tqdm.tqdm(sorted(os.listdir(image_dir))):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in VALID_EXTENSIONS:
            continue
        if filename in existing_filenames:
            continue

        try:
            with Image.open(os.path.join(image_dir, filename)) as img:
                width, height = img.size
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        max_image_id += 1
        images.append({
            "id": max_image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

    with open(output_json_path, "w") as f:
        json.dump({"images":images, "annotations":annotations, "categories":categories}, f, indent=2)

    print(f"Saved updated COCO dataset to: {output_json_path}")


if __name__ == "__main__":
    add_missing_images_to_coco(
        coco_json_path="./data/NE/NE_ground_truth_COCO.json",
        image_dir="./data/NE/images/",
        output_json_path="./data/NE/NE_ground_truth_COCO_all.json"
    )
