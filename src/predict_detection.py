# src/predict_detection.py
from torchvision import datasets
from ultralytics import YOLO
import random
import os
from src import config # Import centralized configuration

def perform_detection():
    """
    Loads a pretrained YOLOv8 model and performs object detection on random
    images from the Food101 test set.
    """
    print("--- Performing Object Detection with Pre-trained YOLOv8 ---")

    os.makedirs(config.DETECTION_RESULTS_DIR, exist_ok=True)

    try:
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8 model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        print("Please ensure you have an internet connection to download the model.")
        return

    print("⏳ Loading Food101 test set to get sample images...")
    try:
        test_dataset = datasets.Food101(root=config.DATA_ROOT, split='test', download=True)
        print("✅ Dataset loaded.")
    except Exception as e:
        print(f"❌ Error loading Food101 dataset: {e}")
        return

    if len(test_dataset.images) == 0:
        print("❌ No images found in the dataset.")
        return

    random_indices = random.sample(range(len(test_dataset.images)), k=config.NUM_IMAGES_TO_DETECT)
    image_paths_to_test = [test_dataset.images[i] for i in random_indices]

    print(f"⏳ Running detection on {config.NUM_IMAGES_TO_DETECT} random images...")
    for img_path in image_paths_to_test:
        full_img_path = os.path.join(config.DATA_ROOT, "food-101", "images", f"{img_path}.jpg")

        if not os.path.exists(full_img_path):
            print(f"⚠️ Warning: Image path not found: {full_img_path}. Skipping.")
            continue

        try:
            results = model(full_img_path)
            base_filename = os.path.basename(full_img_path)
            save_path = os.path.join(config.DETECTION_RESULTS_DIR, f"detected_{base_filename}")
            results[0].save(filename=save_path)
            print(f"  -> Detection result saved to: {save_path}")
        except Exception as e:
            print(f"❌ An error occurred during detection for image {full_img_path}: {e}")

    print(f"\n✅ Detection process complete. Check the '{config.DETECTION_RESULTS_DIR}' folder.")

if __name__ == '__main__':
    perform_detection()
