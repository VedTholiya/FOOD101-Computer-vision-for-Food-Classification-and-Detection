# src/finetune_yolo.py
import os
import random
from ultralytics import YOLO
from roboflow import Roboflow
from PIL import Image
from torchvision import datasets
from src import config # Import centralized configuration

def run_yolo_finetuning():
    """
    Downloads a custom dataset from Roboflow, fine-tunes a YOLOv8 model,
    and runs inference on random images from the Food101 dataset.
    """
    print("--- Step 1: Download Annotated Food Dataset from Roboflow ---")
    if config.ROBOFLOW_API_KEY == "YOUR_ROBOFLOW_API_KEY" or not config.ROBOFLOW_API_KEY:
        print("❌ Error: Roboflow API key is not set. Please update it in 'src/config.py'.")
        return

    try:
        rf = Roboflow(api_key=config.ROBOFLOW_API_KEY)
        project = rf.workspace("project-ucqoe").project("food-detection-zl4dl")
        version = project.version(2)
        dataset = version.download("yolov8")
        yaml_path = os.path.join(dataset.location, "data.yaml")
        print(f"✅ Dataset downloaded to: {dataset.location}")
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return

    print("\n--- Step 2: Train (Fine-Tune) the YOLOv8 Model ---")
    model = YOLO('yolov8n.pt')
    results = model.train(
        data=yaml_path,
        epochs=config.FINETUNE_EPOCHS,
        imgsz=config.FINETUNE_IMG_SIZE,
        project=config.FINETUNE_PROJECT_NAME,
        name=config.FINETUNE_RUN_NAME,
        exist_ok=True
    )
    print("\n✅ Training complete.")

    print("\n--- Step 3: Validate the Custom Model ---")
    best_model_path = os.path.join(results.save_dir, 'weights/best.pt')
    if not os.path.exists(best_model_path):
        print(f"❌ Could not find best model at path: {best_model_path}")
        return

    print(f"✅ Loading best model from: {best_model_path}")
    model = YOLO(best_model_path)
    metrics = model.val()
    print("\n📊 Validation Metrics:")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  mAP50: {metrics.box.map50:.4f}")

    print("\n--- Step 4: Perform Inference on New Images ---")
    try:
        food101_dataset = datasets.Food101(root=config.DATA_ROOT, split='test', download=True)
    except Exception as e:
        print(f"❌ Could not load Food101 dataset for inference: {e}")
        return

    os.makedirs(config.INFERENCE_RESULTS_DIR, exist_ok=True)
    random_indices = random.sample(range(len(food101_dataset)), k=config.NUM_IMAGES_TO_DETECT)
    print(f"Running inference on {config.NUM_IMAGES_TO_DETECT} random images...")

    for i, idx in enumerate(random_indices):
        img, label = food101_dataset[idx]
        results = model(img)
        im_array = results[0].plot()
        im = Image.fromarray(im_array[..., ::-1])
        result_filename = f"result_{i}_{food101_dataset.classes[label]}.jpg"
        im.save(os.path.join(config.INFERENCE_RESULTS_DIR, result_filename))

    print(f"✅ Inference complete. Results saved to '{config.INFERENCE_RESULTS_DIR}/'")

if __name__ == '__main__':
    run_yolo_finetuning()
