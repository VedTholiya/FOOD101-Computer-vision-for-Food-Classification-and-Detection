# src/config.py
import torch
import os

# --- General Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "data"
BATCH_SIZE = 32
# Set NUM_WORKERS to 0 if you are on Windows, otherwise os.cpu_count()
NUM_WORKERS = 0 if os.name == 'nt' else (os.cpu_count() if os.cpu_count() is not None else 0)

# --- Classification Model Configuration ---
CLASSIFICATION_LEARNING_RATE = 1e-3
CLASSIFICATION_EPOCHS = 5
CLASSIFICATION_MODEL_SAVE_PATH = "food101_efficientnet_b0.pth"

# --- Detection Configuration ---
DETECTION_RESULTS_DIR = "detection_results"
NUM_IMAGES_TO_DETECT = 5

# --- YOLO Fine-Tuning Configuration ---
ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_API_KEY" # ❗️ PASTE YOUR ROBOFLOW API KEY HERE
FINETUNE_EPOCHS = 5
FINETUNE_IMG_SIZE = 640
FINETUNE_PROJECT_NAME = 'food_detection_project'
FINETUNE_RUN_NAME = 'yolov8n_finetuned'
INFERENCE_RESULTS_DIR = "inference_results"
