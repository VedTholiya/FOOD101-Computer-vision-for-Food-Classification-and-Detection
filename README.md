# FOOD101-Computer-vision-for-Food-Classification-and-Detection
Food101 consists of 101 food categories, with 101'000 images. For each class, 250 manually reviewed test images are provided as well as 750 training images. On purpose, the training images were not cleaned, and thus still contain some amount of noise.
# Food Image Classification and Detection Project

This project provides tools to work with the Food101 dataset. It includes scripts for:
1.  **Image Classification**: Training an `EfficientNet-B0` model to classify 101 different food categories.
2.  **Object Detection (Pre-trained)**: Using a pre-trained `YOLOv8` model to detect objects in food images.
3.  **Object Detection (Fine-tuned)**: Fine-tuning a `YOLOv8` model on a custom annotated food dataset from Roboflow.

## 🏛️ Repository Structure

```
food-vision-project/
├── .gitignore          # Ignores unnecessary files
├── README.md           # This instruction file
├── requirements.txt    # Project dependencies
└── src/
    ├── __init__.py     # Makes 'src' a Python package
    ├── config.py       # Central configuration for all scripts
    ├── download_dataset.py  # Script to download the Food101 dataset
    ├── train_classifier.py  # Script to train the classification model
    ├── predict_detection.py # Script to run object detection with pre-trained YOLO
    └── finetune_yolo.py     # Script to fine-tune YOLOv8 on a custom dataset
```

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### **Prerequisites**

* Python 3.8 or higher
* PyTorch with CUDA support (recommended for GPU acceleration)

### **Setup Instructions**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/VedTholiya/FOOD101-Computer-vision-for-Food-Classification-and-Detection/tree/main
    cd food-vision-project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♀️ How to Run

All commands should be run from the root directory (`food-vision-project/`).

### **Step 1: Download the Dataset**

First, download the Food101 dataset. This will create a `data/` directory.

```bash
python src/download_dataset.py
```
> **Note**: The dataset is approximately 5GB. This may take a while.

### **Step 2: Train the Image Classifier**

This script trains an `EfficientNet-B0` model on the Food101 dataset and saves the best model weights to `food101_efficientnet_b0.pth`.

```bash
python src/train_classifier.py
```
The script will print the training progress for each epoch.

### **Step 3: Run Object Detection (Pre-trained YOLOv8)**

This script uses a standard, pre-trained `YOLOv8` model to perform object detection on 5 random images from the Food101 test set. The results are saved in the `detection_results/` folder.

```bash
python src/predict_detection.py
```

### **Step 4: Fine-Tune YOLOv8 (Optional)**

This script fine-tunes a `YOLOv8` model on a custom annotated food dataset from Roboflow.

**⚠️ Important:** You need a free [Roboflow account](https://roboflow.com/) and an API key for this step.

1.  **Set your API Key**: Open `src/finetune_yolo.py` and replace `"YOUR_ROBOFLOW_API_KEY"` with your actual key.

2.  **Run the fine-tuning script:**
    ```bash
    python src/finetune_yolo.py
    ```
This will download the dataset, train the model, and save the results (including weights and validation metrics) in the `food_detection_project/` directory. It will also run inference on 5 random Food101 images and save them to `inference_results/`.

## ⚙️ Configuration

You can modify hyperparameters and settings for all scripts in one place: `src/config.py`. This includes batch size, learning rate, number of epochs, file paths, etc.
