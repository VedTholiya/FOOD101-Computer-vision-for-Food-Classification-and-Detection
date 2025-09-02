# src/download_dataset.py
from torchvision import datasets
from src import config
import os

def download_food101(root_dir):
    """
    Downloads the Food101 dataset if it doesn't already exist.

    Args:
        root_dir (str): The root directory to download the data to.
    """
    print("--- Checking for Food101 dataset ---")

    # The dataset creates a 'food-101' subdirectory.
    if os.path.exists(os.path.join(root_dir, "food-101")):
        print("✅ Food101 dataset already found. Skipping download.")
        return

    print("⏳ Downloading Food101 training set (this may take a while)...")
    datasets.Food101(root=root_dir,
                     split="train",
                     download=True)

    print("\n⏳ Downloading Food101 test set...")
    datasets.Food101(root=root_dir,
                     split="test",
                     download=True)

    print("\n✅ Dataset download complete.")

if __name__ == "__main__":
    # Ensure the root data directory exists
    os.makedirs(config.DATA_ROOT, exist_ok=True)
    download_food101(root_dir=config.DATA_ROOT)
