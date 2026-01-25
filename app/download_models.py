
import os
import requests
import zipfile
import io

# Define the models to download
MODELS = {
    "bert-base": "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip",
    "bert-tiny": "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-128_A-2.zip",
    "bert-mini": "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-256_A-4.zip",
    "bert-small": "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-512_A-8.zip",
}

# Define the target directory
TARGET_DIR = "bert_models"

def download_and_unzip(model_name, url, target_dir):
    """Downloads and unzips a model from a URL."""
    model_dir = os.path.join(target_dir, model_name)
    if os.path.exists(model_dir):
        print(f"Model '{model_name}' already exists. Skipping.")
        return

    print(f"Downloading and extracting '{model_name}'...")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(model_dir)
        print(f"Successfully downloaded and extracted '{model_name}'.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading '{model_name}': {e}")
    except zipfile.BadZipFile:
        print(f"Error: Downloaded file for '{model_name}' is not a valid zip file.")

if __name__ == "__main__":
    os.makedirs(TARGET_DIR, exist_ok=True)
    for name, url in MODELS.items():
        download_and_unzip(name, url, TARGET_DIR)
