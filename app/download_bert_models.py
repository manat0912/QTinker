#!/usr/bin/env python3
"""
Download BERT models from Google Cloud Storage to avoid HuggingFace token requirements.
Includes BERT-Large, DistilBERT, and other variants.
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# BERT models repository with direct download URLs from Google Cloud Storage
BERT_MODELS = {
    # BERT-Large Models
    "bert-large-uncased": {
        "url": "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip",
        "size": "~340MB",
        "description": "24-layer, 1024-hidden, 16-heads, 340M parameters"
    },
    "bert-large-cased": {
        "url": "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip",
        "size": "~340MB",
        "description": "24-layer, 1024-hidden, 16-heads, 340M parameters (cased)"
    },
    "bert-large-uncased-wwm": {
        "url": "https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip",
        "size": "~340MB",
        "description": "24-layer, 1024-hidden, 16-heads, Whole Word Masking (uncased)"
    },
    "bert-large-cased-wwm": {
        "url": "https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip",
        "size": "~340MB",
        "description": "24-layer, 1024-hidden, 16-heads, Whole Word Masking (cased)"
    },
    
    # BERT-Base Models
    "bert-base-uncased": {
        "url": "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip",
        "size": "~110MB",
        "description": "12-layer, 768-hidden, 12-heads, 110M parameters"
    },
    "bert-base-cased": {
        "url": "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip",
        "size": "~110MB",
        "description": "12-layer, 768-hidden, 12-heads, 110M parameters (cased)"
    },
    
    # Smaller BERT Models (for distillation)
    "bert-tiny": {
        "url": "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-128_A-2.zip",
        "size": "~10MB",
        "description": "2-layer, 128-hidden, 2-heads (Tiny)"
    },
    "bert-mini": {
        "url": "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-256_A-4.zip",
        "size": "~15MB",
        "description": "4-layer, 256-hidden, 4-heads (Mini)"
    },
    "bert-small": {
        "url": "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-512_A-8.zip",
        "size": "~25MB",
        "description": "4-layer, 512-hidden, 8-heads (Small)"
    },
    "bert-medium": {
        "url": "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-8_H-512_A-8.zip",
        "size": "~50MB",
        "description": "8-layer, 512-hidden, 8-heads (Medium)"
    },
    
    # Multilingual Models
    "bert-multilingual-cased": {
        "url": "https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip",
        "size": "~110MB",
        "description": "104 languages, 12-layer, 768-hidden, 12-heads"
    },
    "bert-chinese": {
        "url": "https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip",
        "size": "~110MB",
        "description": "Chinese Simplified and Traditional, 12-layer, 768-hidden"
    },
}

# DistilBERT models from GitHub (transformers library)
DISTILBERT_MODELS = {
    "distilbert-base-uncased": {
        "repo": "https://huggingface.co/distilbert/distilbert-base-uncased",
        "description": "DistilBERT base model (uncased)"
    },
    "distilbert-base-cased": {
        "repo": "https://huggingface.co/distilbert/distilbert-base-cased",
        "description": "DistilBERT base model (cased)"
    },
    "distilbert-base-multilingual-cased": {
        "repo": "https://huggingface.co/distilbert/distilbert-base-multilingual-cased",
        "description": "DistilBERT multilingual model"
    },
}


def download_file(url: str, output_path: str, model_name: str) -> bool:
    """
    Download a file from URL with progress tracking.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        model_name: Name of the model for logging
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\n📥 Downloading {model_name}...")
        print(f"   URL: {url}")
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download with progress
        def reporthook(blocknum, blocksize, totalsize):
            downloaded = blocknum * blocksize
            if totalsize > 0:
                percent = min(downloaded * 100 / totalsize, 100)
                print(f"   Progress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f}MB / {totalsize / 1024 / 1024:.1f}MB)", end='\r')
        
        urllib.request.urlretrieve(url, output_path, reporthook)
        print(f"\n   ✓ Successfully downloaded {model_name}")
        return True
        
    except Exception as e:
        print(f"\n   ✗ Failed to download {model_name}: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str, model_name: str) -> bool:
    """
    Extract a zip file and handle the extracted directory structure.
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to
        model_name: Name of the model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"📦 Extracting {model_name}...")
        os.makedirs(extract_to, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        # Check if extraction created a single subdirectory and move contents up
        contents = os.listdir(extract_to)
        if len(contents) == 1 and os.path.isdir(os.path.join(extract_to, contents[0])):
            subdir = os.path.join(extract_to, contents[0])
            # Move all contents from subdirectory to parent
            for item in os.listdir(subdir):
                src = os.path.join(subdir, item)
                dst = os.path.join(extract_to, item)
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                shutil.move(src, dst)
            # Remove the now-empty subdirectory
            os.rmdir(subdir)
        
        print(f"   ✓ Extracted {model_name}")
        return True
        
    except Exception as e:
        print(f"   ✗ Failed to extract {model_name}: {e}")
        return False


def download_bert_large_models(models_dir: str = "bert_models") -> Dict[str, bool]:
    """
    Download all BERT-Large models.
    
    Args:
        models_dir: Directory to store models
        
    Returns:
        Dictionary with download results
    """
    results = {}
    bert_models_path = Path(models_dir) / "bert_large"
    bert_models_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("🤖 DOWNLOADING BERT-LARGE MODELS")
    print("="*60)
    
    for model_name, model_info in BERT_MODELS.items():
        if "large" in model_name or "wwm" in model_name:
            model_path = bert_models_path / model_name
            zip_path = str(model_path) + ".zip"
            
            # Download
            if download_file(model_info["url"], zip_path, model_name):
                # Extract
                if extract_zip(zip_path, str(model_path), model_name):
                    results[model_name] = True
                    # Clean up zip file
                    try:
                        os.remove(zip_path)
                    except:
                        pass
                else:
                    results[model_name] = False
            else:
                results[model_name] = False
    
    return results


def download_distilbert_models(models_dir: str = "bert_models") -> Dict[str, bool]:
    """
    Download DistilBERT models information.
    
    Args:
        models_dir: Directory to store models
        
    Returns:
        Dictionary with model information
    """
    results = {}
    distil_models_path = Path(models_dir) / "distilbert"
    distil_models_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("🔥 DISTILBERT MODELS")
    print("="*60)
    print("\nDistilBERT models can be downloaded from HuggingFace Hub or converted locally.")
    print("To use DistilBERT with transformers library:")
    print("  from transformers import AutoModel")
    print("  model = AutoModel.from_pretrained('distilbert-base-uncased')")
    print("\nAvailable DistilBERT models:")
    
    for model_name, model_info in DISTILBERT_MODELS.items():
        print(f"  • {model_name}")
        print(f"    {model_info['description']}")
        results[model_name] = True
    
    return results


def download_smaller_bert_models(models_dir: str = "bert_models") -> Dict[str, bool]:
    """
    Download smaller BERT models useful for distillation.
    
    Args:
        models_dir: Directory to store models
        
    Returns:
        Dictionary with download results
    """
    results = {}
    bert_models_path = Path(models_dir) / "bert_small"
    bert_models_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("📦 DOWNLOADING SMALLER BERT MODELS (for distillation)")
    print("="*60)
    
    for model_name, model_info in BERT_MODELS.items():
        if any(x in model_name for x in ["tiny", "mini", "small", "medium"]):
            model_path = bert_models_path / model_name
            zip_path = str(model_path) + ".zip"
            
            # Download
            if download_file(model_info["url"], zip_path, model_name):
                # Extract
                if extract_zip(zip_path, str(model_path), model_name):
                    results[model_name] = True
                    # Clean up zip file
                    try:
                        os.remove(zip_path)
                    except:
                        pass
                else:
                    results[model_name] = False
            else:
                results[model_name] = False
    
    return results


def create_model_registry(models_dir: str = "bert_models"):
    """
    Create a registry file documenting all downloaded models.
    
    Args:
        models_dir: Directory containing models
    """
    registry_path = Path(models_dir) / "MODEL_REGISTRY.md"
    
    registry_content = """# BERT Models Registry

This directory contains locally downloaded BERT models to avoid HuggingFace token requirements.

## BERT-Large Models

These are the full BERT models, useful as teacher models for knowledge distillation.

"""
    
    for model_name, model_info in BERT_MODELS.items():
        if "large" in model_name or "wwm" in model_name:
            registry_content += f"### {model_name}\n"
            registry_content += f"- **Description**: {model_info['description']}\n"
            registry_content += f"- **Size**: {model_info['size']}\n"
            registry_content += f"- **Path**: `bert_large/{model_name}/`\n\n"
    
    registry_content += """## Smaller BERT Models

These are distilled/smaller versions, useful as student models or for resource-constrained environments.

"""
    
    for model_name, model_info in BERT_MODELS.items():
        if any(x in model_name for x in ["tiny", "mini", "small", "medium"]):
            registry_content += f"### {model_name}\n"
            registry_content += f"- **Description**: {model_info['description']}\n"
            registry_content += f"- **Size**: {model_info['size']}\n"
            registry_content += f"- **Path**: `bert_small/{model_name}/`\n\n"
    
    registry_content += """## DistilBERT Models

DistilBERT models are not downloaded directly but can be used via transformers library.
They are pre-distilled versions of BERT that are ~40% smaller and ~60% faster.

"""
    
    for model_name, model_info in DISTILBERT_MODELS.items():
        registry_content += f"### {model_name}\n"
        registry_content += f"- **Description**: {model_info['description']}\n\n"
    
    registry_content += """## Loading Models in Python

### Using Local BERT Models
```python
from transformers import AutoTokenizer, AutoModel
import os

model_path = "bert_models/bert_large/bert-large-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
```

### Using DistilBERT (automatic download)
```python
from transformers import AutoTokenizer, AutoModel

# This will download from HuggingFace Hub on first use
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')
```

## Model Details

### BERT-Large
- 24 Transformer layers
- 1024 hidden units
- 16 attention heads
- 340M parameters
- Best for: Teacher models in knowledge distillation, fine-tuning on large datasets

### BERT-Small
- 4 Transformer layers
- 512 hidden units
- 8 attention heads
- Good for: Student models, efficient inference

### BERT-Tiny/Mini
- 2-4 Transformer layers
- 128-256 hidden units
- Good for: Mobile/edge deployment, extremely resource-constrained environments

## Notes

- Models are stored uncompressed for faster loading
- Each model directory contains: `bert_config.json`, `vocab.txt`, and checkpoint files
- No HuggingFace token required - all models downloaded from public Google Cloud Storage
"""
    
    with open(registry_path, 'w') as f:
        f.write(registry_content)
    
    print(f"\n📋 Created model registry at {registry_path}")


def main():
    """Main entry point for downloading models."""
    models_dir = "bert_models"
    
    print("\n" + "="*60)
    print("🤗 BERT MODELS DOWNLOADER (No HuggingFace Token Required)")
    print("="*60)
    
    # Download models
    print("\n📊 Starting BERT models download...")
    
    # Download BERT-Large models
    large_results = download_bert_large_models(models_dir)
    
    # Download smaller models
    small_results = download_smaller_bert_models(models_dir)
    
    # Show DistilBERT info
    distil_results = download_distilbert_models(models_dir)
    
    # Create registry
    create_model_registry(models_dir)
    
    # Summary
    print("\n" + "="*60)
    print("📊 DOWNLOAD SUMMARY")
    print("="*60)
    
    all_results = {**large_results, **small_results, **distil_results}
    successful = sum(1 for v in all_results.values() if v)
    total = len(all_results)
    
    print(f"\n✓ Successfully processed: {successful}/{total} models")
    
    if successful > 0:
        print(f"\n✅ Models ready in: {os.path.abspath(models_dir)}/")
        print(f"\n📖 See MODEL_REGISTRY.md for model details and usage instructions")
    
    return all_results


if __name__ == "__main__":
    main()
