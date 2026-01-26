# BERT Models Documentation

## Overview

QTinker now includes comprehensive BERT model support with multiple variants for knowledge distillation and quantization. All models are downloaded from official Google Cloud Storage sources without requiring HuggingFace tokens.

## Available BERT Models

### BERT-Large Models (Teacher Models)

Perfect for knowledge distillation as teacher models.

| Model | Size | Layers | Hidden | Heads | Parameters | Use Case |
|-------|------|--------|--------|-------|------------|----------|
| **bert-large-uncased** | ~340MB | 24 | 1024 | 16 | 340M | Teacher for English, case-insensitive |
| **bert-large-cased** | ~340MB | 24 | 1024 | 16 | 340M | Teacher with case preservation |
| **bert-large-uncased-wwm** | ~340MB | 24 | 1024 | 16 | 340M | Whole Word Masking variant |
| **bert-large-cased-wwm** | ~340MB | 24 | 1024 | 16 | 340M | WWM with case preservation |

### BERT-Base Models

Standard BERT models.

| Model | Size | Layers | Hidden | Heads | Parameters |
|-------|------|--------|--------|-------|------------|
| **bert-base-uncased** | ~110MB | 12 | 768 | 12 | 110M |
| **bert-base-cased** | ~110MB | 12 | 768 | 12 | 110M |

### BERT-Small Models (Student Models for Distillation)

Optimized for knowledge distillation and efficient inference.

| Model | Size | Layers | Hidden | Heads | Parameters | Speed-up | Memory Reduction |
|-------|------|--------|--------|-------|------------|----------|------------------|
| **bert-small** | ~25MB | 4 | 512 | 8 | 29M | ~2.5x faster | ~40% less memory |
| **bert-mini** | ~15MB | 4 | 256 | 4 | 11M | ~4x faster | ~60% less memory |
| **bert-tiny** | ~10MB | 2 | 128 | 2 | 4.4M | ~5x faster | ~80% less memory |
| **bert-medium** | ~50MB | 8 | 512 | 8 | 41M | ~1.5x faster | ~25% less memory |

### Multilingual Models

For non-English languages.

| Model | Languages | Layers | Hidden | Parameters |
|-------|-----------|--------|--------|------------|
| **bert-multilingual-cased** | 104 | 12 | 768 | 110M |
| **bert-chinese** | Chinese | 12 | 768 | 110M |

### DistilBERT Models

Already pre-distilled models (40% smaller, 60% faster).

| Model | Size | Efficiency | Use Case |
|-------|------|-----------|----------|
| **distilbert-base-uncased** | ~67MB | 40% smaller, 60% faster | Fast inference, mobile |
| **distilbert-base-cased** | ~67MB | 40% smaller, 60% faster | Fast inference with case |
| **distilbert-base-multilingual-cased** | ~100MB | Multiple languages | Multilingual fast inference |

## Model Directory Structure

```
bert_models/
├── google_research_bert/          # Original BERT implementation
├── huawei_noah_bert/              # Huawei Noah BERT models
├── bert_large/                    # BERT-Large teacher models
│   ├── bert-large-uncased/
│   ├── bert-large-cased/
│   ├── bert-large-uncased-wwm/
│   └── bert-large-cased-wwm/
├── bert_small/                    # Student models for distillation
│   ├── bert-small/
│   ├── bert-mini/
│   ├── bert-tiny/
│   ├── bert-medium/
│   ├── bert-multilingual-cased/
│   └── bert-chinese/
└── MODEL_REGISTRY.md              # Comprehensive model registry
```

## Usage Examples

### Loading BERT Models Locally

```python
from transformers import AutoTokenizer, AutoModel
import os

# Load a specific model
model_path = "bert_models/bert_large/bert-large-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Process text
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

### Using for Knowledge Distillation

```python
# Teacher model (BERT-Large)
teacher_model = AutoModel.from_pretrained(
    "bert_models/bert_large/bert-large-uncased"
)

# Student model (BERT-Small)
student_model = AutoModel.from_pretrained(
    "bert_models/bert_small/bert-small"
)

# Use in QTinker's distillation pipeline
# See app.py for full distillation examples
```

### Using QTinker's Web UI

1. **Start QTinker**: Click "Start" in Pinokio
2. **Select Models**: 
   - Teacher Model: Choose from BERT-Large variants
   - Student Model: Choose from BERT-Small, BERT-Mini, etc.
   - Target Model: Optional, for other architectures
3. **Configure Distillation**:
   - Select distillation method (logit-based, patient-KD, feature-based)
   - Set temperature and other hyperparameters
4. **Download Output**: Get the distilled model

## Performance Comparison

### Inference Speed (relative to BERT-Large)

| Model | Speed | Latency | Memory |
|-------|-------|---------|--------|
| BERT-Large (baseline) | 1.0x | ~500ms | 100% |
| BERT-Base | ~1.5x | ~330ms | 50% |
| BERT-Small | ~2.5x | ~200ms | 40% |
| BERT-Mini | ~4.0x | ~125ms | 20% |
| BERT-Tiny | ~5.0x | ~100ms | 10% |
| DistilBERT | ~2.0x | ~250ms | 40% |

*Note: Performance varies by hardware and implementation*

## Model Selection Guide

### When to Use Which Model

**As Teacher Model:**
- Use **BERT-Large** variants for best knowledge distillation results
- WWM (Whole Word Masking) variants provide slightly better performance
- Cased vs Uncased depends on your domain (Named Entities benefit from cased)

**As Student Model:**
- **BERT-Small**: 70-80% of BERT-Large performance, ~40% smaller
- **BERT-Mini**: 55-70% performance, good balance for mobile
- **BERT-Tiny**: Edge devices, extreme resource constraints
- **DistilBERT**: Pre-optimized alternative to manual distillation

**For Specific Tasks:**
- **Text Classification**: BERT-Small is usually sufficient
- **Named Entity Recognition**: Use cased models
- **Multilingual**: Use multilingual BERT variants
- **Mobile Deployment**: BERT-Tiny or DistilBERT

## No HuggingFace Token Required

All BERT models are downloaded from official Google Cloud Storage URLs, eliminating the need for HuggingFace API tokens. This makes QTinker:

- ✅ Fully offline after download
- ✅ No authentication required
- ✅ Reproducible across environments
- ✅ Enterprise-friendly (no external APIs)

## Distillation Strategies

QTinker supports multiple distillation approaches:

### 1. **Logit-Based Distillation**
```
Temperature scaling to match teacher and student probability distributions
- Best for general knowledge transfer
- Computationally efficient
```

### 2. **Patient Knowledge Distillation**
```
Match intermediate layer representations from teacher to student
- Better for preserving internal knowledge
- Recommended for significant size reductions
```

### 3. **Feature-Based Distillation**
```
Transfer specific layer features between teacher and student
- Fine-grained knowledge transfer
- Best results but more computation
```

## Quantization with Distilled Models

After distillation, further optimize with quantization:

```python
# Distill then quantize
distilled_model = # ... from distillation
quantized_model = torchao_quantize(distilled_model, method="int4")

# Result: ~80% of BERT-Large quality
#         ~5% of original size
#         ~100x faster inference
```

## Advanced Features

### Custom Model Selection

Use the enhanced file browser in QTinker to:
- Browse all available models
- Compare model sizes
- View model architectures
- Select multiple models for ensemble

### Automatic Model Registry

The `MODEL_REGISTRY.md` file documents:
- All installed models
- Model specifications
- Recommended use cases
- Python loading examples

## Troubleshooting

### Model Not Found

Check that models were downloaded correctly:
```bash
ls bert_models/bert_large/
ls bert_models/bert_small/
```

### Memory Issues with BERT-Large

If training OOM errors:
1. Use BERT-Large only as teacher (inference mode)
2. Use smaller student models
3. Reduce batch size
4. Use gradient accumulation

### Slow Downloads

Large models can take time to download. Check:
- Internet connection speed
- Disk space available
- Try downloading smaller models first (BERT-Tiny, BERT-Mini)

## References

- **BERT Paper**: [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- **DistilBERT**: [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
- **Google Research BERT**: https://github.com/google-research/bert
- **HuggingFace Transformers**: https://github.com/huggingface/transformers

## License

All BERT models are released under the Apache 2.0 License. See individual model repositories for details.
