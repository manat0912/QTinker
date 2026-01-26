# Quick Start: Using BERT Models with QTinker

## Installation

The BERT models are automatically downloaded during installation. No HuggingFace token required!

```bash
# In Pinokio, click "Install"
# Models will be downloaded to: app/bert_models/
```

## Step-by-Step Guide

### 1. Start QTinker

Click the **"Start"** button in Pinokio to launch the web UI.

### 2. Select Models

The web interface provides model selection with:

**Teacher Model** (for knowledge distillation)
- bert-large-uncased *(recommended)*
- bert-large-cased
- bert-large-uncased-wwm
- bert-large-cased-wwm
- bert-base-uncased

**Student Model** (target size/speed)
- bert-small *(recommended for 40% size reduction)*
- bert-mini *(for 60% size reduction)*
- bert-tiny *(for extreme edge devices)*
- bert-medium *(for slight optimization)*

### 3. Choose Distillation Method

**Logit-based Distillation** (Default)
```
Recommended for: Most use cases, balanced results
Temperature: 4.0 (default)
Alpha: 0.5 (default, weight between distillation and task loss)
```

**Patient Knowledge Distillation**
```
Recommended for: Significant size reductions (Large → Tiny)
Matches intermediate layers between teacher and student
Layer mappings: Automatically configured
```

**Feature-based Distillation**
```
Recommended for: Fine-grained knowledge preservation
Transfers specific layer features
More computation required
```

### 4. Configure Hyperparameters

```python
# Example configuration
{
    "temperature": 4.0,           # Controls softness of output
    "alpha": 0.5,                 # Distillation loss weight
    "learning_rate": 2e-5,        # Fine-tuning learning rate
    "num_epochs": 3,              # Training epochs
    "batch_size": 32,             # Batch size
}
```

### 5. Download & Export

After distillation completes:
1. Model is saved to `app/distilled/`
2. Download the `.pth` or `.safetensors` file
3. Use in your application

## Python Usage Examples

### Load a Teacher Model (BERT-Large)

```python
from transformers import AutoTokenizer, AutoModel

# Load from local path
model_path = "bert_models/bert_large/bert-large-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Use for inference
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
```

### Load a Student Model (BERT-Small)

```python
from transformers import AutoTokenizer, AutoModel

# Load from local path
model_path = "bert_models/bert_small/bert-small"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
```

### Fine-tune a Distilled Model

```python
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
import torch

model_path = "distilled/bert_small_distilled.safetensors"
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Your training setup here
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    # ... add train_dataset, eval_dataset
)

trainer.train()
```

### Quantize a Distilled Model

```python
import torch
from transformers import AutoModel

# Load distilled model
model = AutoModel.from_pretrained("distilled/bert_small_distilled.safetensors")

# Quantize to INT4
import torchao
model = torch.ao.quantization.quantize_dynamic(
    model,
    qconfig_spec=torchao.int4_weight_only(),
)

# Save quantized model
torch.save(model.state_dict(), "bert_small_distilled_int4.pt")
```

## Common Workflows

### Workflow 1: Quick Distillation

For fastest results with reasonable quality:

1. **Teacher**: bert-large-uncased
2. **Student**: bert-small
3. **Method**: Logit-based (default)
4. **Epochs**: 1-3
5. **Result**: ~40% smaller, keeps 85-90% of quality

**Time**: ~30 minutes on GPU

### Workflow 2: Aggressive Compression

For extreme size reduction (edge devices):

1. **Teacher**: bert-large-uncased
2. **Student**: bert-tiny
3. **Method**: Patient Knowledge Distillation
4. **Epochs**: 5-10
5. **Result**: ~97% smaller, keeps 70-75% of quality

**Time**: ~2-4 hours on GPU

### Workflow 3: Fine-grained Optimization

For maximum quality retention:

1. **Teacher**: bert-large-uncased-wwm
2. **Student**: bert-small
3. **Method**: Feature-based Distillation
4. **Epochs**: 5-10
5. **Result**: ~40% smaller, keeps 90-95% of quality

**Time**: ~4-6 hours on GPU

## Model Comparison

### Speed & Size Trade-offs

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| BERT-Large | 100% | 1.0x | 100% | Teacher model |
| BERT-Base | 50% | 1.5x | 98% | Baseline |
| BERT-Small | 40% | 2.5x | 85-90% | Mobile, edge |
| BERT-Mini | 20% | 4.0x | 70-80% | Extreme edge |
| BERT-Tiny | 10% | 5.0x | 60-70% | Ultra-light |

### After Distillation

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| Large-Small | 40% | 2.5x | 90-95% | Production |
| Large-Mini | 20% | 4.0x | 80-85% | Mobile |
| Large-Tiny | 10% | 5.0x | 70-75% | IoT devices |

## Tips & Tricks

### 1. Choosing Temperature

- **Higher temperature** (6-8): Softer probability distributions, more knowledge transfer
- **Lower temperature** (2-3): Sharper distributions, closer to task-specific loss

```python
temperature = 4.0  # Start here, adjust based on results
```

### 2. Layer Mapping for Patient-KD

Automatically handled, but you can customize:

```python
# Example: Map 4-layer student to 12-layer teacher
layer_map = {
    0: 0,      # student layer 0 → teacher layer 0
    1: 4,      # student layer 1 → teacher layer 4
    2: 8,      # student layer 2 → teacher layer 8
    3: 11,     # student layer 3 → teacher layer 11
}
```

### 3. Batch Size Considerations

- **GPU VRAM**: Larger batches (64-128) for better gradients
- **CPU/Memory**: Smaller batches (8-16) to fit in memory
- **Quality**: Larger batches usually give better results

### 4. Learning Rate Schedule

- **Distillation**: 1e-5 to 5e-5 (lower than fine-tuning)
- **Combined loss**: 2e-5 (default)
- **Warmup**: First 10% of steps

## Troubleshooting

### Q: Models downloading very slowly
**A:** Large models (340MB) can take time over slow connections. BERT-Tiny and BERT-Mini download much faster.

### Q: Out of Memory (OOM) error
**A:** 
1. Reduce batch size (32 → 16 or 8)
2. Use gradient accumulation
3. Use a smaller student model
4. Use CPU offloading in `accelerate`

### Q: Distillation accuracy is low
**A:**
1. Increase temperature (4.0 → 6.0)
2. Increase epochs (3 → 5 or more)
3. Reduce alpha (0.5 → 0.3) to focus on distillation loss
4. Try Patient-KD instead of logit-based

### Q: Generated model not working
**A:**
1. Check file format (.safetensors vs .pth)
2. Verify model path is correct
3. Ensure tokenizer matches model architecture
4. Check model was fully downloaded (not partial)

## Next Steps

After distillation:

1. **Test**: Run on your downstream task
2. **Quantize**: Further compress with INT4/INT8
3. **Export**: Convert to ONNX for deployment
4. **Deploy**: Use in production with llama.cpp or TensorRT

## Resources

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [QTinker GitHub](https://github.com/manat0912/QTinker)
- [Model Registry](BERT_MODELS.md)

## Support

For issues or questions:
1. Check the logs in `logs/api/` folder
2. Review `BERT_MODELS.md` documentation
3. Check existing GitHub issues
