# QTinker - Distill & Quantize with TorchAO

A modern, production-ready application for distilling and quantizing language models using TorchAO with intelligent GPU/CPU management and Pinokio launcher support.

## Features

- 🎯 **Flexible Model Loading**: Support for HuggingFace models and PyTorch checkpoints
- 🧪 **Advanced Distillation Strategies**: Multiple knowledge distillation methods including:
  - Logit-based Knowledge Distillation (KD)
  - Patient Knowledge Distillation (matching specific layers)
  - Custom projection layers for dimension matching
  - Configurable temperature parameters
- ⚡ **TorchAO Quantization**: Professional-grade quantization with multiple options:
  - INT4 Weight-Only (group_size configurable)
  - INT8 Dynamic Quantization
  - Model-specific quantization configurations
- 🎨 **Gradio Web UI**: Beautiful, responsive web interface with real-time log streaming
- 📦 **Modular Architecture**: Clean separation of concerns with pluggable components
- 🖥️ **Smart GPU/CPU Management**: Automatic device selection and switching
- 💾 **Memory Efficient Processing**: Intelligent VRAM monitoring and fallback strategies
- 🚀 **Pinokio Launcher Integration**: One-click installation, start, update, and reset
- 🔧 **Model Selection UI**: Interactive file pickers for teacher, student, and target models
- 📊 **Registry System**: Comprehensive model registry for tracking supported architectures
- 🔗 **Symbolic Linking**: Automatic model linking for seamless integration

## Project Structure

```
QTinker/
├── app/
│   ├── app.py              # Main entry point (Pinokio compatible)
│   ├── gradio_ui.py        # Full Gradio web interface
│   ├── distillation.py     # Advanced distillation strategies (KD, Patient-KD, etc.)
│   ├── distill_quant_app.py # Legacy desktop UI (Tkinter)
│   ├── model_loader.py     # Unified model loading utilities
│   ├── download_models.py  # Model download and management
│   ├── registry.py         # Model architecture registry
│   ├── run_distillation.py # Distillation pipeline executor
│   ├── bert_models/        # BERT model implementations
│   ├── distilled/          # Output: Distilled models
│   └── quantized/          # Output: Quantized models
├── config/
│   ├── paths.yaml          # Path configuration
│   ├── quant_presets.yaml  # Quantization presets
│   └── settings.yaml       # Application settings
├── configs/
│   └── torchao_configs.py  # TorchAO quantization configurations
├── core/
│   ├── device_manager.py   # GPU/CPU management
│   ├── distillation.py     # Core distillation logic
│   ├── local_llm.py        # Local LLM utilities
│   └── logic.py            # Main pipeline logic
├── settings/
│   └── app_settings.py     # Global application settings
├── data/
│   └── train_prompts.txt   # Training prompts for distillation
├── outputs/                # Output directories
│   ├── distilled/          # Distilled model artifacts
│   └── quantized/          # Quantized model artifacts
├── install.js             # Pinokio installation script
├── start.js               # Pinokio launcher script
├── update.js              # Pinokio update script
├── reset.js               # Pinokio reset script
├── pinokio.js             # Pinokio UI definition
├── select_teacher_model.js # Teacher model selector
├── select_student_model.js # Student model selector
├── select_quantize_model.js # Quantization target selector
├── distill_quantize.js    # Combined distill & quantize trigger
├── link.js                # Model symbolic linking
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project configuration
├── pinokio_meta.json      # Metadata and state persistence
└── README.md              # This file
```

## Installation

### Automatic Installation (Recommended via Pinokio)

Simply open the project in Pinokio and click the "Install" button. The launcher will automatically:
1. Create a Python virtual environment
2. Install all dependencies using `uv pip`
3. Set up PyTorch with CUDA support (if available)
4. Configure the application

### Manual Installation with pip

```bash
pip install -r requirements.txt
```

### Manual Installation with uv (recommended for Pinokio)

```bash
uv pip install -r requirements.txt
```

### CUDA Support

If you need specific CUDA-enabled PyTorch wheels:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Using Pinokio Launcher (Easiest)

1. Open QTinker in Pinokio
2. Click "Install" (first time only) to set up dependencies
3. Click "Start" to launch the Gradio web interface
4. The interface will open automatically in your browser
5. Use the web UI to select models and run distillation/quantization
6. Use the model selector tools in the sidebar to configure your models:
   - **Select Teacher Model**: Choose the teacher model for knowledge distillation
   - **Select Student Model**: Choose the student model to be distilled
   - **Select Quantize Model**: Choose the model to quantize
   - **Distill & Quantize**: Run the complete pipeline

### Running the Application Directly

```bash
python app/app.py
```

Or directly access the Gradio UI:

```bash
cd app
python gradio_ui.py
```

The Gradio interface will be available at `http://localhost:7860`

### Using the Web Interface

1. **Model Selection**: Use the sidebar buttons to select teacher, student, and quantization target models
2. **Model Path**: Enter the path to your model (HuggingFace folder or PyTorch checkpoint)
3. **Model Type**: Select the type of model you're loading:
   - HuggingFace folder
   - PyTorch .pt/.bin file
4. **Quantization Type**: Choose your quantization method:
   - INT4 (weight-only) - More aggressive compression
   - INT8 (dynamic) - Better accuracy with moderate compression
5. **Distillation Strategy** (if applicable):
   - Logit KD - Match output logits
   - Patient KD - Match intermediate layers
6. **Run**: Click "Run Distill + Quantize" to start the pipeline
7. **Monitor**: Watch real-time log output for progress and debugging

### Programmatic Usage

```python
from core.logic import run_pipeline

# Run the complete pipeline
distilled_path, quantized_path = run_pipeline(
    model_path="microsoft/phi-2",
    model_type="HuggingFace folder",
    quant_type="INT8 (dynamic)",
    log_fn=print
)
```

For advanced usage with custom distillation strategies:

```python
from app.distillation import LogitKD, PatientKD
from core.device_manager import DeviceManager

# Create device manager
device_manager = DeviceManager()
device = device_manager.get_device()

# Load teacher and student models
teacher_model = load_model("teacher_path")
student_model = load_model("student_path")

# Apply distillation strategy
strategy = LogitKD(teacher_model, student_model, temperature=3.0)
loss = strategy.compute_loss(student_outputs, teacher_outputs)
```

## Configuration

### TorchAO Quantization Configs

Edit `configs/torchao_configs.py` to customize quantization settings:

```python
from torchao.quantization.configs import Int4WeightOnlyConfig, Int8DynamicConfig

# INT4 Configuration
Int4WeightOnlyConfig(
    group_size=128,          # Default: 128 (lower = more granular, slower)
    inner_k_tiles=8,         # Tiling for optimization
    padding_allowed=True     # Allow padding for performance
)

# INT8 Configuration  
Int8DynamicConfig(
    act_range_method="minmax"  # Range calculation method
)
```

### Application Settings

Edit `settings/app_settings.py` to customize:
- Output directories
- Default model/quantization types
- GPU/CPU management thresholds
- Device switching behavior
- Memory limits

Example:

```python
# Device Management
MIN_VRAM_GB = 2.0              # Minimum VRAM to use GPU
VRAM_THRESHOLD = 0.9           # Use CPU if model > 90% of VRAM
AUTO_DEVICE_SWITCHING = True   # Enable automatic switching

# Output Directories
DISTILLED_OUTPUT_DIR = "outputs/distilled/"
QUANTIZED_OUTPUT_DIR = "outputs/quantized/"

# Model Defaults
DEFAULT_MODEL_TYPE = "HuggingFace folder"
DEFAULT_QUANT_TYPE = "INT8 (dynamic)"
```

### Model Registry

The `registry.py` file maintains a comprehensive registry of supported model architectures with their optimal configurations:

```python
SUPPORTED_MODELS = {
    "phi-2": {
        "type": "causal-lm",
        "default_quant": "INT4",
        "supports_distillation": True
    },
    "bert-base-uncased": {
        "type": "masked-lm",
        "default_quant": "INT8",
        "supports_distillation": True
    },
    # ... more models
}
```

## Advanced Features

### Knowledge Distillation Strategies

QTinker supports multiple knowledge distillation methods:

#### 1. Logit-based Knowledge Distillation (LogitKD)
Matches the output logits between teacher and student models using KL divergence with temperature scaling.

```python
from app.distillation import LogitKD

strategy = LogitKD(teacher_model, student_model, temperature=3.0)
loss = strategy.compute_loss(student_outputs, teacher_outputs)
```

**Best for**: General-purpose distillation, good baseline for most architectures

#### 2. Patient Knowledge Distillation (PatientKD)
Matches hidden states at specific layers between teacher and student models. Useful when student architecture differs significantly from teacher.

```python
from app.distillation import PatientKD

strategy = PatientKD(
    teacher_model, 
    student_model,
    student_layers=[2, 4, 6],      # Layers to extract from student
    teacher_layers=[4, 8, 12],     # Corresponding teacher layers
    loss_fn=F.mse_loss
)
loss = strategy.compute_loss(student_outputs, teacher_outputs)
```

**Best for**: Custom architectures, fine-grained control, layer-specific matching

#### 3. Projection Layer Matching
Automatically handles dimension mismatches between teacher and student hidden states:

```python
from app.distillation import ProjectionLayer

projection = ProjectionLayer(student_dim=768, teacher_dim=1024)
projected_student = projection(student_hidden_states)
```

**Best for**: Distilling to significantly smaller models

### Device Management System

The intelligent device manager ensures optimal GPU/CPU utilization:

- **Automatic GPU Detection**: Detects CUDA (NVIDIA), MPS (Apple Silicon), or CPU
- **VRAM Monitoring**: Real-time GPU memory tracking
- **Automatic Fallback**: Seamlessly switches to CPU when:
  - Less than 2GB VRAM available
  - Model size exceeds 90% of available VRAM
  - GPU runs out of memory during processing
- **Memory Efficiency**: Models loaded on CPU first, then moved to GPU if appropriate
- **Cache Management**: Automatic GPU cache clearing between operations

```python
from core.device_manager import DeviceManager

device_manager = DeviceManager()
device = device_manager.get_device()
print(f"Using device: {device}")  # Outputs: cuda, mps, or cpu
```

### Model Loading Utilities

Unified model loading with automatic format detection:

```python
from app.model_loader import load_model

# Supports multiple formats
model = load_model("facebook/opt-350m")  # HuggingFace
model = load_model("./local_model.pt")   # Local PyTorch
model = load_model("./model/")           # Local folder
```

### Model Registry System

Track and manage supported model architectures:

```python
from app.registry import ModelRegistry

registry = ModelRegistry()
supported_models = registry.get_supported_models()
config = registry.get_model_config("phi-2")
```

## Dependencies

- `torch>=2.0.0` - PyTorch deep learning framework
- `torchao>=0.1.0` - TorchAO quantization library
- `transformers>=4.30.0` - HuggingFace transformers for model loading
- `gradio>=4.0.0` - Web UI framework for interactive interface
- `accelerate>=0.20.0` - Model acceleration utilities
- `pyyaml` - Configuration file handling
- `numpy` - Numerical computing

Full dependency list available in `requirements.txt`

## GPU/CPU Management

The application automatically manages device selection:

- **GPU Detection**: Automatically detects and uses CUDA (NVIDIA) or MPS (Apple Silicon) when available
- **VRAM Monitoring**: Monitors GPU memory usage and switches to CPU when VRAM is limited
- **Automatic Fallback**: Falls back to CPU if:
  - Less than 2GB VRAM is available
  - Model size exceeds 90% of available VRAM
  - GPU runs out of memory during processing
- **Memory Efficient**: Loads models on CPU first, then moves to GPU if appropriate
- **Cache Management**: Automatically clears GPU cache between operations

### Device Settings

You can adjust device management behavior in `settings/app_settings.py`:

```python
MIN_VRAM_GB = 2.0  # Minimum VRAM required to use GPU
VRAM_THRESHOLD = 0.9  # Use CPU if model size > VRAM * threshold
AUTO_DEVICE_SWITCHING = True  # Enable automatic device switching
```

## Output Directories

All output models are saved in standard HuggingFace format for easy reuse:

- **Distilled Models**: `outputs/distilled/` - Models after knowledge distillation
- **Quantized Models**: `outputs/quantized/` - Models after quantization

Each output includes:
- Model weights and architecture
- Tokenizers (when available)
- Configuration files
- Quantization metadata

## GPU/CPU Management

### Automatic Device Selection

The application automatically manages device selection based on available hardware:

**GPU Detection**:
- NVIDIA CUDA GPUs
- Apple Silicon (MPS)
- CPU fallback

**Memory Management**:
- Monitors GPU VRAM in real-time
- Prevents out-of-memory errors
- Switches to CPU when necessary

**Threshold Settings** (configurable in `settings/app_settings.py`):
- MIN_VRAM_GB: Minimum VRAM required (default: 2.0)
- VRAM_THRESHOLD: Use CPU if model > X% of VRAM (default: 0.9 = 90%)
- AUTO_DEVICE_SWITCHING: Enable/disable automatic switching (default: True)

### Device Switching Behavior

The system falls back to CPU if:
- Less than 2GB VRAM available
- Estimated model size exceeds 90% of available VRAM
- GPU runs out of memory during processing

### Manual Device Configuration

```python
from core.device_manager import DeviceManager

device_manager = DeviceManager(
    min_vram_gb=2.0,
    vram_threshold=0.9,
    auto_switching=True
)

device = device_manager.get_device()
```

## Troubleshooting

### Out of Memory (OOM) Errors

**Problem**: Model loading fails with CUDA out of memory
**Solution**: 
- The app will automatically switch to CPU
- Or reduce model size by using smaller teacher/student models
- Or increase VRAM threshold in settings to force CPU usage earlier

### Model Loading Issues

**Problem**: Model fails to load from HuggingFace
**Solution**:
1. Ensure you have internet connection
2. Verify model name is correct
3. Check HuggingFace authentication if using private models
4. Use local model paths instead

### Distillation Not Starting

**Problem**: Distillation script fails to execute
**Solution**:
1. Ensure both teacher and student models are loaded
2. Check that training data is available in `data/train_prompts.txt`
3. Verify CUDA/device availability in logs
4. Check logs folder for detailed error messages

### Performance Issues

**Problem**: Quantization/distillation is slow
**Solution**:
- Reduce batch size in configuration
- Use INT4 quantization for faster processing
- Ensure GPU is available and not occupied by other processes
- Use smaller models for testing

## Output Models

Models are saved in the following structure:

```
outputs/
├── distilled/
│   └── model_name/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer.*
└── quantized/
    └── model_name_quantized/
        ├── config.json
        ├── pytorch_model.bin
        └── tokenizer.*
```

All saved models are compatible with HuggingFace transformers and can be loaded with:

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("outputs/distilled/model_name")
tokenizer = AutoTokenizer.from_pretrained("outputs/distilled/model_name")
```

## Pinokio Launcher Scripts

QTinker is fully integrated with Pinokio for easy one-click operations:

### Available Commands

- **Install**: Automatically sets up Python environment and installs dependencies
- **Start**: Launches the Gradio web interface
- **Update**: Updates QTinker and dependencies to the latest version
- **Reset**: Clears virtual environment and cached files for a fresh start

### Model Selection Tools

Sidebar buttons for easy model management:
- **Select Teacher Model**: Pick teacher model for knowledge distillation
- **Select Student Model**: Pick student model to be distilled
- **Select Quantize Model**: Pick model to quantize
- **Link Models**: Create symbolic links for model references

### Metadata Persistence

Model selections and settings are saved in `pinokio_meta.json`:

```json
{
  "teacher_model": "/path/to/teacher",
  "student_model": "/path/to/student",
  "quantize_model": "/path/to/model"
}
```

## API Documentation

### Python API

#### Run Complete Pipeline

```python
from core.logic import run_pipeline

result = run_pipeline(
    model_path="microsoft/phi-2",
    model_type="HuggingFace folder",
    quant_type="INT8 (dynamic)",
    distill_type="LogitKD",
    temperature=3.0,
    log_fn=print
)
```

#### Load Model

```python
from app.model_loader import load_model

model, tokenizer = load_model(
    model_path="facebook/opt-350m",
    model_type="HuggingFace folder",
    device="cuda"
)
```

#### Quantize Model

```python
from torchao.quantization import quantize_
from torchao.quantization.configs import Int8DynamicConfig

quantize_(model, Int8DynamicConfig())
model.save_pretrained("outputs/quantized/model_name")
```

#### Apply Distillation

```python
from app.distillation import LogitKD

strategy = LogitKD(teacher_model, student_model, temperature=3.0)
for batch in dataloader:
    loss = strategy.compute_loss(
        student_model(**batch),
        teacher_model(**batch)
    )
    loss.backward()
```

### REST API (via Gradio)

When running the app, a Gradio interface provides HTTP endpoints:

```bash
# Gradio automatically generates REST endpoints
# Example: POST to Gradio endpoint with model parameters
curl -X POST "http://localhost:7860/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"model_path": "phi-2", "quant_type": "INT8"}'
```

### CLI Usage

```bash
# Start the application
python app/app.py

# Direct Gradio launch
python app/gradio_ui.py

# Run distillation only
python app/run_distillation.py --teacher-path /path/to/teacher \
                                --student-path /path/to/student

# Download models
python app/download_models.py --model-name "phi-2" --output-dir "./models"
```
