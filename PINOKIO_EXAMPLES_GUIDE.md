# Pinokio Examples Analysis: Launcher Best Practices

Generated: January 26, 2026

---

## 1. Available Example Launcher Projects

### Complete List (85 examples)

The Pinokio examples folder contains a comprehensive range of launcher projects. Here are the major categories:

#### **Video/Image Generation**
- `mochi` - Text-to-video model (Genmo Mochi)
- `flux-webui` - Flux image generation WebUI
- `stable-diffusion-webui-forge` - Advanced SD implementation
- `comfy` - ComfyUI node-based interface
- `RuinedFooocus` - Fooocus fork
- `MFLUX-WEBUI` - Flux web interface
- `aura-sr-upscaler` - Image upscaling
- `sd35` - Stable Diffusion 3.5
- `instant-ir` - Image restoration

#### **Audio/Music**
- `Kokoro-TTS` - Text-to-speech
- `AllTalk-TTS` - Advanced TTS
- `AudioX` - Audio processing
- `audiocraft_plus` - Meta AudioCraft extended
- `e2-f5-tts` - Fast TTS
- `MMAudio` - Multimodal audio
- `openaudio` - Open audio platform
- `Orpheus-TTS-FastAPI` - TTS server
- `rc-stableaudio` - Stable Audio
- `stableaudio` - Audio generation
- `StyleTTS2_Studio` - Style-based TTS
- `text2midi` - Text to MIDI
- `whisper-webui` - Speech-to-text interface
- `yue` - Audio tool

#### **3D/Video**
- `TRELLIS` - 3D generation
- `liveportrait` - Live portrait animation
- `hallo` - Animation synthesis
- `janus` - Video/3D tool
- `pyramidflow` - Video generation
- `stable-fast-3d` - 3D generation
- `Hunyuan3d-2-lowvram` - 3D model generation
- `hunyuanvideo` - Video generation
- `ai-video-composer` - Video composition
- `video-background-removal` - Background removal

#### **AI/LLM Tools**
- `open-webui` - Universal LLM interface
- `sillytavern` - Character AI chat
- `llamafactory` - LLM fine-tuning
- `autogpt` - Autonomous AI agent
- `browser-use` - Browser automation

#### **Specialized Tools**
- `comfy` - Node-based AI workflow
- `applio` - Voice conversion
- `MagicQuill` - Interactive editing
- `MatAnyone` - Material generation
- `omnigen` - Omni-generative tool
- `omniparser` - Parsing tool
- `photomaker2` - Photo editing
- `RMBG-2-Studio` - Background removal
- `clarity-refiners-ui` - Enhancement

#### **Utility & Reference**
- `cli-example` - CLI launcher template
- `installable-cli-example` - Installable CLI
- `serverless_web_app` - Static web app (Ollama Chat)
- `READ2ME` - Documentation tool

---

## 2. Well-Implemented install.js Structure

### Pattern 1: Python Package Installation (Recommended)

**File:** `C:\pinokio\prototype\system\examples\open-webui\install.js`

```javascript
module.exports = {
  "run": [{
    "method": "shell.run",
    "params": {
      "venv": "env",
      "venv_python": "3.11",
      "path": "app",
      "message": [
        "uv pip install open-webui -U",
        "uv pip install onnxruntime==1.20.1"
      ]
    }
  }]
}
```

**Key Practices:**
- ✅ Uses `uv` instead of `pip` (faster, more reliable)
- ✅ Specifies Python version with `venv_python: "3.11"`
- ✅ Virtual environment via `venv: "env"`
- ✅ Works from `path: "app"` (relative path)
- ✅ Multiple packages in array format
- ✅ Pinning specific versions when needed

---

### Pattern 2: Git Clone + Dependencies + PyTorch

**File:** `C:\pinokio\prototype\system\examples\whisper-webui\install.js`

```javascript
module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/jhj0517/Whisper-WebUI app",
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt",
          "uv pip install gradio==5.34.0 numpy==1.26.4"
        ]
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app",
        }
      }
    },
  ]
}
```

**Key Practices:**
- ✅ Clones repository to `app` folder first
- ✅ Separate step for installing dependencies
- ✅ Uses `torch.js` built-in script for PyTorch installation
- ✅ Passes venv and path parameters to torch.js
- ✅ Works across Windows/Mac/Linux

---

### Pattern 3: Complex Installation with GPU Detection

**File:** `C:\pinokio\prototype\system\examples\stable-diffusion-webui-forge\install.js`

```javascript
module.exports = {
  run: [
    // Clone repository
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/lllyasviel/stable-diffusion-webui-forge app",
        ]
      }
    },
    // Configure app settings
    {
      method: "self.set",
      params: {
        "app/ui-config.json": {
          "txt2img/CFG Scale/value": 1.0
        },
        "app/config.json": {
          "forge_preset": "flux"
        }
      }
    },
    // Initialize venv
    {
      method: "shell.run",
      params: {
        message: " ",
        venv: "app/venv"
      }
    },
    // GPU-specific installation (NVIDIA 50 series)
    {
      "when": "{{gpu === 'nvidia' && kernel.gpu_model && / 50.+/.test(kernel.gpu_model) }}",
      "method": "shell.run",
      "params": {
        "venv": "venv",
        "path": "app",
        "message": [
          "uv pip install -U bitsandbytes --force-reinstall",
          "uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall",
          "uv pip install numpy==1.26.2 --force-reinstall",
        ]
      },
    },
    // File sharing for seamless model access
    {
      id: "share",
      method: "fs.share",
      params: {
        drive: {
          upscale_models: ["app/models/ESRGAN"],
          checkpoints: "app/models/Stable-diffusion",
          vae: "app/models/VAE",
          controlnet: "app/models/ControlNet",
          // ... more model folders
        }
      }
    },
  ]
}
```

**Key Practices:**
- ✅ Conditional `when` clauses for GPU detection
- ✅ Uses `self.set` to configure JSON files
- ✅ `fs.share` for model folder access
- ✅ Modular step structure (one action per step)
- ✅ Clear separation of concerns

---

### Pattern 4: Simple Editable Package Installation

**File:** `C:\pinokio\prototype\system\examples\flux-webui\install.js`

```javascript
module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install -r requirements.txt"
        ],
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          xformers: true   // PyTorch extension flag
        }
      }
    },
  ]
}
```

**Key Practices:**
- ✅ Minimal configuration
- ✅ Automatic requirements.txt detection
- ✅ Optional PyTorch extensions via flags
- ✅ Comment-based configuration hints

---

## 3. URL Capture Patterns in start.js

### Pattern 1: Mochi (Generic HTTP URL)

**File:** `C:\pinokio\prototype\system\examples\mochi\start.js`

```javascript
module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          "TRANSFORMERS_VERBOSITY": "info",
          "CUDA_VISIBLE_DEVICES": "0"
        },
        path: "app/demos",
        message: [
          "python gradio_ui.py --model_dir ../checkpoint"
        ],
        on: [{
          "event": "/http:\/\/[0-9.:]+/",
          "done": true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[0]}}"  // Full URL from capture group
      }
    }
  ]
}
```

**Key Pattern:**
- ✅ Regex: `/http:\/\/[0-9.:]+/` - Matches `http://` followed by IP/port
- ✅ No capturing parentheses, so uses `input.event[0]` (full match)
- ✅ `on.done: true` keeps shell alive after match
- ✅ `daemon: true` required for servers

---

### Pattern 2: ComfyUI (Specific Regex with Capture Group)

**File:** `C:\pinokio\prototype\system\examples\comfy\start.js`

```javascript
module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          PYTORCH_ENABLE_MPS_FALLBACK: "1",
          TOKENIZERS_PARALLELISM: "false"
        },
        path: "app",
        message: [
          "{{platform === 'win32' && gpu === 'amd' ? 'python main.py --directml' : 'python main.py'}}"
        ],
        on: [{
          "event": "/starting server.+(http:\/\/[a-zA-Z0-9.]+:[0-9]+)/i",
          "done": true
        }, {
          "event": "/errno/i",
          "break": false
        }, {
          "event": "/error:/i",
          "break": false
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"  // Captured group inside parentheses
      }
    }
  ]
}
```

**Key Pattern:**
- ✅ Regex: `/starting server.+(http:\/\/[a-zA-Z0-9.]+:[0-9]+)/i` with capture group
- ✅ Capture group in parentheses: `(http://...)`
- ✅ Uses `input.event[1]` (first capture group)
- ✅ Multiple `on` handlers for error conditions
- ✅ Case-insensitive flag `/i`
- ✅ Platform-conditional command execution

---

### Pattern 3: Whisper WebUI (Generic URL with Whitespace)

**File:** `C:\pinokio\prototype\system\examples\whisper-webui\start.js`

```javascript
module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: { },
        path: "app",
        message: [
          "python app.py --inbrowser False",
        ],
        on: [{
          "event": "/http:\/\/\\S+/",  // \S+ matches non-whitespace
          "done": true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[0]}}"  // Full match
      }
    }
  ]
}
```

**Key Pattern:**
- ✅ Regex: `/http:\/\/\\S+/` - Matches http:// followed by any non-whitespace
- ✅ More flexible than numeric IP pattern
- ✅ Captures full URLs including hostname/paths
- ✅ Uses `input.event[0]` (no capture groups)

---

### Best Practices Summary for URL Capture

| Pattern | Regex | Use Case | Indexing |
|---------|-------|----------|----------|
| **Mochi (Simple)** | `/http:\/\/[0-9.:]+/` | localhost IP + port | `input.event[0]` |
| **ComfyUI (Specific)** | `/starting server.+(http:\/\/[a-zA-Z0-9.]+:[0-9]+)/i` | Search for specific output text + capture | `input.event[1]` |
| **Whisper (Flexible)** | `/http:\/\/\\S+/` | Any non-whitespace after http:// | `input.event[0]` |

---

## 4. Python Virtual Environment & Package Manager Best Practices

### **Best Practice #1: Always Use `uv` Instead of `pip`**

**Evidence:**
- **All modern examples use `uv`:** open-webui, flux-webui, whisper-webui, comfy
- **Why:** 
  - 10-100x faster than pip
  - Pre-installed with Pinokio
  - Handles dependency resolution better
  - Cross-platform compatible

```javascript
// ❌ AVOID
"message": "pip install -r requirements.txt"

// ✅ PREFER
"message": "uv pip install -r requirements.txt"
```

---

### **Best Practice #2: Specify Python Version**

**File Example:** `open-webui\install.js`

```javascript
{
  "venv": "env",
  "venv_python": "3.11",  // Explicitly specify version
  "message": "uv pip install package-name"
}
```

**Why:**
- Ensures reproducibility across systems
- Prevents version mismatch issues
- Pinokio automatically manages the version installation

---

### **Best Practice #3: Virtual Environment Path Convention**

**Pattern 1 - Root-level venv:**
```javascript
{
  "venv": "env",  // Creates/uses ./env virtual environment
  "path": "app"   // Runs commands in app/ folder
}
```

**Pattern 2 - Nested venv:**
```javascript
{
  "venv": "demos/env",  // Creates in ./demos/env
  "path": "app"
}
```

**Why:**
- Pinokio automatically creates if it doesn't exist
- Always use relative paths (never absolute)
- Helps avoid port conflicts when running multiple projects
- Can be excluded from git via .gitignore

---

### **Best Practice #4: Use Built-in torch.js for PyTorch**

Instead of manually managing PyTorch installation, delegate to built-in script:

```javascript
// ✅ CORRECT
{
  method: "script.start",
  params: {
    uri: "torch.js",
    params: {
      venv: "env",
      path: "app",
      xformers: true,      // Optional extensions
      // triton: true,
      // sageattention: true,
      // flashattention: true
    }
  }
}

// ❌ AVOID (manual installation)
{
  method: "shell.run",
  params: {
    venv: "env",
    message: "pip install torch torchvision..."
  }
}
```

**Why:**
- Handles GPU detection automatically (NVIDIA, AMD, Metal)
- Cross-platform PyTorch installation
- Manages optional extensions (xformers, triton, etc.)
- Pinokio optimizes for each platform

---

### **Best Practice #5: Separate Installation Steps**

```javascript
// ✅ CORRECT - Modular approach
{
  method: "shell.run",
  params: {
    message: "git clone https://... app"
  }
},
{
  method: "shell.run",
  params: {
    venv: "env",
    path: "app",
    message: "uv pip install -r requirements.txt"
  }
},
{
  method: "script.start",
  params: {
    uri: "torch.js",
    params: { venv: "env", path: "app" }
  }
}

// ❌ AVOID - Chained in single step
{
  method: "shell.run",
  params: {
    venv: "env",
    message: [
      "git clone...",
      "uv pip install...",
      "# install torch"
    ]
  }
}
```

**Why:**
- Better error isolation
- Each step can fail independently
- Clearer logging
- Easier to add conditional `when` clauses

---

### **Best Practice #6: Environment Variables**

```javascript
{
  method: "shell.run",
  params: {
    venv: "env",
    path: "app",
    env: {
      "CUDA_VISIBLE_DEVICES": "0",
      "TRANSFORMERS_VERBOSITY": "info",
      "HF_TOKEN": "{{args.hf_token}}"  // Can use template variables
    },
    message: "python main.py"
  }
}
```

**Common Variables:**
- `CUDA_VISIBLE_DEVICES` - GPU selection
- `TOKENIZERS_PARALLELISM` - HuggingFace tokenizer threads
- `HF_HOME` - HuggingFace cache location
- `TORCH_HOME` - PyTorch cache location

---

### **Best Practice #7: Use Multiple Requirements.txt Files**

**Example from ComfyUI:**
```javascript
// Different pip installs for different scenarios
{
  "message": [
    "uv pip install -r requirements.txt",    // Core requirements
    "uv pip install gradio==5.34.0",         // Additional
    "uv pip install numpy==1.26.4"           // Specific versions
  ]
}
```

**Why:**
- Base requirements.txt from project
- Pin specific versions to avoid conflicts
- Add OS/GPU-specific packages as separate lines

---

### **Best Practice #8: Minimal Shell Commands**

**Use Pinokio APIs instead of raw shell:**

```javascript
// ✅ PREFER - Using Pinokio APIs
{
  method: "self.set",
  params: {
    "app/config.json": { "preset": "flux" }
  }
}

// ❌ AVOID - Using echo/sed
{
  method: "shell.run",
  params: {
    message: "echo '{\"preset\":\"flux\"}' > app/config.json"
  }
}
```

---

## 5. Additional Installation Patterns

### **Conda Installation (When Needed)**

```javascript
{
  method: "shell.run",
  params: {
    message: "conda install -y -c conda-forge package-name"
  }
}
```

**When to use:**
- Binary packages not available via pip
- Cross-platform compilation needed
- System-level libraries required

---

### **File Linking for Models**

```javascript
{
  method: "fs.link",
  params: {
    venv: "env"  // OR
    // Custom links for model folders
  }
}
```

**From stable-diffusion-webui-forge:**
```javascript
{
  method: "fs.share",
  params: {
    drive: {
      checkpoints: "app/models/Stable-diffusion",
      vae: "app/models/VAE",
      controlnet: "app/models/ControlNet"
    }
  }
}
```

---

## 6. File Paths Reference

### Key Example Files

| Example | File Path | Purpose |
|---------|-----------|---------|
| **Mochi** | `C:\pinokio\prototype\system\examples\mochi\install.js` | Complex setup with git clone + torch.js |
| **Mochi** | `C:\pinokio\prototype\system\examples\mochi\start.js` | URL regex capture pattern |
| **Open WebUI** | `C:\pinokio\prototype\system\examples\open-webui\install.js` | Simple pip install pattern |
| **Open WebUI** | `C:\pinokio\prototype\system\examples\open-webui\pinokio.js` | Dynamic menu rendering |
| **ComfyUI** | `C:\pinokio\prototype\system\examples\comfy\start.js` | Advanced regex with capture groups |
| **ComfyUI** | `C:\pinokio\prototype\system\examples\comfy\install.js` | Async/await install with file sharing |
| **Whisper WebUI** | `C:\pinokio\prototype\system\examples\whisper-webui\install.js` | Standard Python project |
| **Whisper WebUI** | `C:\pinokio\prototype\system\examples\whisper-webui\start.js` | Flexible URL regex |
| **Flux WebUI** | `C:\pinokio\prototype\system\examples\flux-webui\install.js` | Minimal setup with torch.js |
| **SD Forge** | `C:\pinokio\prototype\system\examples\stable-diffusion-webui-forge\install.js` | GPU detection + fs.share |
| **Serverless** | `C:\pinokio\prototype\system\examples\serverless_web_app\index.html` | Static web app (no launcher needed) |

---

## Summary: Key Takeaways

### ✅ DO:
1. **Use `uv`** for all Python package installations
2. **Use `torch.js`** for PyTorch/GPU library management
3. **Specify `venv_python` version** (e.g., "3.11")
4. **Use relative paths** for `path` parameter
5. **Separate installation steps** by concern (git, venv, packages, torch)
6. **Capture URLs with regex** in `on` array using `done: true`
7. **Set local variables** for dynamic UI via `local.set`
8. **Use conditional `when`** for GPU/platform-specific steps
9. **Pin specific package versions** when known to work
10. **Use `fs.share`** for model folders across projects

### ❌ DON'T:
1. Use `pip` instead of `uv`
2. Use absolute paths in shell commands
3. Manually install PyTorch (use torch.js)
4. Chain all steps into single shell command
5. Forget `daemon: true` for servers
6. Use hardcoded ports (use `{{port}}` template)
7. Ignore `.gitignore` for generated files
8. Modify project's `app/` folder unnecessarily
9. Use Docker unless absolutely necessary
10. Make assumptions about file structure

