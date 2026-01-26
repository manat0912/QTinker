module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: [
        "git clone https://github.com/manat0912/QTinker.git .",
      ]
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "uv pip install gradio",
      ],
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "echo 'Installing core dependencies...'",
        "uv pip install \"numpy<2\" scipy>=1.10.0 pyyaml>=6.0 requests>=2.31.0",
      ],
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "echo 'Installing transformer and diffusion models...'",
        "uv pip install transformers>=4.30.0 accelerate>=0.20.0 diffusers>=0.21.0 safetensors>=0.3.0",
      ],
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "echo 'Installing quantization frameworks...'",
        "uv pip install torchao>=0.2.0 auto-gptq>=0.7.0 bitsandbytes>=0.40.0",
        "uv pip install autoawq>=0.2.0 --no-build-isolation",
      ],
    }
  }, {
    method: "script.start",
    params: {
      uri: "torch.js",
      params: {
        venv: "env",
        path: "app",
        xformers: false,
        triton: true
      }
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "echo 'Installing distillation and knowledge transfer tools...'",
        "uv pip install sentence-transformers>=2.7.0 gguf>=0.1.0",
      ],
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "echo 'Installing ONNX optimization and export tools...'",
        "uv pip install optimum[onnx]>=1.13.0 onnx-simplifier>=0.4.33 onnx>=1.14.0 onnxruntime>=1.16.0",
      ],
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "echo 'Installing Intel Neural Compressor for automated optimization...'",
        "uv pip install neural-compressor>=3.1.0 neural-speed>=2.0.0",
      ],
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "echo 'Installing OpenVINO for cross-platform optimization...'",
        "uv pip install openvino>=2024.1.0",
      ],
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "echo 'Installing framework support libraries...'",
        "{{platform === 'darwin' ? 'uv pip install tensorflow>=2.10.0' : 'uv pip install tensorflow-cpu>=2.10.0'}}",
      ],
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "echo 'Installing additional utilities and visualization tools...'",
        "uv pip install pillow>=9.0.0 opencv-python>=4.7.0 librosa>=0.10.0",
      ],
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "echo 'Installing inference optimization tools...'",
        "{{gpu === 'nvidia' ? (platform === 'win32' ? 'set CMAKE_ARGS=-DGGML_CUDA=on && uv pip install llama-cpp-python>=0.2.0 --no-build-isolation' : 'CMAKE_ARGS=\"-DGGML_CUDA=on\" uv pip install llama-cpp-python>=0.2.0 --no-build-isolation') : 'uv pip install llama-cpp-python>=0.2.0'}}",
      ],
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "echo 'Downloading BERT models...'",
        "git clone --depth 1 https://github.com/google-research/bert bert_models/google_research_bert",
        "git clone --depth 1 https://github.com/huawei-noah/Pretrained-Language-Model bert_models/huawei_noah_bert",
      ]
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "echo 'Downloading BERT-Large, BERT-Small variants and creating model registry...'",
        "python download_bert_models.py"
      ]
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "{{platform === 'win32' ? 'if exist download_models.py python download_models.py' : 'test -f download_models.py && python download_models.py || true'}}"
      ]
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "python -c \"from universal_model_loader import PinokioPathDetector; print(f'Pinokio root detected: {PinokioPathDetector.find_pinokio_root()}')\"",
        "python -c \"from enhanced_file_browser import ModelPathSelector; paths = ModelPathSelector.get_default_paths(); print(f'Teacher models: {paths[\\\"teacher_root\\\"]}'); print(f'Custom models: {paths[\\\"custom_root\\\"]}')\""
      ]
    }
  }, {
    method: "notify",
    params: {
      html: "🎉 <b>QTinker Installation Complete!</b><br><br><b>BERT Models Installed:</b><br>✓ BERT-Large Uncased (24-layer, 1024-hidden, 340M params)<br>✓ BERT-Large Cased (24-layer, 1024-hidden, 340M params)<br>✓ BERT-Large Uncased Whole Word Masking<br>✓ BERT-Large Cased Whole Word Masking<br>✓ BERT-Small (4-layer, 512-hidden) - for distillation<br>✓ BERT-Mini (4-layer, 256-hidden) - for distillation<br>✓ BERT-Tiny (2-layer, 128-hidden) - ultra-light<br>✓ BERT-Medium (8-layer, 512-hidden)<br>✓ Multilingual BERT<br>✓ Chinese BERT<br>✓ DistilBERT variants available<br><br><b>Distillation Features:</b><br>✓ Logit-based Knowledge Distillation (KD)<br>✓ Patient Knowledge Distillation<br>✓ Feature-based Distillation<br><br><b>Quantization (Production-Grade):</b><br>✓ TorchAO (INT4, INT8, FP8, NF4)<br>✓ GPTQ & AutoGPTQ<br>✓ AWQ (Activation-Aware Quantization)<br>✓ Bitsandbytes<br>✓ ONNX Runtime<br><br><b>Advanced Features:</b><br>✓ Gradio Web UI<br>✓ Smart GPU/CPU management<br>✓ No HuggingFace token required<br>✓ Model registry & selection<br><br>👉 Click <b>'Start'</b> to launch the web UI!"
    }
  }
  ]
}