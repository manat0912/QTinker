module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: [
        "git clone https://github.com/manat0912/QTinker.git .",
      ]
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
        "uv pip install gradio",
        "uv pip install -r requirements.txt",
      ],
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "{{#if platform == 'win32'}}mkdir bert_models{{else}}mkdir -p bert_models{{/if}}",
        "git clone --depth 1 https://github.com/google-research/bert bert_models/google_research_bert",
        "git clone --depth 1 https://github.com/huawei-noah/Pretrained-Language-Model bert_models/huawei_noah_bert",
        "python download_models.py"
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
      html: "Installation complete! The app now supports:<br>✓ All Stable Diffusion models<br>✓ All libraries (PyTorch, TensorFlow, JAX, etc.)<br>✓ GGUF quantization<br>✓ Cross-platform path detection<br>Click 'Start' to launch the web UI."
    }
  }
  ]
}