module.exports = {
  run: [{
    when: "{{!kernel.script.exists(cwd, '.git')}}",
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
    method: "notify",
    params: {
      html: "Installation complete! Click the 'start' tab to get started."
    }
  }
  ]
}