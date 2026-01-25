module.exports = {
    daemon: true,
    run: [{
      method: "shell.run",
      params: {
        path: "app",
        venv: "env",
        env: {
          PYTORCH_ENABLE_MPS_FALLBACK: 1,
          PINOKIO_ROOT: "{{cwd}}/../.."
        },
        message: [
          "python app.py",
        ],
        on: [{
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    }, {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
