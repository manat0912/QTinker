module.exports = {
    daemon: true,
    run: [{
      method: "shell.run",
      params: {
        path: "app",
        venv: "env",
        env: {
          PYTORCH_ENABLE_MPS_FALLBACK: 1
        },
        message: [
          "C:\\pinokio\\api\\QTinker\\app\\env\\Scripts\\python.exe app.py",
        ],
        on: [{ "event": "/http:\/\/[0-9.:]+/", "done": true }]
      }
    }, {
      "method": "local.set",
      "params": {
        "url": "{{input.event[0]}}"
      }
    }
  ]
}
