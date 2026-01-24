module.exports = {
  run: [{
    method: "shell.run",
    params: {
      venv: "env",
      path: ".",
      message: [
        "uv pip install --upgrade -r requirements.txt"
      ]
    }
  }]
}
