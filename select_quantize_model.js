module.exports = {
  run: [{
    method: "filepicker",
    params: {
      title: "Select Model to Quantize",
      properties: ["openDirectory"]
    }
  }, {
    method: "json.set",
    params: {
      file: "pinokio_meta.json",
      path: "quantize_model",
      value: "{{input.path}}"
    }
  }]
}
