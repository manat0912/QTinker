module.exports = {
  run: [{
    method: "filepicker",
    params: {
      title: "Select Student Model Folder",
      properties: ["openDirectory"]
    }
  }, {
    method: "json.set",
    params: {
      file: "pinokio_meta.json",
      path: "student_model",
      value: "{{input.path}}"
    }
  }]
}
