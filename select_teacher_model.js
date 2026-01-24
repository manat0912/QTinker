module.exports = {
  run: [{
    method: "filepicker",
    params: {
      title: "Select Teacher Model Folder",
      properties: ["openDirectory"]
    }
  }, {
    method: "json.set",
    params: {
      file: "pinokio_meta.json",
      path: "teacher_model",
      value: "{{input.path}}"
    }
  }]
}
