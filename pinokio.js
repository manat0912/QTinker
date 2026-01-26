module.exports = {
  version: "3.7",
  title: "QTinker",
  description: "Distill and quantize models using TorchAO with intelligent GPU/CPU management",
  icon: "icon.png",
  menu: async (kernel, info) => {
    let running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      update: info.running("update.js"),
      reset: info.running("reset.js")
    }

    let running_tabs = []
    if (running.start) {
      let local = info.local("start.js")
      if (local && local.url) {
        running_tabs.push({
          default: true,
          icon: "fa-solid fa-rocket",
          text: "Open Web UI",
          href: local.url,
        })
        running_tabs.push({
          icon: "fa-solid fa-terminal",
          text: "Terminal",
          href: "start.js",
        })
      } else {
        running_tabs.push({
          default: true,
          icon: "fa-solid fa-terminal",
          text: "Terminal",
          href: "start.js",
        })
      }
    } else {
      if (running.install || running.update || running.reset) {
        running_tabs.push({
          icon: "fa-solid fa-power-off",
          text: "Start",
          href: "start.js",
        })
      } else {
        running_tabs.push({
          default: true,
          icon: "fa-solid fa-power-off",
          text: "Start",
          href: "start.js",
        })
      }
    }

    return [
      ...running_tabs,
      {
        default: running.update,
        icon: "fa-solid fa-plug",
        text: "Update",
        href: "update.js",
      },
      {
        default: running.install,
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js",
      },
      {
        default: running.reset,
        icon: "fa-regular fa-circle-xmark",
        text: "Reset",
        href: "reset.js",
      }
    ]
  }
}
