# Quant Folder 本地 Web 服务

页面不会展示或返回任何 YAML 内容。配置仍由仓库内的 `run.sh` 使用，Web 端只提供模式、平台、模型上传和操作流水线控制。平台包括 VS859、RK3576 和安霸 CVFlow。

在仓库根目录执行：

```bash
python3 Html/server.py
```

浏览器打开 <http://127.0.0.1:8765/>。也可以指定端口：

```bash
python3 Html/server.py --port 8766
```

服务只监听本机回环地址，不对局域网开放。页面可以上传 `.onnx` 模型，并通过现有 `run.sh` 执行以下操作：

`clean-model`、`cut-head`、`quant`、`compile`、`eval`、`float-eval`、`compare`、`validate`、`status`、`clean`。

操作按页面中从左到右的顺序串行执行。比如取消其他步骤，只保留 `quant`、`compile`、`eval`，即可执行：

```text
quant -> compile -> eval
```

页面首次打开时默认勾选的就是这三个步骤；清洗、裁切和清理等可能改变生成物的操作需要手动勾选。

模型上传后保存到 `modes/<mode>/model/`，并自动更新现有 VS859、RK3576 和安霸配置中的模型路径和量化产物基名。上传不会自动开始量化；请确认配置对应的输入尺寸、数据集和平台后再启动流水线。安霸平台可由仓库根目录的 `./setup_conda_envs.sh --ambarella-only` 负责加载、启动或复用 CVTools 容器，并将已验证的 `AMBARELLA_CONTAINER` 写入 `env.sh`；也可以手工填写该变量。

量化和编译仍然使用命令行中的 `env.sh`、Conda 环境和依赖。Web 页面只是模型上传和任务触发入口，不会替代原有脚本逻辑。
