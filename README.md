# Statlas 多运动模式量化工作区

## 目录结构

```text
quant_folder/
├── run.py / run.sh                 # 总入口
├── common/
│   └── evaluation/                 # 各模式共用的 loader 与 metric
└── modes/
    ├── demo/
    │   ├── model/                  # 原始 ONNX 模型
    │   ├── datasets/
    │   │   ├── calibration/images
    │   │   └── evaluation/{images,annotations}
    │   ├── configs/                # quant/eval/visualize/compile 等配置
    │   ├── outputs/                # quant/evaluation/compile 生成结果
    │   └── docs/
    └── soccer/
        ├── model/
        ├── datasets/               # calibration/evaluation/draft/reports
        ├── configs/
        ├── outputs/                # quant/evaluation/compile 生成结果
        ├── tools/
        └── docs/
```

每个运动模式自包含模型、数据、配置和输出。只有通用 Python loader/metric 放在 `common/`，避免复制代码。

## 首次环境配置

机器相关路径不写在代码或 YAML 中。首次拉取仓库后执行：

```bash
cd /path/to/quant_folder
cp env.example.sh env.sh
vim env.sh
```

在 `env.sh` 中配置 Conda 根目录、环境名称和 StatlasCompile 目录。该文件只属于本机并已被
Git 忽略；可提交的变量模板是 `env.example.sh`。

## 总入口

```bash
cd /home/dragonfly/wj_sdk/quant_folder

./run.sh --list
./run.sh soccer status
./run.sh soccer validate
./run.sh soccer quant
./run.sh soccer eval
./run.sh soccer visualize
./run.sh soccer float-eval
./run.sh soccer float-visualize
./run.sh soccer compile

./run.sh demo quant
./run.sh demo eval
./run.sh demo float-eval
./run.sh demo compile
```

`float-eval` 使用 `model/` 下的原始 ONNX 计算 AP，并把框图输出到
`outputs/evaluation/float_visualizations`；`float-visualize` 使用原始 ONNX 对 draft 图片画框。
二者不依赖量化产物，可用于和 `eval`/`visualize` 的量化效果对比。

完整执行量化、量化评估、原始 ONNX 对照评估和编译：

```bash
./run.sh soccer all
```

`all` 会同时生成量化模型的 `visualizations` 和原始 ONNX 的 `float_visualizations`。

只查看实际命令而不执行：

```bash
./run.sh soccer all --dry-run
```

添加 soccer 数据：

```bash
./run.sh soccer add-calibration /path/to/calibration_images
./run.sh soccer import-eval
```

将待评估图片直接放入 `datasets/draft/images`，将审核后的 COCO 标注保存为
`datasets/draft/annotations/instances.json`，再运行 `import-eval`，即可安全合并到正式
evaluation 数据集。

新增运动模式时复制任一模式骨架，至少提供以下配置：

```text
modes/<mode>/configs/quant.yaml
modes/<mode>/configs/eval.yaml
modes/<mode>/configs/visualize.yaml
modes/<mode>/configs/compile.yaml
modes/<mode>/configs/mixed_precision.yaml
```

Soccer 数据清洗、人工标注和量化细节见 [`modes/soccer/docs/PIPELINE.md`](modes/soccer/docs/PIPELINE.md)。

## Git 管理约定

- 代码、配置、文档、标注 JSON，以及各模式的校准、评估和草稿原图均提交到仓库。
- 原始 ONNX 模型、编译产物、量化结果、评估结果、可视化图片、日志和压缩包不提交；这些内容可以通过配置和入口脚本重新生成。
- 当前原图约 0.78 GB。远端仓库若限制总体积或传输速度，可后续再为图片启用 Git LFS；当前按普通 Git 文件管理图片。
- 空的输出目录通过 `.gitkeep` 保留。
