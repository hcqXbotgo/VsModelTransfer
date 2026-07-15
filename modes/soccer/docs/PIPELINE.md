# Soccer 3328×1024 数据、量化、评估与编译流程

本文档是 `yolov5n_falcon2_3328_1024_clean.onnx` 的维护入口。所有命令默认从仓库的
`quant_folder` 根目录执行。首次运行前复制 `env.example.sh` 为 `env.sh`，并配置本机的
Conda 环境与编译器路径。

## 1. 当前状态

数据集采用 COCO Detection JSON，模型输入为 RGB、`[N, C, H, W] = [1, 3, 1024, 3328]`。

当前种子数据：

| 分区 | 图片 | 标注 | 状态 | 用途 |
|---|---:|---:|---|---|
| `evaluation/images` | 11 | 173 | COCO 人工真值 | 正式评估 |
| `draft/images` | 5 | 100 | 浮点模型预标注，未人工审核 | CVAT 标注草稿 |
| `calibration/images` | 5 | 不需要 | 球场全景原图 | PTQ 校准 |

注意：`draft` 绝不能直接用于 AP 计算。模型预测不是 ground truth，只有人工逐框确认并导入
`instances_eval.json` 后才属于正式评估集。

当前正式小数据集的量化基线（11 张，样本很少，只用于回归检查）：

| 日期 | 量化参数 | AP50-95 | AP50 | 备注 |
|---|---|---:|---:|---|
| 2026-07-15 | 浮点 ONNX | 0.1610 | 0.2866 | 6 张 COCO 真值图被拉伸至 3328×1024 |
| 2026-07-15 | ImageNet 20 张校准；检测头 FP16、其余 INT8 | 0.0351 | 0.0668 | 当前主量化产物 |
| 2026-07-15 | 球场 5 张校准；检测头 FP16、其余 INT8 | 0.0042 | 0.0107 | 独立 A/B 产物；在 COCO 域明显下降 |
| 2026-07-15 | 浮点 ONNX | 0.3550 | 0.4399 | 扩充后的 11 张正式 evaluation；`float-eval` 实测 |
| 2026-07-15 | 球场 5 张校准；前处理对齐；检测头 FP16、其余 INT8 | 0.0944 | 0.2487 | RGB、resize/crop、mean/std 与评估一致 |

原始 ONNX 对照评估和 draft 画框：

```bash
./run.sh soccer float-eval
./run.sh soccer float-visualize
```

原始 ONNX 框图分别输出到 `outputs/evaluation/float_visualizations` 和
`outputs/evaluation/draft_float_visualizations`，不会覆盖量化模型结果。

## 2. 目录规范

```text
modes/soccer/datasets/
├── dataset.yaml
├── evaluation/
│   ├── annotations/instances.json # 唯一正式评估标注
│   └── images/                    # 正式评估图片
├── draft/
│   ├── annotations/instances.json # 未审核预标注
│   └── images/                    # 等待人工清洗和标注
├── calibration/
│   └── images/                   # 无需标注，不得与正式评估图片重复
└── reports/
```

约束：

- 原始图片保留原分辨率，不要提前拉伸或覆盖；loader 负责转换到 3328×1024。
- 校准和评估必须使用相同的 RGB、resize/crop 与归一化；当前统一为 RGB、`[0,1]`、
  `mean=[0,0,0]`、`std=[1,1,1]`。
- 支持 `.jpg/.jpeg/.png/.bmp`，推荐 RGB JPEG 或 PNG。
- 评估集和校准集不能使用同一张图片，也不要放相邻视频帧，避免数据泄漏。
- 同一场景连续帧应按场次分组，只选择少量有代表性的帧。
- 正式标注使用 COCO `bbox=[x, y, width, height]`，坐标对应原始图片。
- 文件名必须唯一；管理脚本遇到同名但内容不同的文件会自动增加后缀。

## 3. 图片清洗规范

新增图片前检查：

1. 图片可完整解码，无截断和损坏。
2. 无严重全黑、过曝、失焦或重复帧。
3. 保留典型的远近目标、遮挡、边缘目标、不同光照和不同球场。
4. 不要只挑模型容易识别的图片。
5. 评估图片必须能进行穷举标注；无法判断的图片放校准集，不放评估集。

新增评估数据时不需要运行入箱命令，直接维护以下两个位置：

```text
modes/soccer/datasets/draft/images/
modes/soccer/datasets/draft/annotations/instances.json
```

COCO JSON 中每个 `images[].file_name` 必须能在 `draft/images` 中找到对应图片。
图片和标注未人工审核完成前不要运行 `import-eval`。

## 4. 添加和审核真实评估标注

### 4.1 标注原则

- 必须穷举图片中的目标，不能只修正模型已经检出的框。
- 框应贴合可见目标；严重遮挡时框住可判断的完整目标范围，并保持全数据一致。
- `person`、`sports ball` 等类别使用 COCO 标准英文类名。
- 模糊到无法判断类别的目标不要强行标注。
- 群体目标优先逐人标注；确实无法分离时才使用 `iscrowd=1`。
- 删除重复框、背景误检和错误类别。

### 4.2 使用 CVAT 审核当前 5 张球场图

1. 新建 Detection task，上传 `modes/soccer/datasets/draft/images`。
2. 使用 COCO 80 类标签；至少创建数据实际涉及的类别。
3. 将 `modes/soccer/datasets/draft/annotations/instances.json` 作为 COCO 1.0 预标注导入。
4. 逐张补漏、删误检、修类别、调整边界框。
5. 完成复核后导出为 COCO 1.0，并用导出的 JSON 替换
   `modes/soccer/datasets/draft/annotations/instances.json`。
6. 将审核结果合并进正式数据集：

```bash
./run.sh soccer import-eval --source manual_soccer_review
```

该命令按类别名称映射 category ID，重新分配 image/annotation ID，复制图片并合并到
`evaluation/annotations/instances.json`，不会覆盖已有标注。原始 draft 中仍带有
`review_status=needs_review` 的预标注会被拒绝，不能未经人工审核直接合并。

7. 入口会在导入后自动校验，也可以再次手动校验：

```bash
./run.sh soccer validate
```

校验通过后，才允许提交标注和记录新的评估基线。

## 5. 添加校准图片

校准图片不需要框，但应与真实部署画面同分布。优先使用球场全景、不同光照、不同人数密度，
不要继续使用 ImageNet 随机图片。

```bash
./run.sh soccer add-calibration /path/to/calibration_images
```

添加后修改
`modes/soccer/configs/quant.yaml` 中的
`calibrate_num_sampler`。建议逐步增加到 100～500 张，且不能大于目录中的有效图片数。

## 6. PTQ 量化

检测头不能使用普通 INT8。坐标最大约 3328，而置信度只有 0～1，共用 INT8 scale 会把置信度清零。
必须使用仓库里的检测头 FP16 混合精度配置：

```bash
./run.sh soccer quant
```

输出：

```text
modes/soccer/outputs/quant/yolov5n_falcon2_3328_1024_clean_deploy_model.onnx
modes/soccer/outputs/quant/yolov5n_falcon2_3328_1024_clean_quant_param.yaml
```

每次改变校准图片、observer、位宽或模型后，都要重新评估并记录结果。

当前只有 5 张球场校准图，数量过少。球场校准版在 COCO 小集上更差，暂时不能替换主产物；
必须先完成人工球场真值标注，再根据球场 AP 决定采用哪个版本。

## 7. 正式评估与画框

```bash
./run.sh soccer eval
```

输出：

```text
modes/soccer/outputs/evaluation/
modes/soccer/outputs/evaluation/visualizations/
```

`num_samples: 0` 表示使用全部 approved 图片。数据量较小时 AP 波动很大，既要看 AP，也要逐图检查：

- 是否漏掉远处人物；
- 是否把广告牌、星形图案或篮架识别成目标；
- 框是否偏移；
- 量化结果是否明显差于浮点结果。

无标注图片只做画框时使用：

```bash
./run.sh soccer visualize
```

该命令日志中的 AP=0 没有意义，因为配置为 `visualize_only`。

## 8. 浮点基线

需要区分“模型本身问题”和“量化损失”时，在同一批图片、同一预处理下运行浮点 ONNX：

```bash
python common/evaluation/yolo_coco_metric.py \
  --config modes/soccer/configs/eval.yaml \
  --model modes/soccer/model/yolov5n_falcon2_3328_1024_clean.onnx \
  --num 1000 \
  --conf 0.001
```

量化 AP 应与浮点 AP 同表记录。只记录量化 AP 无法定位精度下降发生在哪个阶段。

## 9. 编译

编译器依赖其随包动态库：

```bash
./run.sh soccer compile
```

编译前确认配置中的模型与量化参数正是刚刚通过评估的版本。不要在评估后再次覆盖量化参数再直接编译。

## 10. 每次迭代的检查清单

1. 新图片进入 inbox 或 calibration，不直接手工散落到 approved。
2. 去重、坏图检查、场景划分完成。
3. 新评估图片完成穷举人工标注。
4. `soccer_dataset.py validate` 通过。
5. 校准集和评估集无重复图片或相邻视频帧。
6. 用 soccer 校准图片重新 PTQ，并带上混合精度 qparam。
7. 运行浮点基线和量化评估。
8. 检查可视化，不只看单一 AP 数字。
9. 在本文“当前状态”表中追加日期、数据版本、浮点 AP、量化 AP 和备注。
10. 仅编译已完成上述检查的量化产物。
