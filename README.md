# Basketball 模型量化与编译端到端运行手册

本仓库用于把 Basketball YOLO ONNX 模型处理为 Statlas 工具链可量化、可评估、可在
VS859 上部署的模型。本文只以 `basketball` 当前目录和配置为主线，覆盖：

```text
准备校准集和评估集
  -> 检查模型与配置
  -> 模型清洗
  -> YOLOv8 DFL 检测头裁切
  -> PTQ / 混合精度量化
  -> 浮点与量化模型评估
  -> StatlasCompile 编译
  -> .mgz 交付检查
```

> **执行前必须先对齐配置。** 当前工作区中的篮球链路并不一致：
>
> - `quant.yaml` 选择的模型名包含 `h1024_w3328`；
> - `eval.yaml` 和 `compile.yaml` 仍引用 `h1920_w1920` 量化产物；
> - `compile.yaml` 引用 head-cut 量化产物，但 `quant` 不会自动执行 `cut-head`；
> - `quant.yaml` 的校准目录当前是 `modes/soccer/datasets/calibration/images`；
> - `modes/basketball/model/` 当前没有可供 `clean-model` 或 `cut-head` 选择的原始 ONNX。
> - 当前 `eval.yaml` 使用 `decode_mode: yolov8_raw`，不能解码 head-cut 的 6 路 4D 输出；
> - 当前校准集与评估集存在一张内容完全相同的图片，正式评估前必须消除数据泄漏。
>
> 不要直接连续执行全流程。先按本文“配置一致性检查”统一模型基名、输入尺寸和数据路径。

## 1. 目录与文件职责

```text
quant_folder/
├── run.sh / run.py                         # 统一命令入口
├── env.example.sh                          # 本机环境变量模板
├── common/
│   ├── evaluation/                         # COCO loader、metric、compare 汇总
│   └── tools/
│       ├── clean_model.py                  # OnnxConvertTool 包装器
│       └── cut_yolov8_head.py              # DFL 检测头裁切
└── modes/basketball/
    ├── model/                              # 原始及处理后的 ONNX
    ├── datasets/
    │   ├── calibration/images/             # PTQ 校准图片
    │   └── evaluation/
    │       ├── images/                     # 正式评估图片
    │       └── annotations/instances.json  # COCO 检测标注
    ├── configs/
    │   ├── quant.yaml                      # PTQ 模型、精度、校准集和前处理
    │   ├── eval.yaml                       # 量化模型评估
    │   ├── compare.yaml                    # 逐层误差比较
    │   └── compile.yaml                    # VS859 编译与硬件前处理
    └── outputs/
        ├── quant/                          # deploy ONNX 和 quant param
        ├── evaluation/                     # AP、可视化、逐层比较
        └── compile/                        # 最终 .mgz
```

源文件与产物要分清：原始 ONNX、图片、COCO JSON 和 YAML 是流程输入；`outputs/` 下的文件是
按配置生成的产物。旧产物存在不代表当前 YAML 仍能生成同名文件。

## 2. 环境配置与命令格式

首次使用时创建本机环境文件：

```bash
cd /home/dragonfly/wj_sdk/quant_folder
cp env.example.sh env.sh
vim env.sh
```

至少确认以下变量：

```bash
export STATLAS_PYTHON=/path/to/conda/env/bin/python
export STATLAS_QUANT=/path/to/conda/env/bin/StatlasQuant
export STATLAS_COMPILE_DIR=/path/to/VS859_ED_release/tools/NPU/statlas
```

`run.sh` 会加载 `env.sh`，再调用：

```text
./run.sh <mode> <operation> [options]
```

篮球模式固定写作 `basketball`，例如：

```bash
./run.sh basketball status
./run.sh basketball validate
```

可用操作及作用：

| 操作 | 作用 | 是否生成模型 |
|---|---|---:|
| `status` | 统计模型、数据、配置和输出文件数 | 否 |
| `validate` | 检查 `eval.yaml` 存在并统计校准/评估目录条目数 | 否 |
| `clean-model` | 使用 Statlas `OnnxConvertTool` 清洗原始 ONNX | 是 |
| `cut-head` | 裁掉 YOLOv8/11 DFL 解码头 | 是 |
| `quant` | 按 `quant.yaml` 做 PTQ | 是 |
| `eval` | 按 `eval.yaml` 评估量化模型 | 否 |
| `float-eval` | 用原始 ONNX 做浮点基线评估 | 否 |
| `compare` | 比较开启/关闭量化时的逐层输出 | 否 |
| `compile` | 按 `compile.yaml` 生成 VS859 `.mgz` | 是 |
| `clean` | 清理指定范围的生成物 | 删除生成物 |
| `all` | 顺序执行 `quant + eval + float-eval + compile` | 是 |

先打印命令、不实际执行：

```bash
./run.sh basketball quant --dry-run
./run.sh basketball compile --dry-run
```

## 3. 准备数据

### 3.1 新增校准集

校准图片不需要标注，用于统计激活分布。手工把图片放到：

```text
modes/basketball/datasets/calibration/images/
```

选择原则：

- 与真实部署摄像头、分辨率、视角、光照和目标密度同分布；
- 覆盖近景/远景、遮挡、空场、多人、球和篮筐等典型状态；
- 删除损坏、全黑、严重模糊、重复或几乎相同的连续帧；
- 不要只选择模型容易识别的图片；
- 不得与评估集重复，也应避免来自同一视频的相邻帧跨集合出现；
- 支持常见的 `.jpg`、`.jpeg`、`.png`、`.bmp`，建议统一使用可完整解码的 RGB JPEG/PNG。

当前 `quant.yaml` 的 `calibrate_num_sampler` 是 `10`。下面是切换到 Basketball 校准目录并将
采样目标提高到 `100` 的**建议修改示例**，不是当前文件原值：

```yaml
dataset:
  calibrate:
    root: modes/basketball/datasets/calibration/images
    calibrate_num_sampler: 100
    batch_size: 1
    num_worker: 1
```

约束：

- `root` 必须指向实际篮球校准集。当前配置仍指向 soccer，运行前必须修改；
- `calibrate_num_sampler` 必须小于或等于目录中可解码图片数；
- 数据变化后必须重新执行 `quant -> eval -> compare -> compile`；
- 少量校准图只能用于打通流程，正式 PTQ 应覆盖足够多的真实部署场景。

查看当前数量：

```bash
./run.sh basketball validate
./run.sh basketball status
```

当前 `validate` 会报告 `calibration images: 145`，但这是 `glob('*')` 的目录条目数，其中包含
`.gitkeep`；当前可识别图片实际为 144 张。不要直接把 `validate` 输出当成采样数上限。

### 3.2 新增评估集

评估数据由图片和 COCO 检测标注组成，直接维护：

```text
modes/basketball/datasets/evaluation/images/
modes/basketball/datasets/evaluation/annotations/instances.json
```

当前篮球类别是：

| category_id | name |
|---:|---|
| 1 | `person` |
| 2 | `ball` |
| 3 | `hoop` |
| 4 | `ballhoop` |

新增图片时必须同步合并 COCO JSON。最小结构示例：

```json
{
  "images": [
    {
      "id": 51,
      "file_name": "new_game_frame_001.jpg",
      "width": 3840,
      "height": 1920
    }
  ],
  "annotations": [
    {
      "id": 1440,
      "image_id": 51,
      "category_id": 2,
      "bbox": [1200.0, 430.0, 28.0, 30.0],
      "area": 840.0,
      "iscrowd": 0,
      "segmentation": []
    }
  ],
  "categories": [
    {"id": 1, "name": "person", "supercategory": "person"},
    {"id": 2, "name": "ball", "supercategory": "ball"},
    {"id": 3, "name": "hoop", "supercategory": "hoop"},
    {"id": 4, "name": "ballhoop", "supercategory": "ballhoop"}
  ]
}
```

合并规则：

- `images[].id` 和 `annotations[].id` 在整个 JSON 内唯一；
- `images[].file_name` 与 `evaluation/images/` 下文件名完全一致；
- `width`、`height` 是原图尺寸，不是模型 resize 后的尺寸；
- `annotations[].image_id` 必须引用存在的 `images[].id`；
- `annotations[].category_id` 必须引用存在的 `categories[].id`；
- COCO `bbox` 是 `[x, y, width, height]`，不是 `[x1, y1, x2, y2]`；
- `area` 对普通矩形框可写为 `width * height`，`iscrowd` 通常为 `0`；
- 每张评估图必须穷举标注目标，不能只保留模型已经检出的框；
- 评估集应覆盖真实难例，但不能与校准集或训练集发生数据泄漏。

当前数据已经发现一处泄漏，下面两个文件 SHA-256 相同：

```text
modes/basketball/datasets/calibration/images/clip-160_frame-33440.jpg
modes/basketball/datasets/evaluation/images/VID_20250324_192214_00_018_019_clip-160_frame-33440.jpg
```

在正式量化和验收前，应从校准集或评估集中移除其中一个，并相应维护 COCO JSON。不要直接删除
评估图片而保留对应的 `images`/`annotations` 记录。

先检查 JSON 语法：

```bash
python3 -m json.tool \
  modes/basketball/datasets/evaluation/annotations/instances.json \
  >/dev/null
```

再运行：

```bash
./run.sh basketball validate
```

注意：篮球模式当前没有专用 `tools/dataset.py`，因此 `validate` 只会确认 `eval.yaml` 存在并统计
校准、评估目录中的条目数。它**不会**检查图片能否解码、COCO ID 是否重复、标注是否越界或
`file_name` 是否全部存在。正式评估前仍需通过标注工具或单独的 COCO 校验脚本完成深度检查。

## 4. 模型准备

### 4.1 放置原始 ONNX

把待处理模型放入：

```text
modes/basketball/model/
```

`run.py` 要求该目录中恰好有一个“原始 ONNX”。以下后缀被视为已处理文件，不参与原始模型选择：

```text
_headcut_raw.onnx
_clean.onnx
_calibrated_model.onnx
_deploy_model.onnx
_simplified.onnx
_opset13.onnx
_fp32.onnx
```

如果没有原始 ONNX，或存在多个不带上述后缀的 ONNX，`clean-model`、`cut-head` 和
`float-eval` 会拒绝执行。当前工作区的篮球 `model/` 为空，必须先放入模型。

### 4.2 模型清洗

模型清洗通过 StatlasQuant 环境中的 `OnnxConvertTool` 处理 ONNX 结构：

```bash
./run.sh basketball clean-model
```

输出与原模型同目录，命名为：

```text
<raw_model_stem>_clean.onnx
```

例如：

```text
modes/basketball/model/basketball_yolov8_raw.onnx
  -> modes/basketball/model/basketball_yolov8_raw_clean.onnx
```

注意：

- 清洗不会修改 `quant.yaml`；需要手工把 `model.onnx_model` 指向清洗产物；
- `_clean.onnx` 不再被识别为“原始模型”，避免重复清洗；
- 只有模型来源明确、图结构已被当前 Statlas 工具链验证时，才可以跳过清洗；
- 清洗后仍应使用 ONNX checker/推理对比确认输入输出名称、shape 和数值行为没有意外变化。

### 4.3 YOLOv8 DFL 检测头裁切

篮球 YOLOv8 检测头包含 DFL Softmax、dist2bbox 和展平后的大网格计算。当前 Statlas 编译器可能
无法对这些算子完成 tile，因此编译用模型通常需要在 4D 检测分支输出处裁掉解码头：

```bash
./run.sh basketball cut-head
```

对检测到 DFL 结构的模型，生成：

```text
<raw_model_stem>_headcut_raw.onnx
<raw_model_stem>_headcut_raw_spec.yaml
```

`_spec.yaml` 记录各尺度输出名、shape、stride、`reg_max` 和类别数。平台运行时需要在 host 端完成：

```text
box 分支 DFL Softmax
  -> dist2bbox + stride
  -> class 分支 sigmoid
  -> 置信度过滤
  -> NMS
```

重要限制：

- `cut-head` 默认裁切的是 `run.py` 识别到的唯一原始 ONNX，不会自动选择 `_clean.onnx`；
- 如果必须裁切清洗后的模型，应直接调用辅助脚本并明确输入输出：

```bash
"${STATLAS_PYTHON}" common/tools/cut_yolov8_head.py \
  --input_model modes/basketball/model/<model>_clean.onnx \
  --output_model modes/basketball/model/<model>_clean_headcut_raw.onnx
```

- `cut-head` 不会修改 `quant.yaml`；要量化裁头模型，必须手工选择该 ONNX；
- 如果脚本没有发现匹配的 DFL 结构，会以成功状态结束但不写 head-cut 模型；必须检查输出日志和文件；
- head-cut 模型输出的是多尺度原始 4D feature maps，不再是最终检测框。

## 5. 配置一致性检查

在执行任何耗时操作前，先确定一个唯一的“本次构建身份”：

```text
模型基名: 例如 basketball_yolov8_h1024_w3328_headcut
输入尺寸: 1024 x 3328
量化输入: 对应的 clean/headcut ONNX
量化输出: 同一基名的 deploy ONNX + quant param YAML
编译输出: 能追溯到该基名和输入格式的 .mgz
```

三个配置必须形成同一条链：

| 阶段 | 配置字段 | 必须指向/匹配 |
|---|---|---|
| PTQ | `quant.yaml:model.onnx_model` | 本次选定的 clean 或 head-cut ONNX |
| PTQ | `dataset.transform.resize/crop` | 模型 NCHW 的 H、W |
| PTQ | `dataset.calibrate.root` | Basketball 校准集 |
| 量化评估 | `eval.yaml:model.onnx_model` | 本次 PTQ 生成的 `_deploy_model.onnx` |
| 量化评估 | `eval.yaml:model.quant_param` | 同一次 PTQ 生成的 `_quant_param.yaml` |
| 量化评估 | `resize_size/crop_size/img_size` | 与 PTQ 输入尺寸一致 |
| 量化评估 | `metric.parameters.decode_mode` | 与 deploy ONNX 输出形态一致 |
| 编译 | `compile.yaml:model` | 通过评估的 deploy ONNX |
| 编译 | `compile.yaml:quantize` | 与 deploy ONNX 同一次生成的 quant param |
| 编译 | `preprocess.pre_shape/pre_stride` | 与模型输入及 runtime buffer 一致 |

### 当前配置中的已知不一致

当前文件不能直接串行使用：

1. `quant.yaml` 的源模型名包含 `h1024_w3328`，激活为 `int16`；
2. `eval.yaml` 引用不含 `opset13_fp32_raw_headcut_raw` 的 `h1920_w1920` 产物；
3. `compile.yaml` 引用含 `opset13_fp32_raw_headcut_raw` 的 `h1920_w1920` 产物；
4. `quant.yaml` 的校准 root 指向 soccer；
5. `compare.yaml` 引用另一组 `h1920_w1920..._clean` 产物，代表图片
   `clip-0_frame-206.jpg` 当前也不在篮球评估目录；
6. `eval.yaml` 当前是 `yolov8_raw`，而 `compile.yaml` 选择的是 head-cut 产物，两者输出形态不匹配；
7. 校准集和评估集包含一张内容相同的图片；
8. `outputs/quant/` 中已有的旧产物不能证明当前 `quant.yaml` 会重新生成同名文件。

只更新 README，不自动修改这些 YAML。执行者必须先选择本次模型，然后逐项改成同一基名和尺寸。

建议先检查所有关键路径：

```bash
rg -n "onnx_model|quant_param|^model:|^quantize:|resize|crop|img_size|pre_shape|pre_stride|root:" \
  modes/basketball/configs/*.yaml

./run.sh basketball quant --dry-run
./run.sh basketball compile --dry-run
```

`--dry-run` 只证明入口将调用哪个 YAML，不会替你检查 YAML 内部的模型身份是否一致。

## 6. 模型量化

### 6.1 `quant.yaml` 参数说明

当前文件的采样数是 `10`、校准 root 是 soccer。下面展示的是保留当前 activation `int16` / weight
`int8` 精度策略、但把数据路径和建议采样目标改为 Basketball 后的**推荐结构**：

```yaml
model:
  onnx_model: /absolute/or/repo/relative/path/to/model.onnx

work_mode: PTQ

quant:
  activation:
    observer: MinMaxObserver
    dtype: int16
    symmetry: true
  weight:
    observer: MinMaxObserver
    dtype: int8
    symmetry: true
    per_channel: true

dataset:
  calibrate:
    root: modes/basketball/datasets/calibration/images
    calibrate_num_sampler: 100
    batch_size: 1
    num_worker: 1
  transform:
    resize: [1024, 3328]
    crop: [1024, 3328]
    mean: [0.0, 0.0, 0.0]
    std: [1.0, 1.0, 1.0]

out_dir: modes/basketball/outputs/quant/
```

字段含义：

| 字段 | 说明 |
|---|---|
| `model.onnx_model` | PTQ 的唯一输入模型；`quant` 不会自动清洗或裁头 |
| `work_mode: PTQ` | 后训练量化（Post-Training Quantization）流程 |
| `observer` | 用校准样本统计张量动态范围的方法；当前为 min/max |
| `dtype` | 量化数据类型；当前 activation `int16`、weight `int8` |
| `symmetry` | 使用对称量化范围，零点通常固定在对称中心 |
| `per_channel` | 权重按输出通道分别统计 scale，通常比 per-tensor 更精确 |
| `calibrate.root` | 校准图片目录 |
| `calibrate_num_sampler` | 实际参与统计的样本数，不能超过有效图片数 |
| `batch_size` | 校准 batch；大分辨率模型通常保持 1 |
| `num_worker` | 数据加载进程/线程数量，需结合内存和环境稳定性设置 |
| `resize/crop` | 模型输入 `[H, W]`，必须与 ONNX 和后续编译一致 |
| `mean/std` | 输入归一化；当前配置表示 RGB 转 tensor 后不额外平移缩放 |
| `out_dir` | deploy ONNX 和 quant param 的输出目录 |

### 6.2 混合精度量化

当前 Basketball 配置已经使用不同数据类型：激活 `int16`、权重 `int8`。这是全局按张量类别设置的
混合位宽，不等同于“只把少数敏感层提升到 16 bit”的逐层混合精度。

`run.py` 的规则是：如果存在
`modes/basketball/configs/mixed_precision.yaml`，执行 `quant` 时会自动追加：

```text
--qparam_cfg modes/basketball/configs/mixed_precision.yaml
```

当前篮球目录没有该文件。确实需要逐层保护时，可根据 `compare` 结果和工具链支持创建，例如：

```yaml
layers:
  - layername: <exact_onnx_tensor_or_layer_name>
    nbit: 16
    observer: MinMaxObserver
    symmetry: true
    per_channel: false
    round_type: None
    static: 1
```

使用原则：

- `layername` 必须与当前 ONNX 完全一致，换模型或重新导出后要重新核对；
- 优先保护量化误差明显、且对最终检测敏感的层，不要无依据地把大量层升到 16 bit；
- 每次修改层列表后都要重新量化、评估、compare 和编译；
- 混合精度会增加精度、性能、内存和带宽之间的权衡，最终以目标板实测为准；
- 不要把“全局 activation int16”与“逐层 qparam override”混为一谈。

### 6.3 执行 PTQ

确认配置一致后：

```bash
./run.sh basketball quant
```

典型输出：

```text
modes/basketball/outputs/quant/<model>_deploy_model.onnx
modes/basketball/outputs/quant/<model>_quant_param.yaml
```

量化完成后记录这两个文件的准确路径，并同步写入 `eval.yaml`、`compare.yaml` 和
`compile.yaml`。以下变化都要求重新量化：

- 原始/清洗/head-cut 模型变化；
- 校准图片或采样数变化；
- resize、crop、mean、std 变化；
- observer、dtype、symmetry、per-channel 变化；
- `mixed_precision.yaml` 变化。

## 7. 模型评估

评估的目的不是只得到一个 AP，而是区分“模型本身误差”和“量化新增误差”。浮点与量化评估必须
使用同一批图片、同一 COCO 标注、同一输入尺寸、同一归一化和同一种输出解码方式。

输出形态与 decoder 的对应关系：

| 模型输出 | 特征 | `decode_mode` |
|---|---|---|
| 完整 YOLOv8 解码输出 | 单个 `[B, 4+nc, N]` 张量 | `yolov8_raw` |
| head-cut DFL 输出 | 3 个尺度的 box/cls，共 6 路 4D feature maps | `yolov8_headcut` |

这两个 decoder 不能互换。当前 `eval.yaml` 是 `yolov8_raw`，如果量化和编译主线改成 head-cut，
必须同步改成 `yolov8_headcut`。

### 7.1 量化模型评估

先把 `eval.yaml` 的 `onnx_model` 和 `quant_param` 改为本次量化产物，然后执行：

```bash
./run.sh basketball eval
```

主要配置：

- `dataset.eval.parameters.ann_file/img_dir`：COCO JSON 和图片；
- `normalize_mean/std`、`resize_size/crop_size`：必须与 PTQ 一致；
- `num_samples: 0`：使用全部评估图片；
- `decode_mode`：完整输出使用 `yolov8_raw`，head-cut 6 路输出使用 `yolov8_headcut`；
- `conf_threshold`、`iou_threshold`：指标计算阈值；
- `vis_dir`、`vis_num`：框图输出目录和数量；
- `result_file`：指标 CSV 文件名。

输出位于：

```text
modes/basketball/outputs/evaluation/metric_result.csv
modes/basketball/outputs/evaluation/visualizations/
```

### 7.2 浮点基线

`./run.sh basketball float-eval` 强制使用 `modes/basketball/model/` 中被识别为唯一原始 ONNX，
并与 `eval` 共用同一个 `eval.yaml`。因此它只适合“原始 ONNX 与量化 deploy ONNX 输出形态相同”
的情况，例如两者都是完整 YOLOv8 输出并共同使用 `yolov8_raw`：

```bash
./run.sh basketball float-eval
```

对于本文编译主线使用的 head-cut 模型，量化评估必须用 `yolov8_headcut`，但完整原始 ONNX 通常
需要 `yolov8_raw`，不能共用一个 decoder。要得到可比的 head-cut 浮点基线，应直接评估同一份
浮点 head-cut ONNX：

```bash
"${STATLAS_PYTHON}" common/evaluation/yolo_coco_metric.py \
  --config modes/basketball/configs/eval.yaml \
  --model modes/basketball/model/<model>_headcut_raw.onnx \
  --num 0 \
  --vis-dir modes/basketball/outputs/evaluation/float_headcut_visualizations
```

此时 `eval.yaml` 必须使用 `decode_mode: yolov8_headcut`，且其数据路径和输入尺寸与量化评估一致。

还有一个实现差异：StatlasQuant 的 `eval` loader 会读取 `normalize_mean/std`，而独立浮点评估脚本
当前只执行 resize、center crop 和 `ToTensor()`，不会应用这两个字段。当前 mean=`0`、std=`1`
时两条路径等价；如果改成其他归一化，必须先修改浮点评估实现或使用其他等价评估方式，否则 AP
不能直接比较。

使用 `./run.sh basketball float-eval` 时，浮点可视化默认位于：

```text
modes/basketball/outputs/evaluation/float_visualizations/
```

### 7.3 逐层量化误差

把 `compare.yaml` 的模型、quant param 和代表图片改成本次构建后执行：

```bash
./run.sh basketball compare
```

生成：

```text
modes/basketball/outputs/evaluation/compare/layer_compare.csv
modes/basketball/outputs/evaluation/compare/layer_compare_sorted.csv
modes/basketball/outputs/evaluation/compare/REPORT.md
```

报告按 `cosine_error = 1 - cosine_similarity` 排序。误差越大，说明该层开启量化前后的输出方向
差异越明显，但它不是最终 AP 贡献的直接因果证明。应结合 AP 下降、框图表现和层位置判断是否
需要调整 observer、校准集或逐层精度。

### 7.4 评估验收

至少检查：

- 浮点与量化 AP50-95、AP50 的绝对下降和相对下降；
- person、ball、hoop、ballhoop 各类别的样本量和 AP；
- 小球、远处人物、遮挡目标、画面边缘目标是否明显退化；
- 是否出现置信度整体被压低、框偏移、重复框或背景误检；
- compare 最差层是否集中在检测 neck/head；
- 评估集规模是否足以支持当前结论。

只有量化评估通过后，才应把同一组 deploy ONNX 和 quant param 交给编译阶段。

## 8. 模型编译

### 8.1 编译前检查

确认 `modes/basketball/configs/compile.yaml`：

1. `model` 是刚通过评估的 deploy ONNX；
2. `quantize` 是与该 ONNX 同一次 PTQ 生成的 quant param；
3. 两个文件模型基名一致；
4. `pre_shape` 与 ONNX 输入 shape 一致；
5. runtime 输入格式、stride、色彩范围与部署代码一致；
6. mean/std 与 PTQ、评估一致；
7. 输出 `.mgz` 名称能区分模型版本、尺寸和输入格式。

然后执行：

```bash
./run.sh basketball compile
```

`run.py` 会设置编译器 `lib/` 到 `LD_LIBRARY_PATH`，并执行：

```text
${STATLAS_COMPILE_DIR}/StatlasCompile -c modes/basketball/configs/compile.yaml
```

### 8.2 编译参数说明

| 字段 | 当前值/形式 | 说明 |
|---|---|---|
| `model` | deploy ONNX 路径 | 必须是本次评估通过的量化 deploy 模型 |
| `quantize` | quant param YAML 路径 | 必须与 `model` 成对，不能混用不同 PTQ 产物 |
| `output` | `outputs/compile/*.mgz` | 最终平台模型输出路径 |
| `optimze_level` | 当前 `2` | 编译优化等级；字段按工具实际拼写，不能自行改成 `optimize_level` |
| `target` | 当前 `VS859` | 目标芯片/平台，必须与部署硬件一致 |
| `mode` | 当前 `0` | 编译器模式选择；含义依当前 StatlasCompile 版本，未经版本文档确认不要修改 |
| `cluster_id` | 当前 `0` | 目标 cluster 编号 |
| `core_num` | 当前 `1` | 使用的 NPU core 数，改变后需重新测性能和资源占用 |
| `preprocess_en` | 当前 `1` | 是否把输入预处理编入 NPU 模型 |
| `model_input_type` | `0=BGR, 1=RGB` | 模型训练/量化所期望的颜色通道顺序 |
| `pre_input_type` | 见下表 | runtime 实际传入的内存格式；当前篮球为 `4=NV12` |
| `pre_color_range` | 见下表 | YUV/RGB 转换使用的色彩标准和 full/limited range |
| `pre_shape` | NCHW | 模型输入 shape，必须与 ONNX 及 PTQ 尺寸一致 |
| `pre_stride` | 当前为四维列表 | runtime 输入 buffer 的 stride 描述，必须与实际内存布局一致 |
| `mean/std` | 当前全 0 / 全 1 | 硬件前处理归一化，必须与量化配置一致 |
| `output_tensor_cast` | 可选 | 部分配置用于控制输出张量转换；当前篮球配置未设置，需按编译器版本确认后使用 |

`pre_input_type` 枚举来自当前配置注释：

| 值 | 输入格式 |
|---:|---|
| 0 | RGB packed |
| 1 | RGB planar |
| 2 | BGR packed |
| 3 | BGR planar |
| 4 | NV12 |
| 5 | NV21 |
| 6 | NV16 |
| 7 | NV61 |

`pre_color_range` 枚举：

| 值 | 色彩范围 |
|---:|---|
| 0 | BT.709 full |
| 1 | BT.709 limited |
| 2 | BT.601 full |
| 3 | BT.601 limited |

当前篮球 `pre_input_type: 4`，即 NV12，因此 `pre_color_range` 会影响 YUV 到 RGB 的颜色解释。
配置中的旧注释如果仍写“RGB->RGB 不涉及 YUV”，不能作为当前 NV12 配置的依据；应以 runtime
实际产生的 NV12 标准为准。

当前 `pre_shape/pre_stride` 写的是 `[1, 3, 1024, 3328]`，而 `model/quantize` 文件名仍包含
`h1920_w1920`。这正是必须在编译前解决的尺寸冲突。不要仅因为旧 `.mgz` 已存在就跳过检查。

### 8.3 编译产物验收

输出位于：

```text
modes/basketball/outputs/compile/<model>.mgz
```

在目标板至少验证：

- `.mgz` 可加载，输入/输出 tensor 数量与预期一致；
- NV12 buffer 的宽、高、stride 和色彩范围正确；
- host 端按 `_headcut_raw_spec.yaml` 对各尺度输出完成 DFL、decode 和 NMS；
- 同一测试图片上，目标板结果与 PC 量化评估结果在可接受误差内；
- 延迟、内存、core 使用和长时间运行稳定性满足部署要求。

## 9. 状态、清理与重跑

查看各阶段文件数：

```bash
./run.sh basketball status
```

预览清理范围：

```bash
./run.sh basketball clean --scope quant --dry-run
./run.sh basketball clean --scope eval --dry-run
./run.sh basketball clean --scope compile --dry-run
./run.sh basketball clean --scope all --dry-run
```

确认后去掉 `--dry-run`。`clean` 只清理 `outputs/` 对应范围，不删除 `model/`、配置或数据集。

不建议在当前配置未对齐时使用：

```bash
./run.sh basketball all
```

`all` 不会自动执行 `clean-model`、`cut-head` 或修正 YAML，只是依次运行：

```text
quant -> eval -> float-eval -> compile
```

在 head-cut 主线上还存在 decoder 冲突：`eval` 需要 `yolov8_headcut`，而 `float-eval` 强制使用
完整原始 ONNX，通常需要 `yolov8_raw`。因此 head-cut 构建不要使用 `all`；应分别运行 `quant`、
`eval`、上文的 head-cut 浮点基线命令和 `compile`。

## 10. 常见问题

### `No raw ONNX found`

`model/` 中没有符合原始模型命名规则的 ONNX。放入原始导出，或检查是否所有文件都带有
`_clean`、`_opset13`、`_fp32` 等已处理后缀。

### `Multiple raw ONNX found`

`model/` 中有多个候选原始 ONNX。每次构建只保留一个候选，其他模型移到子目录或使用明确的
已处理后缀，避免工具猜错。

### `No deploy model found` 或路径不存在

先检查 `quant.yaml` 是否已成功运行，再检查 `eval.yaml`、`compare.yaml`、`compile.yaml` 是否仍
引用旧基名。入口不会自动选择最新产物。

### `cut-head` 没有生成文件

脚本未找到 `[1, reg_max, 4, N]` DFL Softmax 或 4D->3D 展平切点。查看日志确认模型是否属于需要
裁切的 YOLOv8/11 DFL 结构；不要把“命令退出码为 0”误判为“文件一定生成”。

### 量化 AP 明显低于浮点 AP

优先排查数据和前处理是否完全一致、校准集是否覆盖部署分布、activation 位宽是否合适，以及
compare 最差层。不要直接用提高大量层精度掩盖错误的数据路径或尺寸配置。

### 编译成功但目标板结果错误

重点核对 runtime 输入格式、NV12 色彩范围、stride、预处理 mean/std、head-cut 输出顺序和 host
解码参数。编译成功只证明图可编译，不证明端到端输入输出语义正确。

## 11. 交付检查清单

1. `model/` 中本次原始 ONNX 身份明确，清洗和 head-cut 产物可追溯。
2. 校准集与部署分布一致，且与训练/评估集没有泄漏。
3. 新增评估图片已完整合并到 COCO JSON，类别 ID、bbox 和图片尺寸正确。
4. `quant.yaml` 的模型、校准 root、采样数、输入尺寸和归一化正确。
5. 全局 activation/weight 位宽和可选逐层混合精度配置经过评估验证。
6. deploy ONNX 与 quant param 来自同一次 PTQ。
7. `eval.yaml`、`compare.yaml`、`compile.yaml` 已改成同一模型基名和输入尺寸。
8. 浮点 AP、量化 AP、逐类指标、框图和 compare 报告均已检查并记录。
9. `compile.yaml` 的 NV12 格式、range、shape、stride、mean/std 与 runtime 一致。
10. `.mgz` 已在 VS859 目标板完成加载、精度、性能和稳定性验证。

## 12. Git 管理约定

- 提交代码、配置、文档、COCO 标注，以及团队约定需要版本化的数据清单；
- 原始 ONNX、量化产物、评估结果、可视化、日志和 `.mgz` 通常不提交，可由流程重建；
- 提交前检查 YAML 中是否残留仅适用于个人机器的绝对路径；
- 不要通过提交旧产物来掩盖配置链路不一致，模型身份应能由配置和构建记录追溯。
