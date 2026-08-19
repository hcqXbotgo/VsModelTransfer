# Basketball 模型量化与双平台编译端到端运行手册

本仓库用于把 Basketball YOLO ONNX 模型处理为可量化、可评估，并可部署到
VS859 或 RK3576 的模型。本文以 `basketball` 当前配置为主线，并在各平台章节中说明
其他模式的构建方式与差异。完整流程为：

```text
准备校准集和评估集
  -> 检查模型与配置
  -> 模型清洗
  -> YOLOv8 DFL 检测头裁切
  -> PTQ / 混合精度量化
  -> 浮点与量化模型评估
  -> VS859: StatlasQuant + StatlasCompile -> .mgz
  -> RK3576: RKNN Toolkit2 量化/转换 -> .rknn
  -> 目标板交付检查
```

> **执行前必须先对齐配置。** VS859 的 `configs/vs859/quant.yaml`、`eval.yaml` 和
> `compile.yaml` 必须使用
> 同一个模型基名和输入尺寸，不能把旧量化参数、不同尺寸的 deploy ONNX 或其他模型的
> 编译配置混在同一次构建中。
>
> 当前 Basketball 输入尺寸和校准目录已经统一为 `1024 x 3328` 与
> `modes/basketball/datasets/calibration/images`。VS859 的三个配置当前均走 head-cut
> 分支，评估使用 `yolov8_headcut` 解码；量化前基线也必须使用同一 head-cut 输出形态。
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
│       ├── cut_yolo_head.py                 # YOLOv5/v8/v11/26 裁切分发器
│       ├── cut_yolov5_head.py               # YOLOv5 三尺度 NHWC 输出裁切
│       ├── cut_yolov8_head.py               # YOLOv8/11 DFL 检测头裁切
│       └── cut_yolo26_head.py                # YOLO26 三尺度 box/class 分支裁切
└── modes/basketball/
    ├── model/                              # 原始及处理后的 ONNX
    ├── datasets/
    │   ├── calibration/images/             # PTQ 校准图片
    │   └── evaluation/
    │       ├── images/                     # 正式评估图片
    │       └── annotations/instances.json  # COCO 检测标注
    ├── configs/
    │   ├── vs859/
    │   │   ├── quant.yaml                  # Statlas PTQ
    │   │   ├── eval.yaml                   # 量化模型评估
    │   │   ├── compare.yaml                # 逐层误差比较
    │   │   └── compile.yaml                # MGZ 编译与硬件前处理
    │   └── rk3576/
    │       ├── rknn.yaml                   # RKNN 量化、转换和校准配置
    │       └── eval.yaml                   # RKNN COCO 评估与解码配置
    └── outputs/
        ├── quant/                          # deploy ONNX 和 quant param
        ├── evaluation/                     # AP、可视化、逐层比较
        └── compile/
            ├── vs859/                     # StatlasCompile 生成的 .mgz
            └── rk3576/                    # RKNN、manifest 和校准列表
```

源文件与产物要分清：原始 ONNX、图片、COCO JSON 和 YAML 是流程输入；`outputs/` 下的文件是
按配置生成的产物。旧产物存在不代表当前 YAML 仍能生成同名文件。

## 2. 环境配置与命令格式

### 2.1 公共命令格式

`run.sh` 会加载仓库根目录的 `env.sh`，再调用：

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
| `validate` | 检查 VS859 `eval.yaml` 存在并统计校准/评估目录条目数 | 否 |
| `clean-model` | 使用 Statlas `OnnxConvertTool` 清洗原始 ONNX | 是 |
| `cut-head` | 裁掉 YOLOv5 或 YOLOv8/11 的主机后处理部分 | 是 |
| `quant` | 默认做 VS859 PTQ；`--platform rk3576` 量化并生成 RKNN | 是 |
| `eval` | 默认评估 VS859；`--platform rk3576` 评估 RKNN | 否 |
| `float-eval` | 用原始 ONNX 做浮点基线评估 | 否 |
| `compare` | 默认比较 VS859 逐层输出；`--platform rk3576` 运行 RKNN 逐层误差分析 | 否 |
| `compile` | 默认生成 VS859 `.mgz`；指定平台后生成 RK3576 `.rknn` | 是 |
| `clean` | 清理指定范围的生成物 | 删除生成物 |
| `all` | 顺序执行 `quant + eval + float-eval + VS859 compile` | 是 |

先打印命令、不实际执行：

```bash
./run.sh basketball quant --dry-run
./run.sh basketball compile --platform vs859 --dry-run
./run.sh basketball compile --platform rk3576 --dry-run
```

### 2.2 VS859 环境

VS859 的清洗、Statlas PTQ、评估和 MGZ 编译依赖以下变量：

```bash
cd /home/dragonfly/wj_sdk/quant_folder
cp env.example.sh env.sh
vim env.sh
export STATLAS_PYTHON=/path/to/conda/env/bin/python
export STATLAS_QUANT=/path/to/conda/env/bin/StatlasQuant
export STATLAS_COMPILE_DIR=/path/to/VS859_ED_release/tools/NPU/statlas
```

### 2.3 RK3576 环境

RK3576 转换只要求 `RKNN_PYTHON` 指向能够导入 `rknn.api` 的 Python：

```bash
export RKNN_PYTHON=/path/to/rknn-toolkit2-env/bin/python
```

仓库内 Toolkit2 位置为：

```text
dependencies/rknn-toolkit2-2.3.2/
```

RKNN 环境使用该目录的 Toolkit2 2.3.2 wheel，并固定 `onnx==1.16.2`；更高版本 ONNX
删除了 Toolkit2 2.3.2 仍会访问的 `onnx.mapping`。

### 2.4 自动创建两套环境

环境脚本按仓库内相对路径创建相互独立的 Conda 环境，并更新本机 `env.sh`：

```bash
./setup_conda_envs.sh
```

默认依次创建 `dependencies/conda-envs/statlas`（Python 3.8）和
`dependencies/conda-envs/rknn`（Python 3.9），然后把实际解释器路径写入 `env.sh`。
可先检查命令，或只安装一种环境：

```bash
./setup_conda_envs.sh --dry-run
./setup_conda_envs.sh --statlas-only
./setup_conda_envs.sh --rknn-only
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

当前 Basketball `quant.yaml` 的 `calibrate_num_sampler` 是 `20`。下面是保持当前
Basketball 校准目录、将采样目标提高到 `100` 的**建议修改示例**，不是当前文件原值：

```yaml
dataset:
  calibrate:
    root: modes/basketball/datasets/calibration/images
    calibrate_num_sampler: 100
    batch_size: 1
    num_worker: 1
```

约束：

- `root` 必须指向实际篮球校准集；当前 Basketball 配置已指向上述目录；
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

注意：篮球模式当前没有专用 `tools/dataset.py`，因此 `validate` 只会确认
`configs/vs859/eval.yaml` 存在并统计
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
`float-eval` 会拒绝执行。当前 Basketball 模型目录已经包含一个完整 raw ONNX 和对应的
head-cut ONNX；重新替换模型时仍须遵守唯一候选规则。

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

- 清洗不会修改任何平台配置；需要手工把 VS859 `quant.yaml` 或 RK3576 `rknn.yaml` 的
  `model.onnx_model` 指向清洗产物；
- `_clean.onnx` 不再被识别为“原始模型”，避免重复清洗；
- 只有模型来源明确、图结构已被当前 Statlas 工具链验证时，才可以跳过清洗；
- 清洗后仍应使用 ONNX checker/推理对比确认输入输出名称、shape 和数值行为没有意外变化。

### 4.3 YOLOv5/v8/v11 检测头裁切

篮球 YOLOv8 检测头包含 DFL Softmax、dist2bbox 和展平后的大网格计算。当前 Statlas 编译器可能
无法对这些算子完成 tile，因此编译用模型通常需要在 4D 检测分支输出处裁掉解码头：

```bash
./run.sh basketball cut-head
```

同一个 `cut-head` 命令也支持 YOLOv5。分发器会识别三尺度 Detect 头，在每个
`Reshape -> Transpose -> Sigmoid` 分支后裁切，并生成按 stride `8/16/32` 排序的
三个 NHWC 输出：

```text
[1, H/8,  W/8,  3*(5+nc)]
[1, H/16, W/16, 3*(5+nc)]
[1, H/32, W/32, 3*(5+nc)]
```

这些张量保留 sigmoid，但不包含 grid、stride、anchor 解码，符合嵌入式
`YoloV5PostProcessor::process_native_nhwc` 的输入契约。不能直接把完整 YOLOv5
末端的三个 `Concat` 设为输出，因为它们已经完成坐标解码，C++ 会重复解码。

例如 `demo_v5` 的输出为 `[1,88,160,27]`、`[1,44,80,27]`、
`[1,22,40,27]`。裁切器只生成模型和 `_spec.yaml`，不会自动修改
VS859 的 `quant.yaml`、`eval.yaml`、`compile.yaml` 或 RK3576 的 `rknn.yaml`；
现有 `decode_mode: yolov5` 仍只适用于
完整单输出模型。

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

- `cut-head` 优先裁切唯一的 `_clean.onnx`；没有清洗模型时，才选择唯一的原始 ONNX；
- 同时存在多个 `_clean.onnx` 或多个候选原始 ONNX 时会报错，必须先移走无关候选或明确保留本次构建模型；
- `cut-head` 不会修改任何配置；要量化裁头模型，必须在目标平台配置中手工选择该 ONNX；
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

VS859 的三个配置必须形成同一条链：

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

RK3576 使用独立的 `configs/rk3576/rknn.yaml`，至少检查：

| 字段 | 必须指向/匹配 |
|---|---|
| `model.onnx_model` | 本次选定的完整或 head-cut ONNX |
| `target_platform` | `rk3576` |
| `do_quantization` | 是否执行 RKNN PTQ |
| `quant.*` | 本次确定的 RKNN dtype、算法、粒度与混合精度策略 |
| `dataset.root/sample_count` | RKNN 校准图片及数量；仅量化时使用 |
| `preprocess.mean/std` | 与训练和板端输入一致 |

### 当前 Basketball 配置状态

当前三个配置的输入尺寸均为 `1024 x 3328`，校准目录也已切回 Basketball，但模型分支仍有区别：

1. `quant.yaml` 量化 `...fp32_raw_headcut_raw.onnx`，activation 为 `int16`、weight 为 `int8`；
2. `compile.yaml` 使用同基名的 head-cut deploy ONNX 和 quant param，输出到 `compile/vs859/`；
3. `eval.yaml` 使用同基名的 head-cut deploy ONNX 和 `yolov8_headcut` 解码；
4. `outputs/quant/` 仍保留少量旧尺寸产物，文件存在不代表它属于本次构建。

因此 VS859 的 `quant -> eval -> compile` 可以按当前 head-cut 基名和尺寸串行执行；量化前 PR
基线另行使用同一 head-cut 模型生成。

建议先检查所有关键路径：

```bash
rg -n "onnx_model|quant_param|^model:|^quantize:|resize|crop|img_size|pre_shape|pre_stride|root:" \
  modes/basketball/configs/{vs859,rk3576}/*.yaml

./run.sh basketball quant --dry-run
./run.sh basketball compile --dry-run
```

`--dry-run` 只证明入口将调用哪个 YAML，不会替你检查 YAML 内部的模型身份是否一致。

## 6. VS859 模型量化

### 6.1 `quant.yaml` 参数说明

当前文件的采样数是 `20`，校准 root 是 Basketball。下面展示的是保留当前 activation
`int16` / weight `int8` 精度策略、但把建议采样目标提高到 `100` 后的**推荐结构**：

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
`modes/basketball/configs/vs859/mixed_precision.yaml`，执行 `quant` 时会自动追加：

```text
--qparam_cfg modes/basketball/configs/vs859/mixed_precision.yaml
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

## 7. VS859 模型评估

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
  --config modes/basketball/configs/vs859/eval.yaml \
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

### 7.3 叠加 PR 曲线

评估器会在 COCOeval 完成后，按 IoU=0.50 的 101 个 recall 采样点生成 PR 曲线。篮球模式已经在
两个平台的 `eval.yaml` 中配置了以下输出：

```text
VS859:   modes/basketball/outputs/evaluation/pr_curve.svg
RK3576:  modes/basketball/outputs/evaluation/rk3576/pr_curve.svg
```

图中红色实线是当前量化模型，蓝色虚线是量化前模型；若提供板端 predictions，则增加绿色点划线。
图中包含 `all` 和每个类别的曲线，并在
标题中标出对应 AP50。先生成与当前 `decode_mode`、输入尺寸一致的量化前预测 JSON，再执行量化模型评估即可。
如果原始 ONNX 与量化模型输出形态一致，可以直接使用：

```bash
./run.sh basketball float-eval
./run.sh basketball eval
./run.sh basketball eval --platform rk3576
```

`float-eval` 会把基线预测写入
`modes/basketball/outputs/evaluation/pre_quant_predictions.json`。如果只想手工生成基线，独立
评估脚本也支持；当前 Basketball head-cut 配置应传入同样的 head-cut 浮点 ONNX：

```bash
python3 common/evaluation/yolo_coco_metric.py \
  --config modes/basketball/configs/vs859/eval.yaml \
  --model modes/basketball/model/<float_headcut_model>.onnx \
  --pred-json modes/basketball/outputs/evaluation/pre_quant_predictions.json
```

如果还要叠加目标板实际运行结果，把板端生成的标准 COCO predictions 数组保存为：

```text
modes/basketball/outputs/evaluation/board_predictions.json
```

再次执行 `eval` 后，SVG 会增加绿色点划线板端曲线，并在每个面板标题中显示 `board=AP50`。
板端 JSON 必须使用同一评估集的 `image_id`、`category_id`、`bbox` 和 `score`；缺少该文件时只
生成量化模型与浮点基线两条曲线。

量化前和量化模型必须使用相同的评估图片、COCO 标注、输入尺寸、decoder、置信度和 NMS 阈值，
否则曲线不能用于直接比较。若基线 JSON 尚不存在，评估仍会生成当前模型的单条曲线，并打印提示。

### 7.4 逐层量化误差

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

### 7.5 评估验收

至少检查：

- 浮点与量化 AP50-95、AP50 的绝对下降和相对下降；
- person、ball、hoop、ballhoop 各类别的样本量和 AP；
- 小球、远处人物、遮挡目标、画面边缘目标是否明显退化；
- 是否出现置信度整体被压低、框偏移、重复框或背景误检；
- compare 最差层是否集中在检测 neck/head；
- 评估集规模是否足以支持当前结论。

只有量化评估通过后，才应把同一组 deploy ONNX 和 quant param 交给编译阶段。

## 8. VS859 平台流程

VS859 使用 Statlas 工具链。模型先由 `quant.yaml` 完成 PTQ，再由 `compile.yaml` 将成对的
deploy ONNX 和 quant param 编译为 MGZ。

### 8.1 VS859 编译前检查

确认 `modes/basketball/configs/vs859/compile.yaml`：

1. `model` 是刚通过评估的 deploy ONNX；
2. `quantize` 是与该 ONNX 同一次 PTQ 生成的 quant param；
3. 两个文件模型基名一致；
4. `pre_shape` 与 ONNX 输入 shape 一致；
5. runtime 输入格式、stride、色彩范围与部署代码一致；
6. mean/std 与 PTQ、评估一致；
7. `output` 位于 `outputs/compile/vs859/`，名称能区分模型版本、尺寸和输入格式。

### 8.2 VS859 执行命令

VS859 是默认平台，下面两条命令等价：

```bash
./run.sh basketball compile
./run.sh basketball compile --platform vs859
```

建议先执行：

```bash
./run.sh basketball compile --platform vs859 --dry-run
```

`run.py` 会把编译器 `lib/` 加入 `LD_LIBRARY_PATH`，然后执行：

```text
${STATLAS_COMPILE_DIR}/StatlasCompile -c modes/basketball/configs/vs859/compile.yaml
```

批量编译当前全部模式：

```bash
for mode in basketball demo_v11 demo_v26 demo_v5 demo_v8 soccer; do
  ./run.sh "$mode" compile --platform vs859
done
```

### 8.3 VS859 编译参数说明

| 字段 | 当前值/形式 | 说明 |
|---|---|---|
| `model` | deploy ONNX 路径 | 必须是本次评估通过的量化 deploy 模型 |
| `quantize` | quant param YAML 路径 | 必须与 `model` 成对，不能混用不同 PTQ 产物 |
| `output` | `outputs/compile/vs859/*.mgz` | 旌平台模型输出路径 |
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

当前 Basketball 的 `model`、`quantize`、`pre_shape` 和 `pre_stride` 均为 head-cut
`h1024_w3328` 链路。修改模型后必须再次检查这四项，不能因为旧 `.mgz` 已存在就跳过检查。

### 8.4 VS859 输出与验收

输出位于：

```text
modes/<mode>/outputs/compile/vs859/<configured-name>.mgz
```

当前 6 个模式均已成功生成 MGZ。

目标板至少验证：

- MGZ 可以加载，输入/输出 tensor 数量、顺序和 dtype 与部署代码一致；
- NV12 buffer 的宽、高、stride 和色彩范围正确；
- host 端按 `_headcut_raw_spec.yaml` 完成 DFL、decode 和 NMS；
- 同一测试图片上，板端结果与 PC Statlas 量化评估结果在可接受误差内；
- 延迟、内存、core 使用和长时间运行稳定性满足部署要求。

MGZ 文件成功生成只代表 Statlas 离线编译通过，不代表 VS859 板端输入、后处理、精度或性能
已经验收。

## 9. RK3576 平台流程

RK3576 使用 RKNN Toolkit2，只读取 `configs/rk3576/rknn.yaml`。它在一次 `build` 中完成
RKNN 量化与模型转换，不读取 `configs/vs859/` 下的任何 YAML，也不使用 Statlas 生成的
deploy ONNX 或 quant param。

### 9.1 RK3576 输入与转换规则

参数来源如下：

| RKNN 参数 | 来源 |
|---|---|
| 输入 ONNX | `model.onnx_model` |
| 是否量化 | `do_quantization` |
| 量化 dtype | `quant.dtype` |
| 校准算法 | `quant.algorithm` |
| 量化粒度 | `quant.method` |
| 混合精度 | `quant.hybrid_level/auto_hybrid` |
| 校准图片 | `dataset.root/sample_count` |
| 归一化 | `preprocess.mean/std`，转换器按 RKNN 规则换算 |
| 编译优化 | `build.optimization_level/float_dtype` |
| 目标平台 | `target_platform`，并与命令行 `--platform` 交叉检查 |
| 输出 | `outputs/compile/rk3576/<mode>_rk3576.rknn` |

当前 `rknn.yaml` 量化策略字段：

| 字段 | Toolkit2 2.3.2 支持值/说明 |
|---|---|
| `do_quantization` | `true` 执行 PTQ，`false` 生成浮点 RKNN |
| `quant.dtype` | `w8a8`、`w8a16`、`w16a16i`、`w16a16i_dfp`、`w4a16` |
| `quant.algorithm` | `normal`、`mmse`、`kl_divergence`、`gdq` |
| `quant.method` | `layer`、`channel` 或 `group32` 至 `group256` |
| `quant.hybrid_level` | Toolkit2 混合精度等级；当前为 `0` |
| `quant.auto_hybrid` | 是否在 `rknn.build` 开启自动混合精度 |
| `build.optimization_level` | Toolkit2 优化等级；当前为 `3` |
| `build.float_dtype` | 非量化路径的数据类型；Toolkit2 2.3.2 当前支持 `float16` |

Basketball 当前配置示例：

```yaml
model:
  onnx_model: modes/basketball/model/<model>_headcut_raw.onnx

target_platform: rk3576
do_quantization: true

quant:
  dtype: w8a8
  algorithm: normal
  method: channel
  hybrid_level: 0
  auto_hybrid: false

build:
  optimization_level: 3
  float_dtype: float16

dataset:
  root: modes/basketball/datasets/calibration/images
  sample_count: 20

preprocess:
  mean: [0.0, 0.0, 0.0]
  std: [1.0, 1.0, 1.0]
```

当前涉及的 YOLOv5/v8/v11/26 模型都在 `rknn.yaml` 中选择去头 ONNX，并显式设置
`do_quantization: true`、`dtype: w8a8`、`algorithm: normal` 和 `method: channel`。转换器仍会检查
模型结构，避免重复裁切。YOLO26 不能直接量化原生 `[1, 8, 69888]` 打包输出，因为坐标和类别
概率会共享输出量化尺度；专用裁切器把它改成按 stride 8/16/32 排列的六个 box/class 输出后再做
INT8。其 box 分支是四通道直接 `ltrb` 距离，不使用 YOLOv8/11 的 DFL 解码。

RKNN 转换不会修改用于 Statlas 的源 ONNX。若模型含 `vsdeploy::Silu`，转换器会在模型
同目录生成 `_rknn.onnx`，仅为 RKNN 展开成标准 `Sigmoid + Mul`。

### 9.2 RK3576 执行命令

单模式量化并生成 RKNN：

```bash
./run.sh basketball quant --platform rk3576 --dry-run
./run.sh basketball quant --platform rk3576
```

RKNN Toolkit2 在一次 `build` 中同时完成量化和模型生成，因此下面的 `compile` 写法与上述
`quant` 写法执行同一条 RKNN 转换流程，保留用于统一两平台的编译命令：

```bash
./run.sh basketball compile --platform rk3576 --dry-run
./run.sh basketball compile --platform rk3576
```

批量转换当前全部模式：

```bash
for mode in basketball demo_v11 demo_v26 demo_v5 demo_v8 soccer; do
  ./run.sh "$mode" compile --platform rk3576
done
```

`--platform` 对 `quant`、`eval`、`compare` 和 `compile` 操作生效。不要先运行 Statlas `quant` 来为 RKNN 准备模型；
RKNN 路径使用的是 `configs/rk3576/rknn.yaml:model.onnx_model`。

### 9.3 RK3576 模式说明

当前支持以下 6 个模式：

| 模式 | 模型/输入尺寸 | RKNN 输入结构 | RKNN 精度 |
|---|---|---|---|
| `basketball` | YOLOv8, 1024 x 3328 | 去头 6 输出 | int8 |
| `demo_v11` | YOLOv11, 1024 x 3328 | 去头 6 输出 | int8 |
| `demo_v26` | YOLO26, 1024 x 3328 | 去头 6 输出（直接 ltrb + class logits） | int8 |
| `demo_v5` | YOLOv5, 704 x 1280 | 去头 3 输出 | int8 |
| `demo_v8` | YOLOv8, 960 x 960 | 去头 6 输出 | int8 |
| `soccer` | YOLOv5, 1024 x 3328 | 清洗后去头 3 输出 | int8 |

当前 6 个模式均已成功生成 RKNN。输出目录结构为：

```text
modes/<mode>/outputs/compile/rk3576/
├── <mode>_rk3576.rknn
├── <mode>_rk3576.json
└── <mode>_rk3576.dataset.txt       # int8 模型生成
```

### 9.4 RK3576 COCO 评估

各模式的 `configs/rk3576/eval.yaml` 独立描述 RKNN 模型、测试集、输入尺寸、解码方式和输出：

| 字段 | 作用 |
|---|---|
| `model` | 已导出的 `.rknn` 路径 |
| `build_config` | PC 模拟器重建模型时使用的 `rknn.yaml` |
| `dataset.ann_file/img_dir` | COCO 标注和测试图片目录 |
| `dataset.input_size/color_order` | RKNN 输入高宽和 RGB/BGR 顺序 |
| `decode.mode` | `yolov5_headcut`、`yolov8_headcut`、`yolo26_headcut` 或 `yolov8_raw` |
| `decode.conf_threshold/iou_threshold` | 置信度阈值和逐类别 NMS 阈值 |
| `decode.anchors` | YOLOv5 三尺度 anchor，仅 YOLOv5 使用 |
| `decode.class_map` | 模型类别下标到 COCO category id 的映射 |
| `output.*` | 指标、预测 JSON 和画框图片的输出位置 |

PC 模拟器评估：

```bash
./run.sh basketball eval --platform rk3576 --dry-run
./run.sh basketball eval --platform rk3576
```

RKNN Toolkit2 的 PC 模拟器不能直接加载已经导出的 `.rknn`。因此该命令会读取
`build_config`，按同一份 `rknn.yaml` 从 ONNX 和校准集在内存中重新 `build`，再执行推理。
它适合验证转换配置、解码和指标流程，但不是对 `.rknn` 文件本身逐字节验收。

连接 RK3576 目标板后，直接评估已经导出的 `.rknn`：

```bash
./run.sh basketball eval --platform rk3576 \
  --runtime-target rk3576 --device-id <设备ID>
```

输出位于：

```text
modes/<mode>/outputs/evaluation/rk3576/
├── metric_result.csv
├── metric_result.txt
├── predictions.json
└── visualizations/
```

当前 `basketball`、`demo_v11`、`demo_v26`、`demo_v5` 和 `demo_v8` 复用篮球 COCO
评估集。仓库中暂时没有足球专用 COCO 标注，因此 `soccer` 的评估配置也临时使用篮球测试集，
并只映射 `person` 和 `sports ball` 类；该结果只能用于流程联调，不能作为足球模型正式精度结论。

### 9.5 RK3576 逐层量化误差

PC 模拟器分析：

```bash
./run.sh basketball compare --platform rk3576 --dry-run
./run.sh basketball compare --platform rk3576
```

默认使用 `rknn.yaml:dataset.root` 按文件名排序后的第一张校准图片。也可以显式指定一张有代表性的
图片或 NPY 输入：

```bash
./run.sh basketball compare --platform rk3576 \
  --analysis-input /path/to/representative.jpg
```

连接 RK3576 后分析 NPU 各层实际输出：

```bash
./run.sh basketball compare --platform rk3576 \
  --runtime-target rk3576 --device-id <设备ID> \
  --analysis-input /path/to/representative.jpg
```

该操作按 `rknn.yaml` 从 ONNX 重新 build，以便 Toolkit2 同时取得浮点和量化中间输出；它不会
导出或覆盖 `outputs/compile/rk3576/*.rknn`。结果按算法和运行目标隔离：

```text
modes/<mode>/outputs/evaluation/rk3576/accuracy_analysis/
└── <normal|mmse|kl_divergence|gdq>/
    ├── simulator/
    └── rk3576/
```

Toolkit2 会在对应目录生成逐层快照和量化误差结果，本项目另写入 `analysis.json` 记录模型、配置、
代表图片、算法和目标设备。逐层张量可能占用较多磁盘空间；分析完成后可用
`./run.sh <mode> clean --scope eval` 清理。`do_quantization: false` 的浮点 RKNN 没有量化误差，
该命令会直接停止。

### 9.6 RK3576 输出与验收

目标板至少验证：

- RKNN 可以加载，输入/输出 tensor 数量、顺序、layout 和 dtype 与后处理一致；
- RKNN runtime 的颜色顺序、resize 和 mean/std 与 `rknn.yaml` 及 ONNX 输入 shape 一致；
- YOLOv5 三输出或 YOLOv8/11/26 六输出的 stride 顺序与对应 C++ 后处理一致；
- YOLO26 按 `box8, cls8, box16, cls16, box32, cls32` 解析，类别分支做 sigmoid，框分支按直接
  `ltrb` 距离解码，不执行 DFL；
- 同一测试图片上，板端结果与浮点/量化基线在可接受误差内；
- 延迟、内存和长时间运行稳定性满足部署要求。

RKNN 文件成功生成只代表 Toolkit2 离线转换通过，不代表 RK3576 板端输入、后处理、精度或性能
已经验收。

## 10. 状态、清理与重跑

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
旧版本遗留在 `outputs/compile/` 根目录的产物不会被平台命令覆盖，也不应作为本次构建结果交付。

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

## 11. 常见问题

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

## 12. 交付检查清单

1. `model/` 中本次原始 ONNX 身份明确，清洗和 head-cut 产物可追溯。
2. 校准集与部署分布一致，且与训练/评估集没有泄漏。
3. 新增评估图片已完整合并到 COCO JSON，类别 ID、bbox 和图片尺寸正确。
4. `quant.yaml` 的模型、校准 root、采样数、输入尺寸和归一化正确。
5. 全局 activation/weight 位宽和可选逐层混合精度配置经过评估验证。
6. deploy ONNX 与 quant param 来自同一次 PTQ。
7. `eval.yaml`、`compare.yaml`、`compile.yaml` 已改成同一模型基名和输入尺寸。
8. 浮点 AP、量化 AP、逐类指标、框图和 compare 报告均已检查并记录。
9. `compile.yaml` 的 NV12 格式、range、shape、stride、mean/std 与 runtime 一致。
10. `rknn.yaml` 的模型、量化策略、校准集和 mean/std 已确认并记录在 manifest。
11. `.mgz` 和 `.rknn` 分别位于 `compile/vs859/` 与 `compile/rk3576/`，没有混用旧根目录产物。
12. `.mgz` 已在 VS859、`.rknn` 已在 RK3576 目标板完成加载、精度、性能和稳定性验证。

## 13. Git 管理约定

- 提交代码、配置、文档、COCO 标注，以及团队约定需要版本化的数据清单；
- 原始 ONNX、量化产物、评估结果、可视化、日志、`.mgz` 和 `.rknn` 通常不提交，可由流程重建；
- 提交前检查 YAML 中是否残留仅适用于个人机器的绝对路径；
- 不要通过提交旧产物来掩盖配置链路不一致，模型身份应能由配置和构建记录追溯。
