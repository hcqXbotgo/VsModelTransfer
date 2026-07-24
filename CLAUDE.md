# CLAUDE.md — quant_folder

本仓用于将 ONNX 检测模型量化、评估、编译为旌平台（Statlas）模型。支持 `soccer` 和 `demo` 两种
运动模式，每个模式自包含 model/datasets/configs/outputs。通用 loader 与 metric 放在 `common/`。

## 入口

所有操作通过 `./run.sh <mode> <operation>`：

```bash
./run.sh --list                    # 列出可用模式
./run.sh <mode> status             # 查看文件计数
./run.sh <mode> validate           # 校验数据/路径
./run.sh <mode> quant              # PTQ 量化（→ outputs/quant/）
./run.sh <mode> eval               # 量化模型 AP（→ outputs/evaluation/metric_result.csv）
./run.sh <mode> float-eval         # 原始 ONNX AP 基线（→ outputs/evaluation/float_visualizations/）
./run.sh <mode> visualize          # 量化模型对 draft 图画框（不计 AP）
./run.sh <mode> float-visualize    # 原始 ONNX 对 draft 图画框
./run.sh <mode> compare            # 逐层余弦误差（→ outputs/evaluation/compare/）
./run.sh <mode> compile            # 编译为平台模型（DFL 头模型自动用切头版）
./run.sh <mode> cut-head            # 切检测头（v8/v11；v5 为 no-op，-> outputs/quant/*_headcut_*.onnx）
./run.sh <mode> all                # quant + eval + float-eval + compile
./run.sh <mode> <op> --dry-run     # 只打印命令不执行
```

`run.sh` 会 `source env.sh` 拿到本机的 `STATLAS_PYTHON` / `STATLAS_QUANT` / `STATLAS_COMPILE_DIR`。
首次拉取仓库后：`cp env.example.sh env.sh` 再编辑。

## 关键路径

- 入口脚本：`run.py`、`run.sh`
- 共用评估：`common/evaluation/yolo_coco_metric.py`（YOLO+AOC COCO metric）、
  `common/evaluation/coco_loader.py`、`common/evaluation/summarize_compare.py`
- 模式根：`modes/<mode>/{model,datasets,configs,outputs,docs}`
- 量化产物：`modes/<mode>/outputs/quant/<basename>_deploy_model.onnx` + `..._quant_param.yaml`
- 评估产物：`modes/<mode>/outputs/evaluation/metric_result.csv`、`compare/{layer_compare*.csv,REPORT.md}`、
  `{float_,}visualizations/`
- 模式文档：`modes/soccer/docs/PIPELINE.md`（含数据/量化/编译全流程）、`modes/demo/docs/README.md`

## 量化损失评估（2026-07-16 实测）

同一图片、同一前处理下，对比原始 ONNX（`float-eval`）与量化模型（`eval`）的 COCO AP，以及
`compare` 在一张代表图上的逐层余弦误差（`cosine_error = 1 - cos_sim`）。

### Soccer（yolov5n_falcon2_3328_1024_clean，11 张正式评估图，球场 5 张校准，检测头 FP16+其余 INT8）

| 模型 | AP50-95 | AP50 |
|---|---:|---:|
| 原始 ONNX | 0.3452 | 0.4368 |
| 量化 | 0.0881 | 0.2576 |
| **绝对下降** | **−0.2571** | **−0.1792** |
| **相对下降** | **−74.5%** | **−41.0%** |

逐层余弦误差（185 层，代表图 `calibration/images/clip-0_frame-5266.jpg`）：
mean=0.0139、median=0.0127、p90=0.0288、p99=0.0463、worst=0.0575。
误差最大的层集中在检测 neck/head（`onnx::Concat_450`、`onnx::Concat_339`、`input.104` 等
Split/Concat/Silu 输出）。完整排名见 `modes/soccer/outputs/evaluation/compare/REPORT.md`。

### Demo（yolov5s，COCO val2017 抽 10 张，ImageNet 20 张校准，检测头 FP16+其余 INT8）

| 模型 | AP50-95 | AP50 |
|---|---:|---:|
| 原始 ONNX | 0.4177 | 0.5871 |
| 量化 | 0.3318 | 0.5520 |
| **绝对下降** | **−0.0859** | **−0.0351** |
| **相对下降** | **−20.6%** | **−6.0%** |

逐层余弦误差（242 层，代表图 `calibration/images/ILSVRC2012_val_00002251.JPEG`）：
mean=0.0151、median=0.0103、p90=0.0355、p99=0.0523、worst=0.0729。
误差最大层为 `/24/Split_2_output_2`、`/8/cv2/act/Mul_output_0` 等 early/mid neck 的 Split 与
SiLU·Mul 激活输出。完整排名见 `modes/demo/outputs/evaluation/compare/REPORT.md`。

### 结论

- **Demo 量化健康**：AP50 仅掉 6%，逐层误差分布与典型 INT8 PTQ 一致，主产物可直接用。
- **Soccer 损失偏大**：AP50-95 掉 ~75%、AP50 掉 ~41%。原因：
  1. **评估集只有 11 张 COCO 真值**，AP 方差大；且 3328×1024 的极宽分辨率 + 远距离小目标对
     量化误差更敏感。
  2. **校准集只有 5 张球场图**，分布覆盖不足（PIPELINE.md 已记录该问题，建议扩到 100~500 张）。
  3. **检测 neck 的 Split/Concat/Silu 层误差显著**，需要继续排查是否应将更多早期层纳入 FP16。
- 重新跑过量化后请把新的 float/quant AP 行同步到 `modes/soccer/docs/PIPELINE.md` 的「当前状态」表。

## 跑全流程

```bash
cd /home/dragonfly/wj_sdk/quant_folder

# 一次性跑完 soccer 量化+评估+float 基线+编译
./run.sh soccer all

# 只补 float 基线（其余产物已存在时）
./run.sh soccer float-eval
./run.sh demo   float-eval

# 只重做逐层比较
./run.sh soccer compare
./run.sh demo   compare
```

`compare` 的代表图通过 `modes/<mode>/configs/compare.yaml` 的 `Compare.input_file` 指定；
`layer_dump: false` 默认不落盘中间 NPY（3328×1024 单图张量可达数 GB）。

## 添加/审核数据（仅 soccer）

```bash
./run.sh soccer add-calibration /path/to/calibration_images   # 加校准图
./run.sh soccer import-eval                                    # 把 draft 合并到正式 evaluation
./run.sh soccer validate                                       # 入箱后必须校验
```

draft 路径：`modes/soccer/datasets/draft/{images,annotations/instances.json}`。
正式评估路径：`modes/soccer/datasets/evaluation/{images,annotations/instances.json}`。
**draft 永远不能直接用来算 AP**，必须人工审核后通过 `import-eval` 合并。

## 量化关键约束（soccer）

- 检测头坐标与置信度动态范围差距巨大，共用 INT8 scale 会清零置信度 → 必须使用
  `configs/mixed_precision.yaml` 的检测头 FP16 混合精度配置。
- 校准、评估必须用相同前处理：RGB、`[0,1]`、`mean=[0,0,0]`、`std=[1,1,1]`、resize/crop 到
  模型输入尺寸（soccer 为 `[1024, 3328]`，demo 为 `640`）。
- 改动校准图、observer、位宽或模型后必须重跑 `quant → eval → float-eval → compare` 并记录对比。

## 编译：YOLOv8/v11 检测头切头（basketball 等 DFL 头模型）

YOLOv8/v11 检测头在展平的网格轴上做 DFL（`Softmax`）+ dist2bbox
（`Slice`/`Sub`/`Div`）解码，Statlas 编译器对这些算子无法 tile，会 core-dump
（`time step assign init_group_data_secs failed`）。YOLOv5 头是加性解码
（`Mul`/`Pow`/`Add`），能整模型内联编译。

- `quant` 量化后**自动**对 DFL 头切头，产出 `outputs/quant/<model>_headcut_deploy_model.onnx`
  + `*_headcut_deploy_model_spec.yaml`（host 解码规格）。切头是结构化检测，对 v5 是安全 no-op。
- `compile` 自动检测并改用切头模型编译；`.mgz` 输出的是 6 个原始 4D 特征图，**DFL+解码+NMS
  必须在 host 端做**（reg_max/stride/nc 见 spec）。`eval`/`float-eval` 走完整 ONNX 算 AP，不受影响。
- 详见 `modes/basketball/docs/COMPILE.md`、切头脚本 `common/tools/cut_yolov8_head.py`。
