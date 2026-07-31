# Basketball YOLOv8 编译与检测头切头

basketball 模型是 YOLOv8n，实际输入 `[1, 3, 1024, 1024]`（文件名里的
`h1920_w1920` 是导出时的训练尺寸，部署/量化配置统一用 `1024×3328`，见
`configs/quant.yaml`）。

## 为什么必须切头

YOLOv8 的检测头在展平的网格轴上做 **DFL（`Softmax`）+ dist2bbox
（`Slice`/`Sub`/`Div`）** 解码。以 1024×3328 为例，三个尺度网格合计 69888：

```
detect-branch Conv -> [1, C, Hg, Wg]   (4D, NPU 可 tile)
        |
   Reshape(4D->3D) [1, C, Hg*Wg]        ← 检测头入口
        |
   跨尺度 Concat -> DFL Softmax -> Slice/Sub/Div 解码   ← NPU 无法 tile
        |
   最终 Concat -> [1, 8, 69888]
```

Statlas 编译器对 `Softmax`/`Transpose`/`Slice`/`Sub`/`Div` 沿这个展平轴无法
tile，会报：

```
Reshape can't tile in dim 2 ...
ConcatOp::backwardNCHW, can't tile in axis with other op now.
TransposeOp::backwardNCHW fail ...
time step assign init_group_data_secs failed for group N
```

随后 core-dump（`optimze_level` 0/1/2 均失败，已实测）。对照之下 soccer 的
YOLOv5 头是**加性解码**（`Mul`/`Pow`/`Add`），与 backbone 同族算子，同尺寸
69888 能正常编译。

## 显式切头与量化

切头、量化和编译是相互独立的操作，推荐按顺序显式执行：

```bash
./run.sh basketball cut-head
./run.sh basketball quant
./run.sh basketball eval
./run.sh basketball compile
```

- `cut-head` 只在 `model/` 下生成 `<model>_headcut_raw.onnx` 和对应的
  decode spec，不修改任何 YAML。
- `quant` 只量化 `configs/quant.yaml` 中 `model.onnx_model` 指定的模型。
- `eval` 只使用 `configs/eval.yaml` 中指定的量化模型和 qparam。
- `compile` 只使用 `configs/compile.yaml` 中指定的模型和 qparam。

因此，重新切头或更换模型后，必须显式确认三个配置文件中的路径指向预期产物。

切头脚本 `common/tools/cut_yolov8_head.py` 是**结构化检测**（按 4D->3D 空间展平
的 Reshape 模式 + DFL 解码算子判定），不依赖节点命名，因此对 YOLOv8/v11 都适用。
对 YOLOv5（5D 重塑，不匹配模式）是安全 no-op，不会产出文件。

## 切头后的输出

切在 6 个检测分支 conv 输出处，输出 6 个 4D NCHW 特征图（每尺度 box + cls）：

| 输出 | 形状 | 含义 |
|---|---|---|
| `cv2.0/cv2.0.2` | `[1, 64, 128, 416]` | stride 8 box（4×reg_max=64） |
| `cv3.0/cv3.0.2` | `[1, 4, 128, 416]` | stride 8 cls（nc=4） |
| `cv2.1/cv2.1.2` | `[1, 64, 64, 208]` | stride 16 box |
| `cv3.1/cv3.1.2` | `[1, 4, 64, 208]` | stride 16 cls |
| `cv2.2/cv2.2.2` | `[1, 64, 32, 104]` | stride 32 box |
| `cv3.2/cv3.2.2` | `[1, 4, 32, 104]` | stride 32 cls |

精确的输出名、reg_max、stride、nc 见 `*_headcut_raw_spec.yaml`。

## host 端解码（必须）

`.mgz` 输出的是原始特征图，**DFL + 解码 + NMS 必须在 host 端做**：

1. box 分支：`[1, 4*reg_max, H, W]` -> reshape `[1, 4, reg_max, H*W]` ->
   transpose `[1, reg_max, 4, H*W]` -> softmax(reg_max 维) -> 加权求和得
   `[1, 4, H*W]`（dist）。
2. cls 分支：`[1, nc, H, W]` -> sigmoid。
3. dist2bbox：dist × stride + anchor 网格 -> xyxy（stride 8/16/32）。
4. 三尺度拼成 `[N, 4+nc]` -> NMS。

reg_max=16、nc=4、stride {8,16,32}，详见 spec 文件。

## 评估

`eval`（量化评估）跑的是**切头 deploy ONNX**：StatlasQuant 量化推理出 6 个量化
特征图，`yolo_coco_metric.py` 的 `yolov8_headcut` 解码器在 host 端做 DFL + dist2bbox
+ NMS（`common/evaluation/yolo_coco_metric.py::_decode_yolov8_headcut`）。这和部署链路
完全一致，所以 `eval` 的 AP = 真实部署精度。

`run.sh basketball eval` 不检测或替换模型，完全使用
`modes/basketball/configs/eval.yaml`。评估切头模型时，该配置必须显式指向对应的
deploy ONNX 和 qparam，并使用 `decode_mode: yolov8_headcut`。

`float-eval`（浮点基线）跑**原始 FP32 完整 ONNX**（带 DFL 头，FP32 解码），是浮点
上界。注意 `eval`（量化切头 + host 解码）和 `float-eval`（FP32 完整）口径不同：
前者是部署精度，后者是 FP32 上界。

对比参考（实测）：完整量化模型 + INT8 头解码（旧口径）AP50=0.482；切头 + host FP32
解码（新口径）AP50=0.692（2 图小样本，趋势是切头 host 解码更高，因消除了 INT8
Softmax/解码损失）。完整 19 图数字见 `metric_result.csv`。
