# Demo 模式

- 模型：`../model/yolov5s.onnx`
- 校准集：`../datasets/calibration/images`（20 张）
- 评估集：`../datasets/evaluation`（COCO val2017）
- 配置：`../configs`
- 产物：`../outputs`

校准和评估统一使用 RGB、`640×640`、`[0,1]`、`mean=[0,0,0]`、
`std=[1,1,1]`。

```bash
./run.sh demo quant
./run.sh demo eval
./run.sh demo float-eval
./run.sh demo visualize
./run.sh demo compile
```

2026-07-15 对齐前处理后的 10 张固定样本结果：

| 模型 | AP50-95 | AP50 |
|---|---:|---:|
| 原始 ONNX | 0.4357 | 0.5963 |
| INT8 + 检测头 FP16 | 0.3318 | 0.5520 |
