# Demo 模式

- 模型：`../model/yolov5s.onnx`
- 校准集：`../datasets/calibration/images`（20 张）
- 评估集：`../datasets/evaluation`（COCO val2017）
- 配置：`../configs`
- 产物：`../outputs`

```bash
./run.sh demo quant
./run.sh demo eval
./run.sh demo visualize
./run.sh demo compile
```
