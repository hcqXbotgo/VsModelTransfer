#!/usr/bin/env python3
"""Dispatch head cutting to the supported YOLO model-family implementation."""
import argparse
import sys
from pathlib import Path

import onnx
from onnx import shape_inference

import cut_yolov5_head
import cut_yolov8_head
import cut_yolo26_head


def cut_head(input_path, output_path):
    model = shape_inference.infer_shapes(onnx.load(str(input_path)))
    if cut_yolov5_head.find_cut_points(model):
        print('[cut_yolo_head] detected YOLOv5 three-scale Detect head')
        return cut_yolov5_head.cut_head(input_path, output_path)

    print('[cut_yolo_head] trying YOLOv8/11 DFL head')
    if cut_yolov8_head.cut_head(str(input_path), str(output_path)):
        return True
    print('[cut_yolo_head] trying YOLO26 NMS-free head')
    if cut_yolo26_head.cut_head(str(input_path), str(output_path)):
        return True
    outputs = list(model.graph.output)
    if len(outputs) == 1 and len(
            outputs[0].type.tensor_type.shape.dim) == 3:
        print('[cut_yolo_head] native packed/NMS-free output; no head cut needed')
        return None
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--input_model', required=True)
    parser.add_argument('--output_model', required=True)
    args = parser.parse_args()
    input_path = Path(args.input_model)
    output_path = Path(args.output_model)
    if not input_path.is_file():
        sys.exit('input not found: {}'.format(input_path))
    result = cut_head(input_path, output_path)
    if result is False:
        sys.exit('no supported YOLOv5 or YOLOv8/11 head found')


if __name__ == '__main__':
    main()
