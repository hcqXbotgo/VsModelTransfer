#!/usr/bin/env python3
"""Thin wrapper around statlas_quant.onnx_convert.OnnxConvertTool.

This module lives in common/tools/ so the same wrapper is reused across
all modes. It runs OnnxConvertTool (vendored inside statlas_quant) and
prints useful diagnostics on failure.

Usage:
    python clean_model.py --input_model raw.onnx --output_model raw_clean.onnx
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='OnnxConvertTool wrapper.')
    parser.add_argument('--input_model', required=True, type=str)
    parser.add_argument('--output_model', required=True, type=str)
    parser.add_argument('--opset_version', type=int, default=None)
    args = parser.parse_args()

    if not Path(args.input_model).exists():
        sys.exit('input not found: {}'.format(args.input_model))

    try:
        from statlas_quant.onnx_convert import main as oct_main
    except ImportError as exc:
        sys.exit('statlas_quant not available: {}'.format(exc))

    # statlas_quant.onnx_convert reads sys.argv directly; replicate what its
    # CLI does so we can call the underlying function without argv hacking.
    from statlas_quant.onnx_utils import onnx_convert
    print('[clean_model] input:  {}'.format(args.input_model))
    print('[clean_model] output: {}'.format(args.output_model))
    onnx_convert(args.input_model, args.output_model, args.opset_version, {})
    print('[clean_model] done. size: {:.1f} MB'.format(
        os.path.getsize(args.output_model) / 1024 / 1024))


if __name__ == '__main__':
    main()
