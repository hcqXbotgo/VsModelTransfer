#!/usr/bin/env python3
"""Single entry point for every sports mode operation."""
import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODES_ROOT = ROOT / 'modes'
DEFAULT_QUANT = Path('StatlasQuant')
DEFAULT_PYTHON = Path('python3')
DEFAULT_COMPILER_ROOT = (
    ROOT.parent / 'VS859' / 'VS859_ED_release' / 'tools' / 'NPU' / 'statlas')


def executable(env_name, fallback, command_name=None):
    configured = os.environ.get(env_name)
    if configured:
        return Path(configured)
    if Path(fallback).exists():
        return Path(fallback)
    if command_name:
        found = shutil.which(command_name)
        if found:
            return Path(found)
    return Path(fallback)


def available_modes():
    if not MODES_ROOT.exists():
        return []
    return sorted(path.name for path in MODES_ROOT.iterdir()
                  if path.is_dir() and (path / 'configs').is_dir())


def require(path, description):
    if not Path(path).exists():
        raise SystemExit('{} not found: {}'.format(description, path))
    return Path(path)


def show_command(command):
    print('+', ' '.join(shlex.quote(str(item)) for item in command), flush=True)


def run_command(command, dry_run=False, env=None):
    show_command(command)
    if not dry_run:
        subprocess.run([str(item) for item in command], cwd=str(ROOT),
                       env=env, check=True)


def config(mode, name):
    return require(MODES_ROOT / mode / 'configs' / '{}.yaml'.format(name),
                   '{} {} config'.format(mode, name))


def quant_command(mode):
    """Build the StatlasQuant PTQ command for the configured model."""
    quant = require(executable('STATLAS_QUANT', DEFAULT_QUANT, 'StatlasQuant'),
                    'StatlasQuant')
    cmd = [quant, '--quant_cfg', config(mode, 'quant')]
    mp_path = MODES_ROOT / mode / 'configs' / 'mixed_precision.yaml'
    if mp_path.exists():
        cmd += ['--qparam_cfg', mp_path]
    return cmd


def run_quant(mode, dry_run):
    """Quantize exactly the model selected by the mode's quant.yaml."""
    run_command(quant_command(mode), dry_run)


def eval_command(mode, operation='eval', cfg=None):
    quant = require(executable('STATLAS_QUANT', DEFAULT_QUANT, 'StatlasQuant'),
                    'StatlasQuant')
    cfg = cfg or config(mode, operation)
    return [quant, '--quant_cfg', cfg]


def original_model(mode):
    """Return the single original ONNX export for float operations."""
    return raw_model_path(mode)


def deploy_model(mode):
    """The quantized deploy ONNX produced by `quant` (excludes head-cut)."""
    qdir = MODES_ROOT / mode / 'outputs' / 'quant'
    candidates = sorted(p for p in qdir.glob('*_deploy_model.onnx')
                        if 'headcut' not in p.name)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit('No deploy model found under {} (run quant first)'.format(qdir))
    names = ', '.join(p.name for p in candidates)
    raise SystemExit('Multiple deploy models under {}: {}'.format(qdir, names))


def headcut_raw_model(mode):
    """Deterministic output path for the explicit ``cut-head`` operation."""
    raw = raw_model_path(mode)
    return raw.with_name(raw.stem + '_headcut_raw.onnx')


# Suffixes that indicate an ONNX is already a processed/cleaned version,
# not a "raw" export that needs cleaning.
PROCESSED_SUFFIXES = (
    '_headcut_raw.onnx',
    '_clean.onnx',
    '_calibrated_model.onnx',
    '_deploy_model.onnx',
    '_simplified.onnx',
    '_opset13.onnx',
    '_fp32.onnx',
)


def raw_model_path(mode):
    """Find the raw (unprocessed) ONNX under modes/<mode>/model/.

    Skips files with known processed suffixes (e.g. xxx_clean.onnx) so
    OnnxConvertTool isn't run twice on the same file.
    """
    model_dir = MODES_ROOT / mode / 'model'
    candidates = []
    for p in sorted(model_dir.glob('*.onnx')):
        if any(p.name.endswith(suf) for suf in PROCESSED_SUFFIXES):
            continue
        if p.name.startswith('_') or p.name.startswith('.'):
            continue
        candidates.append(p)
    if len(candidates) == 0:
        raise SystemExit(
            'No raw ONNX found under {}/ (all files look already cleaned)'.format(
                model_dir))
    if len(candidates) > 1:
        names = ', '.join(p.name for p in candidates)
        raise SystemExit(
            'Multiple raw ONNX found under {}: {}. Move extras to a subfolder '
            'or rename with a processed suffix.'.format(model_dir, names))
    return candidates[0]


def clean_model_command(mode):
    """Build OnnxConvertTool command to clean the raw ONNX.

    Output goes to <raw_basename>_clean.onnx next to the input.
    """
    convert_tool = require(
        executable('STATLAS_PYTHON', DEFAULT_PYTHON, 'python3'),
        'python (from StatlasQuant env)')
    script = require(ROOT / 'common' / 'tools' / 'clean_model.py',
                     'clean_model helper script')
    raw = raw_model_path(mode)
    clean = raw.with_name(raw.stem + '_clean.onnx')
    return [convert_tool, script, '--input_model', raw,
            '--output_model', clean], raw, clean


def clean_model(mode, dry_run):
    command, raw, clean = clean_model_command(mode)
    print('raw:   {}'.format(raw))
    print('clean: {}'.format(clean))
    run_command(command, dry_run)


def cut_head_command(mode, input_path=None, output_path=None):
    """Build the head-cut command for the original ONNX.

    Cuts the DFL decode head off the original model, producing a headless
    ONNX (raw 4D feature maps) that is used as PTQ input. Pass
    input_path/output_path to cut an arbitrary ONNX.

    No-op for non-DFL heads (YOLOv5): the script detects the 4D->3D DFL
    reshape pattern and writes nothing, exiting 0.
    """
    python = require(executable('STATLAS_PYTHON', DEFAULT_PYTHON, 'python3'),
                     'Python')
    script = require(ROOT / 'common' / 'tools' / 'cut_yolov8_head.py',
                     'cut_yolov8_head helper script')
    if input_path is None:
        input_path = raw_model_path(mode)
    if output_path is None:
        output_path = headcut_raw_model(mode)
    return [python, script, '--input_model', input_path,
            '--output_model', output_path], input_path, output_path


def cut_head(mode, dry_run):
    command, raw, headcut = cut_head_command(mode)
    print('raw:     {}'.format(raw))
    print('headcut: {}'.format(headcut))
    run_command(command, dry_run)


def float_eval_command(mode):
    python = require(executable('STATLAS_PYTHON', DEFAULT_PYTHON, 'python3'),
                     'Python')
    evaluator = require(ROOT / 'common' / 'evaluation' / 'yolo_coco_metric.py',
                        'float ONNX evaluator')
    vis_dir = (MODES_ROOT / mode / 'outputs' / 'evaluation' /
               'float_visualizations')
    return [python, evaluator, '--config', config(mode, 'eval'),
            '--model', original_model(mode), '--num', '0',
            '--vis-dir', vis_dir]


def compare_commands(mode):
    quant = require(executable('STATLAS_QUANT', DEFAULT_QUANT, 'StatlasQuant'),
                    'StatlasQuant')
    python = require(executable('STATLAS_PYTHON', DEFAULT_PYTHON, 'python3'),
                     'Python')
    summarizer = require(ROOT / 'common' / 'evaluation' / 'summarize_compare.py',
                         'layer compare summarizer')
    output = MODES_ROOT / mode / 'outputs' / 'evaluation' / 'compare'
    raw_csv = output / 'layer_compare.csv'
    sorted_csv = output / 'layer_compare_sorted.csv'
    report = output / 'REPORT.md'
    return (
        [quant, '--quant_cfg', config(mode, 'compare')],
        [python, summarizer, '--input', raw_csv, '--output', sorted_csv,
         '--report', report],
    )


def compile_yaml(mode):
    """Return the mode's explicit compiler configuration."""
    return config(mode, 'compile')


def eval_yaml(mode):
    """Return the mode's explicit quantized-evaluation configuration."""
    return config(mode, 'eval')


def compile_command(mode, cfg=None):
    compiler_root = Path(os.environ.get('STATLAS_COMPILE_DIR',
                                        DEFAULT_COMPILER_ROOT))
    compiler = require(compiler_root / 'StatlasCompile', 'StatlasCompile')
    cfg = cfg or config(mode, 'compile')
    return [compiler, '-c', cfg], compiler_root


def run_compile(mode, dry_run):
    """Compile exactly the model and qparam selected by compile.yaml."""
    command, compiler_root = compile_command(mode, compile_yaml(mode))
    env = os.environ.copy()
    old_path = env.get('LD_LIBRARY_PATH', '')
    env['LD_LIBRARY_PATH'] = '{}{}'.format(
        compiler_root / 'lib', ':' + old_path if old_path else '')
    run_command(command, dry_run, env=env)


def run_all(mode, dry_run):
    """Run configured operations without implicitly generating model files."""
    run_quant(mode, dry_run)
    run_command(eval_command(mode), dry_run)
    run_command(float_eval_command(mode), dry_run)
    run_compile(mode, dry_run)


def validate(mode, dry_run):
    mode_root = MODES_ROOT / mode
    dataset_tool = mode_root / 'tools' / 'dataset.py'
    annotations = mode_root / 'datasets' / 'evaluation' / 'annotations' / 'instances.json'
    images = mode_root / 'datasets' / 'evaluation' / 'images'
    if dataset_tool.exists():
        python = require(executable('STATLAS_PYTHON', DEFAULT_PYTHON, 'python3'),
                         'Python')
        run_command([python, dataset_tool, 'validate',
                     '--annotations', annotations, '--images', images], dry_run)
        return

    # Modes without a custom dataset manager still get basic path checks.
    eval_cfg = config(mode, 'eval')
    print('config:', eval_cfg)
    print('calibration images:',
          len(list((mode_root / 'datasets' / 'calibration' / 'images').glob('*'))))
    print('evaluation images:',
          len(list((mode_root / 'datasets' / 'evaluation' / 'images').glob('*'))))


def print_status(mode):
    root = MODES_ROOT / mode
    print('mode:', mode)
    for relative in ('model', 'datasets/calibration', 'datasets/evaluation',
                     'datasets/draft', 'configs', 'outputs/quant',
                     'outputs/evaluation', 'outputs/compile'):
        path = root / relative
        if path.exists():
            files = sum(1 for item in path.rglob('*') if item.is_file())
            print('  {:28s} {} file(s)'.format(relative, files))


def clean_mode(mode, scope='quant', dry_run=False):
    """Remove generated products.

    scope:
      - 'quant': only outputs/quant/ (default, safest)
      - 'eval':  only outputs/evaluation/ subdirs (metric, compare, visualizations)
      - 'compile': only outputs/compile/
      - 'all':  everything under outputs/
    """
    root = MODES_ROOT / mode / 'outputs'

    scope_map = {
        'quant': ['quant'],
        'eval': ['evaluation/metric_result.csv',
                 'evaluation/compare',
                 'evaluation/visualizations',
                 'evaluation/float_visualizations',
                 'evaluation/draft_visualizations',
                 'evaluation/visualize_only',
                 'evaluation/draft_float_visualizations'],
        'compile': ['compile'],
        'all': ['quant', 'evaluation', 'compile'],
    }
    targets = scope_map.get(scope, [scope])

    removed = []
    for rel in targets:
        path = root / rel
        if not path.exists():
            continue
        if path.is_file():
            removed.append((path, 'file', path.stat().st_size))
        else:
            n_files = sum(1 for _ in path.rglob('*') if _.is_file())
            total_size = sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
            removed.append((path, f'dir({n_files} files)', total_size))

    print(f'mode: {mode}    scope: {scope}')
    if not removed:
        print('  (nothing to remove)')
        return

    for path, kind, size in removed:
        size_str = f'{size/1024/1024:.1f} MB' if size >= 1024*1024 else f'{size/1024:.1f} KB'
        print(f'  [{kind}] {path}  ({size_str})')

    if dry_run:
        print('  (--dry-run, not deleted)')
        return

    for path, _, _ in removed:
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    print(f'  ✓ cleaned')


def main():
    modes = available_modes()
    parser = argparse.ArgumentParser(
        description='Run quantization workflow by sports mode.')
    parser.add_argument('mode', nargs='?', choices=modes)
    parser.add_argument(
        'operation', nargs='?',
        choices=('quant', 'eval', 'float-eval', 'compare', 'compile', 'cut-head',
                 'validate', 'status', 'all', 'clean', 'clean-model'))
    parser.add_argument('--list', action='store_true', help='List available modes')
    parser.add_argument('--dry-run', action='store_true', help='Print only')
    parser.add_argument('--scope', default='quant',
                        choices=('quant', 'eval', 'compile', 'all'),
                        help='clean scope (only used by clean operation)')
    args = parser.parse_args()

    if args.list:
        print('\n'.join(modes))
        return
    if not args.mode or not args.operation:
        parser.error('mode and operation are required (or use --list)')

    if args.operation == 'quant':
        run_quant(args.mode, args.dry_run)
    elif args.operation == 'eval':
        run_command(eval_command(args.mode, 'eval', eval_yaml(args.mode)),
                    args.dry_run)
    elif args.operation == 'float-eval':
        run_command(float_eval_command(args.mode), args.dry_run)
    elif args.operation == 'compare':
        for command in compare_commands(args.mode):
            run_command(command, args.dry_run)
    elif args.operation == 'compile':
        run_compile(args.mode, args.dry_run)
    elif args.operation == 'cut-head':
        cut_head(args.mode, args.dry_run)
    elif args.operation == 'validate':
        validate(args.mode, args.dry_run)
    elif args.operation == 'status':
        print_status(args.mode)
    elif args.operation == 'clean':
        clean_mode(args.mode, args.scope, args.dry_run)
    elif args.operation == 'clean-model':
        clean_model(args.mode, args.dry_run)
    elif args.operation == 'all':
        run_all(args.mode, args.dry_run)


if __name__ == '__main__':
    main()
