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


def quant_command(mode, model_path=None):
    """Build the StatlasQuant PTQ command.

    If model_path is given, the quant config's onnx_model is overridden to it
    (used when running PTQ on a head-cut raw model). Otherwise the config is
    used as-is.
    """
    quant = require(executable('STATLAS_QUANT', DEFAULT_QUANT, 'StatlasQuant'),
                    'StatlasQuant')
    cmd = [quant, '--quant_cfg', config(mode, 'quant')]
    mp_path = MODES_ROOT / mode / 'configs' / 'mixed_precision.yaml'
    if mp_path.exists():
        cmd += ['--qparam_cfg', mp_path]
    return cmd


def maybe_cut_head_raw(mode, dry_run):
    """Cut the head off the original ONNX before PTQ, if it is a DFL head.

    Writes ``model/<base>_headcut_raw.onnx``. Returns True if PTQ should run
    on the head-cut model (i.e. the cut produced a file), False if PTQ should
    run on the original model unchanged (YOLOv5, or cut was a no-op).
    """
    raw = original_model(mode)
    headcut_raw = headcut_raw_model(mode)
    if dry_run:
        print('(would cut head off {} -> {})'.format(raw.name, headcut_raw.name))
        return headcut_raw.exists()
    # Always re-cut (cheap) so the head-cut raw tracks the current original.
    cmd, _, _ = cut_head_command(mode, input_path=raw, output_path=headcut_raw)
    try:
        run_command(cmd, False)
    except SystemExit:
        return False
    produced = headcut_raw.exists() and headcut_raw.stat().st_size > 0
    return produced


def eval_command(mode, operation='eval', cfg=None):
    quant = require(executable('STATLAS_QUANT', DEFAULT_QUANT, 'StatlasQuant'),
                    'StatlasQuant')
    cfg = cfg or config(mode, operation)
    return [quant, '--quant_cfg', cfg]


def original_model(mode):
    models = sorted((MODES_ROOT / mode / 'model').glob('*.onnx'))
    if len(models) == 1:
        return models[0]

    quant_cfg = MODES_ROOT / mode / 'configs' / 'quant.yaml'
    if quant_cfg.exists():
        for line in quant_cfg.read_text(encoding='utf-8').splitlines():
            if line.strip().startswith('onnx_model:'):
                value = line.split(':', 1)[1].strip().strip('"\'')
                selected = Path(value)
                if not selected.is_absolute():
                    selected = ROOT / selected
                if selected.exists():
                    return selected


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
    """Path of the head-cut *raw* model (under model/), used as PTQ input.

    ``quant`` cuts the head off the original ONNX *before* running StatlasQuant
    when this is a DFL-head (v8/v11) model, so the deploy model produced by PTQ
    is already headless and compiles directly. For YOLOv5 (no DFL head) the cut
    is a no-op and this file is not created.
    """
    raw = original_model(mode)
    return raw.with_name(raw.stem + '_headcut_raw.onnx')


# Suffixes that indicate an ONNX is already a processed/cleaned version,
# not a "raw" export that needs cleaning.
PROCESSED_SUFFIXES = (
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
        input_path = original_model(mode)
    if output_path is None:
        output_path = headcut_raw_model(mode)
    return [python, script, '--input_model', input_path,
            '--output_model', output_path], input_path, output_path


def cut_head(mode, dry_run):
    command, raw, headcut = cut_head_command(mode)
    print('raw:     {}'.format(raw))
    print('headcut: {}'.format(headcut))
    run_command(command, dry_run)


def float_command(mode, operation):
    python = require(executable('STATLAS_PYTHON', DEFAULT_PYTHON, 'python3'),
                     'Python')
    evaluator = require(ROOT / 'common' / 'evaluation' / 'yolo_coco_metric.py',
                        'float ONNX evaluator')
    config_name = 'eval' if operation == 'float-eval' else 'visualize'
    vis_name = ('float_visualizations' if operation == 'float-eval'
                else 'draft_float_visualizations')
    vis_dir = MODES_ROOT / mode / 'outputs' / 'evaluation' / vis_name
    return [python, evaluator, '--config', config(mode, config_name),
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
    """Resolve the compile config to use.

    ``quant`` cuts the head off the original ONNX *before* PTQ, so the deploy
    model produced by PTQ is already headless (raw 4D feature maps) and
    compiles directly. The mode's compile.yaml is used unchanged.
    """
    return config(mode, 'compile')


def eval_yaml(mode):
    """Resolve the eval config to use.

    ``quant`` cuts the head off the original ONNX *before* PTQ, so the deploy
    model produced by PTQ is already headless (raw 4D feature maps). `eval`
    runs that headless deploy model with the host-side DFL/dist2bbox decode
    (decode_mode: yolov8_headcut in eval.yaml). The mode's eval.yaml is used
    unchanged.
    """
    return config(mode, 'eval')


def compile_command(mode, cfg=None):
    compiler_root = Path(os.environ.get('STATLAS_COMPILE_DIR',
                                        DEFAULT_COMPILER_ROOT))
    compiler = require(compiler_root / 'StatlasCompile', 'StatlasCompile')
    cfg = cfg or config(mode, 'compile')
    return [compiler, '-c', cfg], compiler_root


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


def add_calibration(mode, paths, dry_run):
    if not paths:
        raise SystemExit('add-calibration requires one or more image paths')
    tool = require(MODES_ROOT / mode / 'tools' / 'dataset.py',
                   '{} dataset manager'.format(mode))
    python = require(executable('STATLAS_PYTHON', DEFAULT_PYTHON, 'python3'),
                     'Python')
    dataset_root = MODES_ROOT / mode / 'datasets'
    run_command([python, tool, 'add', '--root', dataset_root,
                 '--kind', 'calibration'] + [Path(item) for item in paths], dry_run)


def import_eval(mode, paths, source, dry_run):
    mode_root = MODES_ROOT / mode
    if not paths:
        annotations = mode_root / 'datasets' / 'draft' / 'annotations' / 'instances.json'
        images = mode_root / 'datasets' / 'draft' / 'images'
    elif len(paths) == 2:
        annotations = Path(paths[0])
        images = Path(paths[1])
    else:
        raise SystemExit(
            'import-eval accepts either no paths (use draft) or: '
            '<COCO annotations.json> <images directory>')
    tool = require(MODES_ROOT / mode / 'tools' / 'dataset.py',
                   '{} dataset manager'.format(mode))
    python = require(executable('STATLAS_PYTHON', DEFAULT_PYTHON, 'python3'),
                     'Python')
    annotations = require(annotations, 'reviewed COCO annotations')
    images = require(images, 'reviewed image directory')
    dataset_root = MODES_ROOT / mode / 'datasets'
    run_command([python, tool, 'import-coco', '--root', dataset_root,
                 '--annotations', annotations, '--images', images,
                 '--source', source], dry_run)
    if not dry_run:
        validate(mode, False)


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
        choices=('quant', 'eval', 'visualize', 'float-eval',
                 'float-visualize', 'compare', 'compile', 'cut-head',
                 'validate', 'status', 'all', 'add-calibration',
                 'import-eval', 'clean', 'clean-model'))
    parser.add_argument('paths', nargs='*', help='Image paths for add operations')
    parser.add_argument('--source', default='manual_coco_annotation',
                        help='Annotation source recorded by import-eval')
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
        # Pre-PTQ head cut for DFL-head (v8/v11) models: cut the head off the
        # original ONNX first, then run PTQ on the head-cut model so the deploy
        # model is already headless and compiles directly. No-op for YOLOv5.
        headcut_raw = headcut_raw_model(args.mode)
        use_headcut = maybe_cut_head_raw(args.mode, args.dry_run)
        if use_headcut and not args.dry_run:
            # Temporarily point quant.yaml's onnx_model at the head-cut raw.
            quant_yaml = config(args.mode, 'quant')
            text = quant_yaml.read_text(encoding='utf-8')
            original_text = text
            raw_line = '  onnx_model: {}'.format(headcut_raw.relative_to(ROOT))
            new_lines = []
            for line in text.splitlines():
                if line.strip().startswith('onnx_model:'):
                    new_lines.append(raw_line)
                else:
                    new_lines.append(line)
            quant_yaml.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
            try:
                run_command(quant_command(args.mode), args.dry_run)
            finally:
                quant_yaml.write_text(original_text, encoding='utf-8')
        else:
            run_command(quant_command(args.mode), args.dry_run)
        # No post-PTQ cut: if PTQ ran on the head-cut raw, the deploy model is
        # already headless. For YOLOv5 (no DFL head) the pre-cut was a no-op
        # and the deploy model has no DFL head to cut anyway.
    elif args.operation in ('eval', 'visualize'):
        if args.operation == 'eval':
            cfg = eval_yaml(args.mode)
        else:
            cfg = config(args.mode, args.operation)
        run_command(eval_command(args.mode, args.operation, cfg), args.dry_run)
    elif args.operation in ('float-eval', 'float-visualize'):
        run_command(float_command(args.mode, args.operation), args.dry_run)
    elif args.operation == 'compare':
        for command in compare_commands(args.mode):
            run_command(command, args.dry_run)
    elif args.operation == 'compile':
        command, compiler_root = compile_command(args.mode, compile_yaml(args.mode))
        env = os.environ.copy()
        old_path = env.get('LD_LIBRARY_PATH', '')
        env['LD_LIBRARY_PATH'] = '{}{}'.format(
            compiler_root / 'lib', ':' + old_path if old_path else '')
        run_command(command, args.dry_run, env=env)
    elif args.operation == 'cut-head':
        cut_head(args.mode, args.dry_run)
    elif args.operation == 'validate':
        validate(args.mode, args.dry_run)
    elif args.operation == 'status':
        print_status(args.mode)
    elif args.operation == 'add-calibration':
        add_calibration(args.mode, args.paths, args.dry_run)
    elif args.operation == 'import-eval':
        import_eval(args.mode, args.paths, args.source, args.dry_run)
    elif args.operation == 'clean':
        clean_mode(args.mode, args.scope, args.dry_run)
    elif args.operation == 'clean-model':
        clean_model(args.mode, args.dry_run)
    elif args.operation == 'all':
        # quant already cuts the head pre-PTQ for DFL-head models.
        headcut_raw = headcut_raw_model(args.mode)
        use_headcut = maybe_cut_head_raw(args.mode, args.dry_run)
        if use_headcut and not args.dry_run:
            quant_yaml = config(args.mode, 'quant')
            original_text = quant_yaml.read_text(encoding='utf-8')
            new_lines = []
            for line in original_text.splitlines():
                if line.strip().startswith('onnx_model:'):
                    new_lines.append('  onnx_model: {}'.format(
                        headcut_raw.relative_to(ROOT)))
                else:
                    new_lines.append(line)
            quant_yaml.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
            try:
                run_command(quant_command(args.mode), args.dry_run)
            finally:
                quant_yaml.write_text(original_text, encoding='utf-8')
        else:
            run_command(quant_command(args.mode), args.dry_run)
        run_command(eval_command(args.mode), args.dry_run)
        run_command(float_command(args.mode, 'float-eval'), args.dry_run)
        command, compiler_root = compile_command(
            args.mode, compile_yaml(args.mode) if not args.dry_run
            else config(args.mode, 'compile'))
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = str(compiler_root / 'lib') + (
            ':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        run_command(command, args.dry_run, env=env)


if __name__ == '__main__':
    main()
