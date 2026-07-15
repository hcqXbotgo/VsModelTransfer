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
    quant = require(executable('STATLAS_QUANT', DEFAULT_QUANT, 'StatlasQuant'),
                    'StatlasQuant')
    return [quant, '--quant_cfg', config(mode, 'quant'),
            '--qparam_cfg', config(mode, 'mixed_precision')]


def eval_command(mode, operation='eval'):
    quant = require(executable('STATLAS_QUANT', DEFAULT_QUANT, 'StatlasQuant'),
                    'StatlasQuant')
    return [quant, '--quant_cfg', config(mode, operation)]


def compile_command(mode):
    compiler_root = Path(os.environ.get('STATLAS_COMPILE_DIR',
                                        DEFAULT_COMPILER_ROOT))
    compiler = require(compiler_root / 'StatlasCompile', 'StatlasCompile')
    return [compiler, '-c', config(mode, 'compile')], compiler_root


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


def main():
    modes = available_modes()
    parser = argparse.ArgumentParser(
        description='Run quantization workflow by sports mode.')
    parser.add_argument('mode', nargs='?', choices=modes)
    parser.add_argument(
        'operation', nargs='?',
        choices=('quant', 'eval', 'visualize', 'compile', 'validate',
                 'status', 'all', 'add-calibration',
                 'import-eval'))
    parser.add_argument('paths', nargs='*', help='Image paths for add operations')
    parser.add_argument('--source', default='manual_coco_annotation',
                        help='Annotation source recorded by import-eval')
    parser.add_argument('--list', action='store_true', help='List available modes')
    parser.add_argument('--dry-run', action='store_true', help='Print only')
    args = parser.parse_args()

    if args.list:
        print('\n'.join(modes))
        return
    if not args.mode or not args.operation:
        parser.error('mode and operation are required (or use --list)')

    if args.operation == 'quant':
        run_command(quant_command(args.mode), args.dry_run)
    elif args.operation in ('eval', 'visualize'):
        run_command(eval_command(args.mode, args.operation), args.dry_run)
    elif args.operation == 'compile':
        command, compiler_root = compile_command(args.mode)
        env = os.environ.copy()
        old_path = env.get('LD_LIBRARY_PATH', '')
        env['LD_LIBRARY_PATH'] = '{}{}'.format(
            compiler_root / 'lib', ':' + old_path if old_path else '')
        run_command(command, args.dry_run, env=env)
    elif args.operation == 'validate':
        validate(args.mode, args.dry_run)
    elif args.operation == 'status':
        print_status(args.mode)
    elif args.operation == 'add-calibration':
        add_calibration(args.mode, args.paths, args.dry_run)
    elif args.operation == 'import-eval':
        import_eval(args.mode, args.paths, args.source, args.dry_run)
    elif args.operation == 'all':
        run_command(quant_command(args.mode), args.dry_run)
        run_command(eval_command(args.mode), args.dry_run)
        command, compiler_root = compile_command(args.mode)
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = str(compiler_root / 'lib') + (
            ':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        run_command(command, args.dry_run, env=env)


if __name__ == '__main__':
    main()
