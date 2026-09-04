#!/usr/bin/env python3
"""Local browser UI for the quant_folder workflow.

The service binds to localhost, does not expose YAML contents, and accepts
only known workflow operations. Uploaded ONNX files are stored in the chosen
mode's model directory and their references are updated in internal configs.
"""

import argparse
import cgi
import json
import os
import re
import shutil
import subprocess
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent
PLATFORMS = ('vs859', 'rk3576', 'ambarella')
OPERATIONS = ('clean-model', 'cut-head', 'quant', 'compile', 'eval',
              'float-eval', 'compare', 'validate', 'status', 'clean')
PIPELINE_LIMIT = 8
UPLOAD_LIMIT = 4 * 1024 * 1024 * 1024
jobs = {}
jobs_lock = threading.Lock()


def safe_mode(value):
    if not value or Path(value).name != value:
        raise ValueError('invalid mode')
    path = ROOT / 'modes' / value
    if not path.is_dir() or not (path / 'configs').is_dir():
        raise ValueError('unknown mode')
    return value


def safe_platform(value):
    if value not in PLATFORMS:
        raise ValueError('unsupported platform')
    return value


def json_response(handler, status, value):
    payload = json.dumps(value, ensure_ascii=False).encode('utf-8')
    handler.send_response(status)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.send_header('Content-Length', str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def update_key(path, key, value):
    """Replace a YAML scalar while retaining comments and formatting."""
    if not path.is_file():
        return False
    text = path.read_text(encoding='utf-8')
    pattern = re.compile(r'^(\s*' + re.escape(key) + r':\s*).*$')
    updated, count = pattern.subn(lambda match: match.group(1) + value,
                                  text, count=1)
    if count and updated != text:
        path.write_text(updated, encoding='utf-8')
        return True
    return False


def update_model_configs(mode, filename):
    """Point platform configs at a newly uploaded ONNX model."""
    model_rel = 'modes/{}/model/{}'.format(mode, filename)
    stem = Path(filename).stem
    quant_dir = 'modes/{}/outputs/quant/'.format(mode)
    changed = []
    config_root = ROOT / 'modes' / mode / 'configs'
    replacements = {
        'vs859/quant.yaml': {'onnx_model': model_rel},
        'vs859/eval.yaml': {
            'onnx_model': quant_dir + stem + '_deploy_model.onnx',
            'quant_param': quant_dir + stem + '_quant_param.yaml',
        },
        'vs859/compile.yaml': {
            'model': quant_dir + stem + '_deploy_model.onnx',
            'quantize': quant_dir + stem + '_quant_param.yaml',
            'output': 'modes/{}/outputs/compile/vs859/{}.mgz'.format(mode, stem),
        },
        'vs859/compare.yaml': {
            'onnx_model': quant_dir + stem + '_deploy_model.onnx',
            'quant_param': quant_dir + stem + '_quant_param.yaml',
        },
        'rk3576/rknn.yaml': {'onnx_model': model_rel},
        'ambarella/compile.yaml': {'model': model_rel},
    }
    for relative, keys in replacements.items():
        path = config_root / relative
        for key, value in keys.items():
            if update_key(path, key, value):
                changed.append(str(path.relative_to(ROOT)))
    return sorted(set(changed))


def run_pipeline(job_id, mode, operations, platform):
    try:
        for index, operation in enumerate(operations, 1):
            command = [str(ROOT / 'run.sh'), mode, operation,
                       '--platform', platform]
            with jobs_lock:
                jobs[job_id]['current_operation'] = operation
                jobs[job_id]['command'] = command
                jobs[job_id]['logs'] += '\n=== [{}/{}] {} ===\n'.format(
                    index, len(operations), operation)
            process = subprocess.Popen(
                command, cwd=str(ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                env=os.environ.copy())
            with jobs_lock:
                jobs[job_id]['pid'] = process.pid
            for line in process.stdout:
                with jobs_lock:
                    jobs[job_id]['logs'] += line
            return_code = process.wait()
            if return_code != 0:
                with jobs_lock:
                    jobs[job_id]['status'] = 'failed'
                    jobs[job_id]['returncode'] = return_code
                    jobs[job_id]['failed_operation'] = operation
                return
        with jobs_lock:
            jobs[job_id]['status'] = 'succeeded'
            jobs[job_id]['returncode'] = 0
    except Exception as error:
        with jobs_lock:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['logs'] += '{}\n'.format(error)


class Handler(BaseHTTPRequestHandler):
    server_version = 'QuantFolderWeb/2.0'

    def log_message(self, format_string, *args):
        return

    def send_file(self, path, content_type):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == '/':
                return self.send_file(STATIC / 'index.html',
                                      'text/html; charset=utf-8')
            if parsed.path == '/api/modes':
                modes = sorted(path.name for path in (ROOT / 'modes').iterdir()
                               if path.is_dir() and (path / 'configs').is_dir())
                return json_response(self, 200, {
                    'modes': modes, 'platforms': PLATFORMS,
                    'operations': OPERATIONS,
                })
            if parsed.path.startswith('/api/jobs/'):
                job_id = parsed.path.rsplit('/', 1)[-1]
                with jobs_lock:
                    job = jobs.get(job_id)
                    if not job:
                        raise FileNotFoundError('job not found')
                    return json_response(self, 200, dict(job))
            self.send_error(404)
        except (ValueError, FileNotFoundError) as error:
            json_response(self, 400, {'error': str(error)})
        except Exception as error:
            json_response(self, 500, {'error': str(error)})

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == '/api/upload':
                return self.upload_model()
            if parsed.path == '/api/jobs':
                return self.start_job()
            self.send_error(404)
        except (ValueError, FileNotFoundError) as error:
            json_response(self, 400, {'error': str(error)})
        except Exception as error:
            json_response(self, 500, {'error': str(error)})

    def upload_model(self):
        length = int(self.headers.get('Content-Length', '0'))
        if length <= 0 or length > UPLOAD_LIMIT:
            raise ValueError('upload must be between 1 byte and 4 GiB')
        content_type = self.headers.get('Content-Type', '')
        if not content_type.startswith('multipart/form-data'):
            raise ValueError('upload must use multipart/form-data')
        form = cgi.FieldStorage(
            fp=self.rfile, headers=self.headers,
            environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': content_type,
                     'CONTENT_LENGTH': str(length)}, keep_blank_values=False)
        mode = safe_mode(form.getfirst('mode'))
        item = form['model'] if 'model' in form else None
        if not item or not getattr(item, 'filename', None):
            raise ValueError('missing model file')
        filename = Path(item.filename).name
        if not filename.lower().endswith('.onnx'):
            raise ValueError('only .onnx model files are accepted')
        if not re.fullmatch(r'[A-Za-z0-9._-]+', filename):
            raise ValueError('model filename may contain only letters, digits, dot, underscore and hyphen')
        destination = ROOT / 'modes' / mode / 'model' / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and form.getfirst('overwrite') != 'true':
            raise ValueError('model already exists; select overwrite to replace it')
        # Copy in bounded chunks so a large ONNX does not require one
        # allocation and the request can make steady progress.
        with destination.open('wb') as output:
            shutil.copyfileobj(item.file, output, length=1024 * 1024)
        changed = update_model_configs(mode, filename)
        return json_response(self, 201, {
            'model': str(destination.relative_to(ROOT)),
            # Return only a count; configuration paths and contents stay private.
            'updated_config_count': len(changed),
        })

    def start_job(self):
        length = int(self.headers.get('Content-Length', '0'))
        if length <= 0 or length > 32 * 1024:
            raise ValueError('invalid request body')
        body = json.loads(self.rfile.read(length).decode('utf-8'))
        mode = safe_mode(body.get('mode'))
        platform = safe_platform(body.get('platform'))
        operations = body.get('operations')
        if not isinstance(operations, list) or not operations:
            raise ValueError('select at least one operation')
        if len(operations) > PIPELINE_LIMIT:
            raise ValueError('too many operations')
        if any(operation not in OPERATIONS for operation in operations):
            raise ValueError('unsupported operation in pipeline')
        with jobs_lock:
            running = next((item for item in jobs.values()
                            if item['status'] == 'running'), None)
            if running:
                raise ValueError('another job is running: {}'.format(running['id']))
            job_id = uuid.uuid4().hex[:12]
            job = {'id': job_id, 'status': 'running', 'mode': mode,
                   'platform': platform, 'operations': operations,
                   'current_operation': None, 'failed_operation': None,
                   'logs': '', 'returncode': None, 'pid': None, 'command': None}
            jobs[job_id] = job
        threading.Thread(target=run_pipeline,
                         args=(job_id, mode, operations, platform),
                         daemon=True).start()
        return json_response(self, 202, job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print('Quant folder UI: http://{}:{}/'.format(args.host, args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
